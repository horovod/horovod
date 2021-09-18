// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================

#include "gloo_context.h"

#include <chrono>
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "gloo/allgather.h"
#include "gloo/barrier.h"
#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/file_store.h"
#include "gloo/rendezvous/prefix_store.h"

#ifdef __linux__
#include "gloo/transport/tcp/device.h"
using attr = gloo::transport::tcp::attr;
constexpr auto CreateDevice = gloo::transport::tcp::CreateDevice;
#else
// Use uv on macOS as TCP requires epoll (Linux-only)
#include "gloo/transport/uv/device.h"
using attr = gloo::transport::uv::attr;
constexpr auto CreateDevice = gloo::transport::uv::CreateDevice;
#endif

#if HAVE_MPI
#include "gloo/mpi/context.h"
#endif

#include "http_store.h"
#include "memory_store.h"
#include "../utils/env_parser.h"

namespace horovod {
namespace common {

int ParseNextInt(std::stringstream& ss) {
  assert(ss.good());

  std::string substr;
  getline(ss, substr, ',');

  return (int) std::strtol(substr.c_str(), nullptr, 10);
}

std::chrono::milliseconds GetTimeoutFromEnv() {
  auto s = std::chrono::seconds(GetIntEnvOrDefault(HOROVOD_GLOO_TIMEOUT_SECONDS, 30));
  return std::chrono::duration_cast<std::chrono::milliseconds>(s);
}

std::shared_ptr<gloo::Context> Rendezvous(const std::string& prefix,
                                          const char* server_addr_env, int server_port,
                                          int rank, int size,
                                          std::shared_ptr<gloo::transport::Device>& dev,
                                          std::chrono::milliseconds timeout) {
  std::unique_ptr<GlooStore> store;
  if (server_addr_env != nullptr) {
    std::string server_addr = server_addr_env;
    store.reset(new HTTPStore(server_addr, server_port, prefix, rank));
  } else {
    store.reset(new MemoryStore());
  }
  LOG(DEBUG) << prefix << " rendezvous started for rank=" << rank << ", size=" << size
             << ", dev={" << dev->str() << "}, timeout="
             << std::to_string(std::chrono::duration_cast<std::chrono::seconds>(timeout).count());

  auto context = std::make_shared<gloo::rendezvous::Context>(rank, size);
  context->setTimeout(timeout);
  context->connectFullMesh(*store, dev);
  {
    // Don't finalize the store until all clients have had a chance to connect.
    gloo::BarrierOptions opts(context);
    opts.setTimeout(timeout);
    gloo::barrier(opts);
  }
  store->Finalize();
  return context;
}

#if HAVE_MPI
void GlooContext::InitializeFromMPI(MPIContext& mpi_ctx,
                                    const std::string& gloo_iface) {
  if (!enabled_) {
    return;
  }

  // TODO(sihan): Add support for multiple interfaces:
  //  https://github.com/facebookincubator/gloo/issues/190
  gloo_iface_ = gloo_iface;
  attr device_attr;
  device_attr.iface = gloo_iface_;
  device_attr.ai_family = AF_UNSPEC;
  auto dev = CreateDevice(device_attr);
  timeout_ = GetTimeoutFromEnv();

  auto context =
      std::make_shared<gloo::mpi::Context>(mpi_ctx.GetMPICommunicator(GLOBAL));
  context->setTimeout(timeout_);
  context->connectFullMesh(dev);
  ctx = context;

  global_ctx = ctx;

  auto cross_context =
      std::make_shared<gloo::mpi::Context>(mpi_ctx.GetMPICommunicator(CROSS));
  cross_context->setTimeout(timeout_);
  cross_context->connectFullMesh(dev);
  cross_ctx = cross_context;

  auto local_context =
      std::make_shared<gloo::mpi::Context>(mpi_ctx.GetMPICommunicator(LOCAL));
  local_context->setTimeout(timeout_);
  local_context->connectFullMesh(dev);
  local_ctx = local_context;
}
#endif

void GlooContext::Initialize(const std::string& gloo_iface) {
  if (!enabled_) {
    return;
  }

  // Create a device for communication
  // TODO(sihan): Add support for multiple interfaces:
  //  https://github.com/facebookincubator/gloo/issues/190
  gloo_iface_ = gloo_iface;
  attr device_attr;
  device_attr.iface = gloo_iface_;

  device_attr.ai_family = AF_UNSPEC;
  auto dev = CreateDevice(device_attr);
  timeout_ = GetTimeoutFromEnv();

  auto host_env = std::getenv(HOROVOD_HOSTNAME);
  hostname_ = host_env != nullptr ? std::string(host_env) : std::string("localhost");

  int rank = GetIntEnvOrDefault(HOROVOD_RANK, 0);
  int size = GetIntEnvOrDefault(HOROVOD_SIZE, 1);
  int local_rank = GetIntEnvOrDefault(HOROVOD_LOCAL_RANK, 0);
  int local_size = GetIntEnvOrDefault(HOROVOD_LOCAL_SIZE, 1);
  int cross_rank = GetIntEnvOrDefault(HOROVOD_CROSS_RANK, 0);
  int cross_size = GetIntEnvOrDefault(HOROVOD_CROSS_SIZE, 1);

  auto rendezvous_addr_env = std::getenv(HOROVOD_GLOO_RENDEZVOUS_ADDR);
  auto rendezvous_port = GetIntEnvOrDefault(HOROVOD_GLOO_RENDEZVOUS_PORT, -1);
  if (rendezvous_addr_env != nullptr) {
    LOG(DEBUG) << "rendezvous server address: " << rendezvous_addr_env;
  } else {
    LOG(DEBUG) << "no rendezvous server provided, assuming single process execution";
  }

  bool elastic = GetBoolEnvOrDefault(HOROVOD_ELASTIC, false);
  if (elastic && reset_) {
    LOG(DEBUG) << "elastic mode reinitialization started, reset rank=" << rank << " size=" << size;
    std::string server_addr = rendezvous_addr_env;
    std::string scope = HOROVOD_GLOO_GET_RANK_AND_SIZE;
    HTTPStore init_store(server_addr, rendezvous_port, scope, rank);

    auto key = hostname_ + ":" + std::to_string(local_rank);
    std::vector<char> result = init_store.get(key);
    std::string s(result.begin(), result.end());
    std::stringstream ss(s);

    int last_rank = rank;
    int last_size = size;
    int last_local_rank = local_rank;
    int last_local_size = local_size;
    int last_cross_rank = cross_rank;
    int last_cross_size = cross_size;

    rank = ParseNextInt(ss);
    if (rank == -1) {
      // Signals that this host is not part of the job
      std::ostringstream out;
      out << hostname_ << "[" << local_rank << "] has been removed from elastic job";
      throw std::runtime_error(out.str());
    }

    size = ParseNextInt(ss);
    local_rank = ParseNextInt(ss);
    local_size = ParseNextInt(ss);
    cross_rank = ParseNextInt(ss);
    cross_size = ParseNextInt(ss);

    SetEnv(HOROVOD_RANK, std::to_string(rank).c_str());
    SetEnv(HOROVOD_SIZE, std::to_string(size).c_str());
    SetEnv(HOROVOD_LOCAL_RANK, std::to_string(local_rank).c_str());
    SetEnv(HOROVOD_LOCAL_SIZE, std::to_string(local_size).c_str());
    SetEnv(HOROVOD_CROSS_RANK, std::to_string(cross_rank).c_str());
    SetEnv(HOROVOD_CROSS_SIZE, std::to_string(cross_size).c_str());
    LOG(DEBUG) << "elastic mode reinitialization complete, updated" <<
                  " rank: " << last_rank << " -> " << rank <<
                  " size: " << last_size << " -> " << size <<
                  " local_rank: " << last_local_rank << " -> " << local_rank <<
                  " local_size: " << last_local_size << " -> " << local_size <<
                  " cross_rank: " << last_cross_rank << " -> " << cross_rank <<
                  " cross_size: " << last_cross_size << " -> " << cross_size;
  }

  ctx = Rendezvous(HOROVOD_GLOO_GLOBAL_PREFIX,
                   rendezvous_addr_env, rendezvous_port,
                   rank, size, dev, timeout_);
  LOG(DEBUG) << "Global Gloo context initialized.";

  global_ctx = ctx;

  local_ctx = Rendezvous(HOROVOD_GLOO_LOCAL_PREFIX + hostname_,
                         rendezvous_addr_env, rendezvous_port,
                         local_rank, local_size, dev, timeout_);
  LOG(DEBUG) << "Local Gloo context initialized.";

  cross_ctx = Rendezvous(HOROVOD_GLOO_CROSS_PREFIX + std::to_string(local_rank),
                         rendezvous_addr_env, rendezvous_port,
                         cross_rank, cross_size, dev, timeout_);
  LOG(DEBUG) << "Cross-node Gloo context initialized.";
}

namespace {

// Returns the global rank of each process in sub_ctx
std::vector<int>
EnumerateSubRanks(const std::shared_ptr<gloo::Context>& global_ctx,
                  const std::shared_ptr<gloo::Context>& sub_ctx) {
  std::vector<int> result;
  if (sub_ctx != nullptr) {
    auto global_rank = global_ctx->rank;
    auto sub_rank = sub_ctx->rank;
    auto sub_size = sub_ctx->size;
    result.resize((size_t)sub_size);
    result[sub_rank] = global_rank;
    {
      gloo::AllgatherOptions opts(sub_ctx);
      opts.setInput(&global_rank, 1);
      opts.setOutput(result.data(), sub_size);
      gloo::allgather(opts);
    }
  }
  return result;
}

std::vector<int> RanksIntersection(std::vector<int> ranks,
                                   std::vector<int> target_ranks) {
  std::sort(ranks.begin(), ranks.end());
  std::sort(target_ranks.begin(), target_ranks.end());
  std::vector<int> result;
  std::set_intersection(ranks.begin(), ranks.end(), target_ranks.begin(),
                        target_ranks.end(), std::back_inserter(result));
  return result;
}

} // namespace

void GlooContext::InitializeForProcessSet(const GlooContext& global_context,
                                          const std::vector<int>& registered_ranks) {
  assert(global_context.IsEnabled());

  auto rendezvous_addr_env = std::getenv(HOROVOD_GLOO_RENDEZVOUS_ADDR);
  auto rendezvous_port = GetIntEnvOrDefault(HOROVOD_GLOO_RENDEZVOUS_PORT, -1);
  if (rendezvous_addr_env != nullptr) {
    LOG(DEBUG) << "rendezvous server address: " << rendezvous_addr_env;
  } else {
    LOG(DEBUG) << "no rendezvous server provided, assuming single process execution";
  }

  attr device_attr;
  device_attr.iface = gloo_iface_;
  device_attr.ai_family = AF_UNSPEC;
  auto dev = CreateDevice(device_attr);

  global_ctx = global_context.ctx;
  timeout_ = global_context.timeout_;
  hostname_ = global_context.hostname_;

  std::vector<int> ranks;
  if (registered_ranks.empty()) {
    ranks.resize(global_context.ctx->size);
    std::iota(ranks.begin(), ranks.end(), 0);
  } else {
    ranks = registered_ranks;
  }

  std::string process_set_hash;
  {
    std::ostringstream oss_num;
    std::copy(ranks.begin(), ranks.end(),
              std::ostream_iterator<int>(oss_num, ","));
    std::ostringstream oss;
    oss << std::hex << std::hash<std::string>{}(oss_num.str());
    process_set_hash = oss.str();
    LOG(DEBUG) << "Initializing GlooContext for process set: [" << oss_num.str()
               << "], hash: " << process_set_hash;
  }

  // All processes in global_context.local_ctx, global_context.cross_ctx
  // call EnumerateSubRanks() here:
  auto global_context_all_local_ranks =
      EnumerateSubRanks(global_context.ctx, global_context.local_ctx);
  auto global_context_all_cross_ranks =
      EnumerateSubRanks(global_context.ctx, global_context.cross_ctx);
  
  auto global_context_rank = global_context.ctx->rank;
  auto rank = static_cast<int>(
      std::distance(ranks.begin(), std::find(ranks.begin(), ranks.end(),
                                             global_context_rank)));
  auto size = static_cast<int>(ranks.size());
  bool current_process_included = (rank < size);
  if (!current_process_included) {
    // leaving null: ctx, local_ctx, and cross_ctx 
    return;
  }
  LOG(DEBUG) << "Global Gloo context for process set with rank: " << rank
             << ", size: " << size;
  
  std::string process_set_suffix = "_process_set_hash_" + process_set_hash;

  // 1) process-set-limited global context 
  ctx = Rendezvous(HOROVOD_GLOO_GLOBAL_PREFIX + process_set_suffix,
                   rendezvous_addr_env, rendezvous_port, rank, size, dev,
                   timeout_);
  LOG(DEBUG) << "Global Gloo context initialized for process set with hash "
             << process_set_hash << ".";

  // 2) process-set-limited local context
  auto global_context_local_ranks =
      RanksIntersection(ranks, global_context_all_local_ranks);
  auto it_local_rank =
      std::find(global_context_local_ranks.begin(),
                global_context_local_ranks.end(), global_context_rank);
  int local_rank = -1;
  if (it_local_rank != global_context_local_ranks.end()) {
    local_rank = static_cast<int>(
        std::distance(global_context_local_ranks.begin(), it_local_rank));
    auto local_size = static_cast<int>(global_context_local_ranks.size());
    LOG(DEBUG) << "Local Gloo context for process set with rank: " << local_rank
               << ", size: " << local_size;

    local_ctx =
        Rendezvous(HOROVOD_GLOO_LOCAL_PREFIX + hostname_ + process_set_suffix,
                   rendezvous_addr_env, rendezvous_port, local_rank, local_size,
                   dev, timeout_);
    LOG(DEBUG) << "Local Gloo context initialized for process set with hash "
               << process_set_hash << ".";
  }
  
  // 3) process-set-limited cross context
  auto global_context_cross_ranks =
      RanksIntersection(ranks, global_context_all_cross_ranks);
  auto it_cross_rank =
      std::find(global_context_cross_ranks.begin(),
                global_context_cross_ranks.end(), global_context_rank);
  if (it_cross_rank != global_context_cross_ranks.end()) {
    auto cross_rank = static_cast<int>(std::distance(
        global_context_cross_ranks.begin(), it_cross_rank));
    auto cross_size = static_cast<int>(global_context_cross_ranks.size());
    assert(local_rank >= 0);
    LOG(DEBUG) << "Cross Gloo context for process set with rank: " << cross_rank
               << ", size: " << cross_size;

    cross_ctx = Rendezvous(HOROVOD_GLOO_CROSS_PREFIX +
                               std::to_string(local_rank) + process_set_suffix,
                           rendezvous_addr_env, rendezvous_port, cross_rank,
                           cross_size, dev, timeout_);
    LOG(DEBUG) << "Cross-node Gloo context for process set with hash "
               << process_set_hash << ".";
  }
}


void GlooContext::Finalize() {
  if (!enabled_) {
    return;
  }

  global_ctx.reset();
  ctx.reset();
  cross_ctx.reset();
  local_ctx.reset();
  reset_ = true;
}

std::shared_ptr<gloo::Context>
GlooContext::GetGlooContext(Communicator communicator) const {
  switch (communicator) {
  case Communicator::GLOBAL:
    return ctx;
  case Communicator::LOCAL:
    return local_ctx;
  case Communicator::CROSS:
    return cross_ctx;
  default:
    throw std::logic_error("Unsupported communicator type.");
  }
}

} // namespace common
} // namespace horovod
