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
#include <memory>
#include <sstream>
#include <stdexcept>

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
  attr device_attr;
  device_attr.iface = gloo_iface;
  device_attr.ai_family = AF_UNSPEC;
  auto dev = CreateDevice(device_attr);
  auto timeout = GetTimeoutFromEnv();

  auto context =
      std::make_shared<gloo::mpi::Context>(mpi_ctx.GetMPICommunicator(GLOBAL));
  context->setTimeout(timeout);
  context->connectFullMesh(dev);
  ctx = context;

  auto cross_context =
      std::make_shared<gloo::mpi::Context>(mpi_ctx.GetMPICommunicator(CROSS));
  cross_context->setTimeout(timeout);
  cross_context->connectFullMesh(dev);
  cross_ctx = cross_context;

  auto local_context =
      std::make_shared<gloo::mpi::Context>(mpi_ctx.GetMPICommunicator(LOCAL));
  local_context->setTimeout(timeout);
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
  attr device_attr;
  device_attr.iface = gloo_iface;

  device_attr.ai_family = AF_UNSPEC;
  auto dev = CreateDevice(device_attr);
  auto timeout = GetTimeoutFromEnv();

  auto host_env = std::getenv(HOROVOD_HOSTNAME);
  std::string hostname = host_env != nullptr ? std::string(host_env) : std::string("localhost");

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

    auto key = hostname + ":" + std::to_string(local_rank);
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
      out << hostname << "[" << local_rank << "] has been removed from elastic job";
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
                   rank, size, dev, timeout);
  LOG(DEBUG) << "Global Gloo context initialized.";

  local_ctx = Rendezvous(HOROVOD_GLOO_LOCAL_PREFIX + hostname,
                         rendezvous_addr_env, rendezvous_port,
                         local_rank, local_size, dev, timeout);
  LOG(DEBUG) << "Local Gloo context initialized.";

  cross_ctx = Rendezvous(HOROVOD_GLOO_CROSS_PREFIX + std::to_string(local_rank),
                         rendezvous_addr_env, rendezvous_port,
                         cross_rank, cross_size, dev, timeout);
  LOG(DEBUG) << "Cross-node Gloo context initialized.";
}

void GlooContext::Finalize() {
  if (!enabled_) {
    return;
  }

  ctx.reset();
  cross_ctx.reset();
  local_ctx.reset();
  reset_ = true;
}

std::shared_ptr<gloo::Context>
GlooContext::GetGlooContext(Communicator communicator) {
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
