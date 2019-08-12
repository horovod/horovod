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

#include "gloo/rendezvous/context.h"
#include "gloo/rendezvous/file_store.h"
#include "gloo/rendezvous/prefix_store.h"
#include "gloo/transport/tcp/device.h"

#include "http_store.h"

#if HAVE_MPI
#include "gloo/mpi/context.h"
#endif

namespace horovod {
namespace common {

#if HAVE_MPI
void GlooContext::InitializeFromMPI(MPIContext& mpi_ctx,
                                    const std::string& gloo_iface) {
  if (!enabled_) {
    return;
  }

  // TODO(sihan): Add support for multiple interfaces:
  //  https://github.com/facebookincubator/gloo/issues/190
  gloo::transport::tcp::attr attr;
  attr.iface = gloo_iface;
  attr.ai_family = AF_UNSPEC;
  auto dev = gloo::transport::tcp::CreateDevice(attr);

  auto context =
      std::make_shared<gloo::mpi::Context>(mpi_ctx.GetMPICommunicator(GLOBAL));
  context->connectFullMesh(dev);
  ctx = context;

  auto cross_context =
      std::make_shared<gloo::mpi::Context>(mpi_ctx.GetMPICommunicator(CROSS));
  cross_context->connectFullMesh(dev);
  cross_ctx = cross_context;

  auto local_context =
      std::make_shared<gloo::mpi::Context>(mpi_ctx.GetMPICommunicator(LOCAL));
  local_context->connectFullMesh(dev);
  local_ctx = local_context;
}
#endif

void GlooContext::Finalize() {
  if (!enabled_) {
    return;
  }

  ctx.reset();
  cross_ctx.reset();
  local_ctx.reset();
}

void GlooContext::Initialize(const std::string& gloo_iface) {
  if (!enabled_) {
    return;
  }

  // Create a tcp device for communication
  // TODO(sihan): Add support for multiple interfaces:
  //  https://github.com/facebookincubator/gloo/issues/190
  gloo::transport::tcp::attr attr;
  attr.iface = gloo_iface;

  attr.ai_family = AF_UNSPEC;
  auto dev = gloo::transport::tcp::CreateDevice(attr);

  const std::string rendezvous_server_addr = std::getenv
      (HOROVOD_GLOO_RENDEZVOUS_ADDR);
  auto rendezvous_server_port =
      std::strtol(std::getenv(HOROVOD_GLOO_RENDEZVOUS_PORT), nullptr, 10);

  LOG(DEBUG) << "Rendezvous server address " << rendezvous_server_addr;

  // Get rendezvous info from env
  int rank = std::strtol(getenv(HOROVOD_RANK), nullptr, 10);
  int size = std::strtol(getenv(HOROVOD_SIZE), nullptr, 10);
  int local_rank = std::strtol(getenv(HOROVOD_LOCAL_RANK), nullptr, 10);
  int local_size = std::strtol(getenv(HOROVOD_LOCAL_SIZE), nullptr, 10);
  int cross_rank = std::strtol(getenv(HOROVOD_CROSS_RANK), nullptr, 10);
  int cross_size = std::strtol(getenv(HOROVOD_CROSS_SIZE), nullptr, 10);

  // Global rendezvous
  const std::string global_scope(HOROVOD_GLOO_GLOBAL_PREFIX);
  auto rendezvous = HTTPStore(rendezvous_server_addr, rendezvous_server_port,
                              global_scope, rank);
  LOG(DEBUG) << "Global Rendezvous started for rank " << rank
             << ", total size of " << size;
  auto context = std::make_shared<gloo::rendezvous::Context>(rank, size);
  context->connectFullMesh(rendezvous, dev);
  ctx = context;
  rendezvous.Finalize();
  LOG(DEBUG) << "Global Gloo context initialized.";

  // Local rendezvous
  const std::string local_scope =
      std::string(HOROVOD_GLOO_LOCAL_PREFIX) + std::to_string(cross_rank);
  auto local_rendezvous = HTTPStore(rendezvous_server_addr,
                                    rendezvous_server_port, local_scope,
                                    local_rank);
  LOG(DEBUG) << "Local Rendezvous started for rank " << rank
             << ", total size of " << local_size;
  auto local_context =
      std::make_shared<gloo::rendezvous::Context>(local_rank, local_size);
  local_context->connectFullMesh(local_rendezvous, dev);
  local_ctx = local_context;
  local_rendezvous.Finalize();
  LOG(DEBUG) << "Local Gloo context initialized.";

  // Cross rendezvous
  const std::string cross_scope =
      std::string(HOROVOD_GLOO_CROSS_PREFIX) + std::to_string(local_rank);
  auto cross_rendezvous = HTTPStore(rendezvous_server_addr,
                                    rendezvous_server_port, cross_scope,
                                    cross_rank);
  LOG(DEBUG) << "Cross-node Rendezvous started for rank " << rank
             << ", total size of " << size;
  auto cross_context =
      std::make_shared<gloo::rendezvous::Context>(cross_rank, cross_size);
  cross_context->connectFullMesh(cross_rendezvous, dev);
  cross_ctx = cross_context;
  cross_rendezvous.Finalize();
  LOG(DEBUG) << "Cross-node Gloo context initialized.";
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
