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

#include "gloo/mpi/context.h"
#include "gloo/transport/tcp/device.h"

namespace horovod {
namespace common {

void GlooContext::InitializeFromMPI(const MPI_Comm& mpi_comm,
                                    const char* gloo_iface) {
  gloo::transport::tcp::attr attr;
  // TODO(sihan): add interface load balancing after
  //  https://github.com/facebookincubator/gloo/issues/183 is resolved
  attr.iface = gloo_iface;
  attr.ai_family = AF_UNSPEC;
  auto dev = gloo::transport::tcp::CreateDevice(attr);

  auto context = std::make_shared<gloo::mpi::Context>(mpi_comm);
  context->connectFullMesh(dev);
  ctx = context;
}

void GlooContext::Finalize() {
    ctx.reset();
}

} // namespace common
} // namespace horovod
