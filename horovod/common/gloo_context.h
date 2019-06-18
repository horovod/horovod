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
// =============================================================================

#ifndef HOROVOD_GLOO_CONTEXT_H
#define HOROVOD_GLOO_CONTEXT_H

#include "gloo/context.h"
#include "mpi.h"

#include "common.h"

namespace horovod {
namespace common {

struct GlooContext {
  void InitializeFromMPI(const MPI_Comm &mpi_comm, const char* gloo_iface);

  void Finalize();

  std::shared_ptr<gloo::Context> ctx;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_GLOO_CONTEXT_H
