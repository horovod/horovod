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

#include "ddl_mpi_context_manager.h"

namespace horovod {
namespace common {

void DDL_MPIContextManager::EnvInitialize(int required) {
  // DDLInit calls MPI_Init
  DDLAllreduce::DDLInit(&ddl_context_, &gpu_context_);
}

void DDL_MPIContextManager::EnvFinalize() {
  // ddl_finalize calls MPI_Finalize
  ddl_finalize();
}

} // namespace common
} // namespace horovod
