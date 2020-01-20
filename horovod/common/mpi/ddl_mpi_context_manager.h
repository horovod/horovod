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

#ifndef HOROVOD_DDL_MPI_CONTEXT_MANAGER_H
#define HOROVOD_DDL_MPI_CONTEXT_MANAGER_H

#include "mpi_context.h"
#include "../ops/gpu_operations.h"
#include "../ops/ddl_operations.h"

namespace horovod {
namespace common {

// Derived from MPIContextManager since DDL is able to manage MPI environment
// (initialization and finalization).
class DDL_MPIContextManager : public MPIContextManager {
public:
  // Constructor, store the reference of ddl context and gpu context.
  DDL_MPIContextManager(DDLContext& ddl_context, GPUContext& gpu_context)
      : ddl_context_(ddl_context), gpu_context_(gpu_context){};

  // Initialize MPI environment with DDLInit().
  void EnvInitialize(int required) override;

  // Finalize MPI environment with ddl_finalize().
  void EnvFinalize() override;

  DDLContext& ddl_context_;
  GPUContext& gpu_context_;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_DDL_MPI_CONTEXT_MANAGER_H
