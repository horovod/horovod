// Copyright 2019 Microsoft. All Rights Reserved.
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

#ifndef HOROVOD_ADASUM_MPI_OPERATIONS_H
#define HOROVOD_ADASUM_MPI_OPERATIONS_H

#include "mpi.h"
#include <iostream>

#include "adasum/adasum_mpi.h"
#include "collective_operations.h"

namespace horovod {
namespace common {

class AdasumMPIAllreduceOp : public AdasumMPI, public AllreduceOp {
public:
  AdasumMPIAllreduceOp(MPIContext* mpi_context,
                       HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_ADASUM_MPI_OPERATIONS_H
