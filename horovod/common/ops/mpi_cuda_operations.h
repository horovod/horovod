// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
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

#ifndef HOROVOD_MPI_CUDA_OPERATIONS_H
#define HOROVOD_MPI_CUDA_OPERATIONS_H

#include "../mpi_context.h"
#include "cuda_operations.h"

namespace horovod {
namespace common {

class MPI_CUDAAllreduce : public CUDAAllreduce {
public:
  MPI_CUDAAllreduce(MPIContext* mpi_context, CUDAContext* cuda_context, HorovodGlobalState* global_state);
  virtual ~MPI_CUDAAllreduce()=default;

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

protected:
  MPIContext* mpi_context_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_MPI_CUDA_OPERATIONS_H
