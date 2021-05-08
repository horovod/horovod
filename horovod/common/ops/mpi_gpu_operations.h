// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef HOROVOD_MPI_GPU_OPERATIONS_H
#define HOROVOD_MPI_GPU_OPERATIONS_H

#include "gpu_operations.h"

namespace horovod {
namespace common {

class MPI_GPUAllreduce : public GPUAllreduce {
public:
  MPI_GPUAllreduce(GPUContext* gpu_context, HorovodGlobalState* global_state);
  virtual ~MPI_GPUAllreduce()=default;

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;
};

class MPI_GPUAllgather : public GPUAllgather {
public:
  MPI_GPUAllgather(GPUContext* gpu_context, HorovodGlobalState* global_state);
  virtual ~MPI_GPUAllgather()=default;

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;
};

// TODO: Add MPI_GPUBroadcast implementation

class MPI_GPUAlltoall : public GPUAlltoall {
public:
  MPI_GPUAlltoall(GPUContext* gpu_context, HorovodGlobalState* global_state);
  virtual ~MPI_GPUAlltoall()=default;

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_MPI_GPU_OPERATIONS_H
