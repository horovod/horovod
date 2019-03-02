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

#ifndef HOROVOD_MPI_OPERATIONS_H
#define HOROVOD_MPI_OPERATIONS_H

#include <iostream>

#include "mpi.h"

#include "../common.h"
#include "../global_state.h"
#include "collective_operations.h"

#if HAVE_CUDA
#include "cuda_operations.h"
#endif

namespace horovod {
namespace common {

struct MPIContext {
  MPI_Datatype GetMPIDataType(std::shared_ptr<Tensor> tensor);

  MPI_Datatype GetMPIDataType(DataType dtype);

  MPI_Op GetMPISumOp(DataType dtype);

  MPI_Comm GetMPICommunicator(Communicator comm);

  int GetMPITypeSize(DataType dtype);

  // MPI custom data type for float16.
  MPI_Datatype mpi_float16_t;
  MPI_Op mpi_float16_sum;

  // Private MPI communicator for Horovod to ensure no collisions with other
  // threads using MPI.
  MPI_Comm mpi_comm;

  // Node-local communicator.
  MPI_Comm local_comm;

  // Cross-node communicator for hierarchical allreduce.
  MPI_Comm cross_comm;

  // MPI Window used for shared memory allgather
  MPI_Win window;
};

class MPIAllreduce : public AllreduceOp {
public:
  MPIAllreduce(MPIContext* mpi_context, HorovodGlobalState* global_state);

  virtual ~MPIAllreduce() = default;

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  bool Enabled(ParameterManager& param_manager,
               std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  void MemcpyEntryInFusionBuffer(void* buffer_data_at_offset, TensorTableEntry& e,
                                 std::vector<TensorTableEntry>& entries) override;

  void MemcpyEntryOutFusionBuffer(void* buffer_data_at_offset, TensorTableEntry& e,
                                  std::vector<TensorTableEntry>& entries) override;

  MPIContext* mpi_context_;
};

#if HAVE_CUDA
class MPI_CUDAAllreduce : public CUDAAllreduce {
public:
  MPI_CUDAAllreduce(MPIContext* mpi_context, CUDAContext* cuda_context,
                    CommunicationContext* comm_context, HorovodGlobalState* global_state);
  virtual ~MPI_CUDAAllreduce()=default;

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

protected:
  MPIContext* mpi_context_;
};
#endif

class MPIAllgather : public AllgatherOp {
public:
  MPIAllgather(MPIContext* mpi_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  bool Enabled(ParameterManager& param_manager,
               std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  MPIContext* mpi_context_;
};

class MPIHierarchicalAllgather : public MPIAllgather {
public:
  MPIHierarchicalAllgather(MPIContext* mpi_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  bool Enabled(ParameterManager& param_manager,
               std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

private:
  void Barrier();
};

class MPIBroadcast : public BroadcastOp {
public:
  MPIBroadcast(MPIContext* mpi_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  bool Enabled(ParameterManager& param_manager,
               std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  MPIContext* mpi_context_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_MPI_OPERATIONS_H
