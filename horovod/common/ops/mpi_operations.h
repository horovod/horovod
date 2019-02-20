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
#include "../communication_channel.h"
#include "../global_state.h"
#include "collective_operations.h"

#if HAVE_CUDA
#include "cuda_operations.h"
#endif

namespace horovod {
namespace common {

class MPIChannel : public Channel {
public:
  void Allreduce(const void* buffer_data, int64_t num_elements,
                 TensorTableEntry& first_entry, const void* sendbuff,
                 Communicator comm) override;

  void Allgatherv(const void* sendbuf, int sendcount, DataType sendtype,
                  void* recvbuf, const int recvcounts[],
                  const int displs[], DataType recvtype,
                  Communicator comm) override;

  void Broadcast(const void* buffer_data, int64_t num_elements,
                 DataType dtype, int root_rank,
                 Communicator comm) override;

  void Barrier(Communicator comm) override;

  void AllocateSharedBuffer(int64_t window_size, int element_size, void* baseptr, Communicator comm) override;

  void FreeSharedBuffer() override;

  void QuerySharedBuffer(int rank, void* baseptr) override;

  void GetTypeSize(DataType dtype, int* out) override;

  MPI_Datatype GetMPIDataType(std::shared_ptr<Tensor> tensor);

  MPI_Datatype GetMPIDataType(DataType dtype);

  MPI_Comm GetMPICommunicator(Communicator comm);

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
  MPIAllreduce(MPIChannel* mpi_channel, HorovodGlobalState* global_state);

  virtual ~MPIAllreduce() = default;

  bool Enabled(ParameterManager& param_manager,
               std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  void DoAllreduce(std::vector<TensorTableEntry>& entries,
                   const void* fused_input_data, void* buffer_data,
                   int64_t& num_elements, size_t& buffer_len) override;

  MPIChannel* mpi_channel_;
};

#if HAVE_CUDA
class MPI_CUDAAllreduce : public CUDAAllreduce {
public:
  MPI_CUDAAllreduce(MPIChannel* mpi_channel, CUDAContext* cuda_context,
                    CommunicationContext* comm_context, HorovodGlobalState* global_state);
  virtual ~MPI_CUDAAllreduce()=default;

protected:
  void DoAllreduce(std::vector<TensorTableEntry>& entries,
                   const void* fused_input_data, void* buffer_data,
                   int64_t& num_elements, size_t& buffer_len) override;

  MPIChannel* mpi_channel_;
};
#endif

class MPIAllgather : public AllgatherOp {
public:
  MPIAllgather(MPIChannel* mpi_channel, HorovodGlobalState* global_state);

  bool Enabled(ParameterManager& param_manager,
               std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  void DoAllgatherv(std::vector<TensorTableEntry>& entries,
                    const void* sendbuf, int sendcount, DataType sendtype,
                    void* recvbuf, const int recvcounts[],
                    const int displs[], DataType recvtype) override;

  int GetElementSize(DataType dtype) const override;

  MPIChannel* mpi_channel_;
};

class MPIHierarchicalAllgather : public HierarchicalAllgather {
public:
  MPIHierarchicalAllgather(MPIChannel* mpi_channel, HorovodGlobalState* global_state);

  bool Enabled(ParameterManager& param_manager,
               std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  void DoAllgatherv(std::vector<TensorTableEntry>& entries,
                    const void* sendbuf, int sendcount, DataType sendtype,
                    void* recvbuf, const int recvcounts[],
                    const int displs[], DataType recvtype) override;

  void Barrier() override;

  void FreeSharedBuffer() override;

  void AllocateSharedBuffer(int64_t total_size_in_bytes, int element_size) override;

  MPIChannel* mpi_channel_;
};

class MPIBroadcast : public BroadcastOp {
public:
  MPIBroadcast(MPIChannel* mpi_channel, HorovodGlobalState* global_state);

  bool Enabled(ParameterManager& param_manager,
               std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  void DoBroadcast(std::vector<TensorTableEntry>& entries,
                   const void* buffer_data, int64_t num_elements,
                   DataType dtype, int root_rank) override;

  MPIChannel* mpi_channel_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_MPI_OPERATIONS_H
