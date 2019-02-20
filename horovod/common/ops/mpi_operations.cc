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

#include "mpi_operations.h"

namespace horovod {
namespace common {

void MPIChannel::Allreduce(const void* buffer_data, int64_t num_elements,
                           TensorTableEntry& first_entry, const void* sendbuff,
                           Communicator comm) {
  int op = MPI_Allreduce(sendbuff != nullptr ? sendbuff : MPI_IN_PLACE, (void*) buffer_data,
                         (int) num_elements,
                         GetMPIDataType(first_entry.tensor),
                         first_entry.tensor->dtype() == HOROVOD_FLOAT16 ? mpi_float16_sum : MPI_SUM,
                         GetMPICommunicator(comm));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Allreduce failed, see MPI output for details.");
  }
}

void MPIChannel::Allgatherv(const void* sendbuf, int sendcount, DataType sendtype,
                            void* recvbuf, const int recvcounts[],
                            const int displs[], DataType recvtype,
                            Communicator comm) {
  int op = MPI_Allgatherv(sendbuf != nullptr ? sendbuf : MPI_IN_PLACE, sendcount, GetMPIDataType(sendtype),
                          recvbuf, recvcounts, displs, GetMPIDataType(recvtype),
                          GetMPICommunicator(comm));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Allgatherv failed, see MPI output for details.");
  }
}

void MPIChannel::Broadcast(const void* buffer_data, int64_t num_elements,
                           DataType dtype, int root_rank,
                           Communicator comm) {
  int op = MPI_Bcast((void*) buffer_data,
                     (int) num_elements,
                     GetMPIDataType(dtype),
                     root_rank,
                     GetMPICommunicator(comm));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Broadcast failed, see MPI output for details.");
  }
}

void MPIChannel::Barrier(Communicator comm) {
  int op = MPI_Barrier(GetMPICommunicator(comm));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Barrier failed, see MPI output for details.");
  }
}

void MPIChannel::AllocateSharedBuffer(int64_t window_size, int element_size, void* baseptr, Communicator comm) {
  MPI_Win_allocate_shared(
      window_size, element_size, MPI_INFO_NULL, GetMPICommunicator(comm),
      baseptr, &window);
}

void MPIChannel::FreeSharedBuffer() {
  MPI_Win_fence(0, window);
  MPI_Win_free(&window);
}

void MPIChannel::QuerySharedBuffer(int rank, void* baseptr) {
  int disp_unit;
  MPI_Aint winsize;
  MPI_Win_shared_query(window, rank, &winsize, &disp_unit, baseptr);
}

void MPIChannel::GetTypeSize(DataType dtype, int* out) {
  MPI_Type_size(GetMPIDataType(dtype), out);
}

MPI_Datatype MPIChannel::GetMPIDataType(const std::shared_ptr<Tensor> tensor) {
  return GetMPIDataType(tensor->dtype());
}

MPI_Datatype MPIChannel::GetMPIDataType(const DataType dtype) {
  switch (dtype) {
    case HOROVOD_UINT8:
      return MPI_UINT8_T;
    case HOROVOD_INT8:
      return MPI_INT8_T;
    case HOROVOD_UINT16:
      return MPI_UINT16_T;
    case HOROVOD_INT16:
      return MPI_INT16_T;
    case HOROVOD_INT32:
      return MPI_INT32_T;
    case HOROVOD_INT64:
      return MPI_INT64_T;
    case HOROVOD_FLOAT16:
      return mpi_float16_t;
    case HOROVOD_FLOAT32:
      return MPI_FLOAT;
    case HOROVOD_FLOAT64:
      return MPI_DOUBLE;
    case HOROVOD_BOOL:
      return MPI_C_BOOL;
    case HOROVOD_BYTE:
      return MPI_BYTE;
    case HOROVOD_NULL:
      return MPI_DATATYPE_NULL;
    default:
      throw std::logic_error("Type " + DataType_Name(dtype) +
                             " is not supported in MPI mode.");
  }
}

MPI_Comm MPIChannel::GetMPICommunicator(Communicator comm) {
  switch (comm) {
    case GLOBAL:
      return mpi_comm;
    case LOCAL:
      return local_comm;
    case CROSS:
      return cross_comm;
    default:
      throw std::logic_error("Communicator " + CommunicatorName(comm) +
                             " is not supported in MPI mode.");
  }
}

void DoMPIAllreduce(MPIChannel* mpi_channel,
                    std::vector<TensorTableEntry>& entries,
                    void* buffer_data, int64_t& num_elements, size_t& buffer_len) {
  auto& first_entry = entries[0];
  const void* sendbuf = entries.size() > 1 || first_entry.tensor->data() == first_entry.output->data()
                        ? nullptr : first_entry.tensor->data();
  mpi_channel->Allreduce(buffer_data, num_elements, first_entry, sendbuf, Channel::Communicator::GLOBAL);
}

MPIAllreduce::MPIAllreduce(MPIChannel* mpi_channel, HorovodGlobalState* global_state)
    : AllreduceOp(global_state), mpi_channel_(mpi_channel) {}

bool MPIAllreduce::Enabled(ParameterManager& param_manager,
                           std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

void MPIAllreduce::DoAllreduce(std::vector<TensorTableEntry>& entries,
                               const void* fused_input_data, void* buffer_data,
                               int64_t& num_elements, size_t& buffer_len) {
  global_state_->timeline.ActivityStartAll(entries, MPI_ALLREDUCE);
  DoMPIAllreduce(mpi_channel_, entries, buffer_data, num_elements, buffer_len);
  global_state_->timeline.ActivityEndAll(entries);
}

#if HAVE_CUDA
MPI_CUDAAllreduce::MPI_CUDAAllreduce(MPIChannel* mpi_channel,
                                     HorovodGlobalState* global_state)
                                     : CUDAAllreduce(cuda_context, comm_context, global_state),
                                       mpi_channel_(mpi_channel) {}

void MPI_CUDAAllreduce::DoAllreduce(std::vector<TensorTableEntry>& entries,
                                    const void* fused_input_data, void* buffer_data,
                                    int64_t& num_elements, size_t& buffer_len) {
  global_state_->timeline.ActivityStartAll(entries, MPI_ALLREDUCE);
  DoMPIAllreduce(mpi_channel_, entries, buffer_data, num_elements, buffer_len);
  global_state_->timeline.ActivityEndAll(entries);
}
#endif

MPIAllgather::MPIAllgather(MPIChannel* mpi_channel, HorovodGlobalState* global_state)
    : AllgatherOp(global_state), mpi_channel_(mpi_channel) {}

bool MPIAllgather::Enabled(ParameterManager& param_manager,
                           std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

void MPIAllgather::DoAllgatherv(std::vector<TensorTableEntry>& entries,
                                const void* sendbuf, int sendcount, DataType sendtype,
                                void* recvbuf, const int recvcounts[],
                                const int displs[], DataType recvtype) {
  global_state_->timeline.ActivityStartAll(entries, MPI_ALLGATHER);
  mpi_channel_->Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
                           Channel::Communicator::GLOBAL);
  global_state_->timeline.ActivityEndAll(entries);
}

int MPIAllgather::GetElementSize(DataType dtype) const {
  int element_size;
  mpi_channel_->GetTypeSize(dtype, &element_size);
  return element_size;
}

MPIHierarchicalAllgather::MPIHierarchicalAllgather(MPIChannel* mpi_channel,
                                                   HorovodGlobalState* global_state)
    : HierarchicalAllgather(global_state),
      mpi_channel_(mpi_channel) {}

bool MPIHierarchicalAllgather::Enabled(ParameterManager& param_manager,
                                       std::vector<TensorTableEntry>& entries,
                                       const Response& response) const {
  return param_manager.HierarchicalAllgather();
}

void MPIHierarchicalAllgather::DoAllgatherv(std::vector<TensorTableEntry>& entries,
                                            const void* sendbuf, int sendcount, DataType sendtype,
                                            void* recvbuf, const int recvcounts[],
                                            const int displs[], DataType recvtype) {
  // Perform the cross-node allgather. If the cluster is homogeneous all
  // local ranks participate, otherwise local rank 0 handles all data
  global_state_->timeline.ActivityStartAll(entries, MPI_CROSS_ALLGATHER);
  if (global_state_->is_homogeneous || global_state_->local_rank == 0) {
    mpi_channel_->Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
                             Channel::Communicator::CROSS);
  }
  Barrier();
  global_state_->timeline.ActivityEndAll(entries);
}

void MPIHierarchicalAllgather::Barrier() {
  mpi_channel_->Barrier(Channel::Communicator::GLOBAL);
}

void MPIHierarchicalAllgather::FreeSharedBuffer() {
  if (global_state_->shared_buffer != nullptr) {
    mpi_channel_->FreeSharedBuffer();
    global_state_->shared_buffer = nullptr;
  }
}

void MPIHierarchicalAllgather::AllocateSharedBuffer(int64_t total_size_in_bytes, int element_size) {
  int64_t window_size = global_state_->local_rank == 0 ? total_size_in_bytes : 0;
  mpi_channel_->AllocateSharedBuffer(window_size, element_size, &global_state_->shared_buffer,
                                     Channel::Communicator::LOCAL);
  if (global_state_->local_rank != 0) {
    mpi_channel_->QuerySharedBuffer(0, &global_state_->shared_buffer);
  }
  global_state_->shared_buffer_size = total_size_in_bytes;
}

MPIBroadcast::MPIBroadcast(MPIChannel* mpi_channel, HorovodGlobalState* global_state)
    : BroadcastOp(global_state), mpi_channel_(mpi_channel) {}

bool MPIBroadcast::Enabled(ParameterManager& param_manager,
                           std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

void MPIBroadcast::DoBroadcast(std::vector<TensorTableEntry>& entries,
                               const void* buffer_data, int64_t num_elements,
                               DataType dtype, int root_rank) {
  global_state_->timeline.ActivityStartAll(entries, MPI_BCAST);
  mpi_channel_->Broadcast(buffer_data, num_elements, dtype, root_rank,
                          Channel::Communicator::GLOBAL);
  global_state_->timeline.ActivityEndAll(entries);
}

} // namespace common
} // namespace horovod
