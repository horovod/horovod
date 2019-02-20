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

MPI_Datatype MPIContext::GetMPIDataType(const std::shared_ptr<Tensor> tensor) {
  return GetMPIDataType(tensor->dtype());
}

MPI_Datatype MPIContext::GetMPIDataType(const DataType dtype) {
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

MPI_Comm MPIContext::GetMPICommunicator(Communicator comm) {
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

void MPI_Allreduce(MPIContext& ctx, const void* buffer_data, int64_t num_elements,
                   TensorTableEntry& first_entry, const void* sendbuff,
                   Communicator comm) {
  int op = MPI_Allreduce(sendbuff != nullptr ? sendbuff : MPI_IN_PLACE, (void*) buffer_data,
                         (int) num_elements,
                         ctx.GetMPIDataType(first_entry.tensor),
                         first_entry.tensor->dtype() == HOROVOD_FLOAT16 ? ctx.mpi_float16_sum : MPI_SUM,
                         ctx.GetMPICommunicator(comm));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Allreduce failed, see MPI output for details.");
  }
}

void MPI_Allgatherv(MPIContext& ctx, const void* sendbuf, int sendcount, DataType sendtype,
                    void* recvbuf, const int* recvcounts,
                    const int* displs, DataType recvtype,
                    Communicator comm) {
  int op = MPI_Allgatherv(sendbuf != nullptr ? sendbuf : MPI_IN_PLACE, sendcount, ctx.GetMPIDataType(sendtype),
                          recvbuf, recvcounts, displs, ctx.GetMPIDataType(recvtype),
                          ctx.GetMPICommunicator(comm));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Allgatherv failed, see MPI output for details.");
  }
}

void MPI_Broadcast(MPIContext& ctx, const void* buffer_data, int64_t num_elements,
                   DataType dtype, int root_rank,
                   Communicator comm) {
  int op = MPI_Bcast((void*) buffer_data,
                     (int) num_elements,
                     ctx.GetMPIDataType(dtype),
                     root_rank,
                     ctx.GetMPICommunicator(comm));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Broadcast failed, see MPI output for details.");
  }
}

void MPI_Barrier(MPIContext& ctx, Communicator comm) {
  int op = MPI_Barrier(ctx.GetMPICommunicator(comm));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Barrier failed, see MPI output for details.");
  }
}

void MPI_AllocateSharedBuffer(MPIContext& ctx, int64_t window_size, int element_size, void* baseptr, Communicator comm) {
  MPI_Win_allocate_shared(
      window_size, element_size, MPI_INFO_NULL, ctx.GetMPICommunicator(comm),
      baseptr, &ctx.window);
}

void MPI_FreeSharedBuffer(MPIContext& ctx) {
  MPI_Win_fence(0, ctx.window);
  MPI_Win_free(&ctx.window);
}

void MPI_QuerySharedBuffer(MPIContext& ctx, int rank, void* baseptr) {
  int disp_unit;
  MPI_Aint winsize;
  MPI_Win_shared_query(ctx.window, rank, &winsize, &disp_unit, baseptr);
}

void MPI_GetTypeSize(MPIContext& ctx, DataType dtype, int* out) {
  MPI_Type_size(ctx.GetMPIDataType(dtype), out);
}

void DoMPIAllreduce(MPIContext* mpi_context,
                    std::vector<TensorTableEntry>& entries,
                    void* buffer_data, int64_t& num_elements, size_t& buffer_len) {
  auto& first_entry = entries[0];
  const void* sendbuf = entries.size() > 1 || first_entry.tensor->data() == first_entry.output->data()
                        ? nullptr : first_entry.tensor->data();
  MPI_Allreduce(*mpi_context, buffer_data, num_elements, first_entry, sendbuf, Communicator::GLOBAL);
}

MPIAllreduce::MPIAllreduce(MPIContext* mpi_context, HorovodGlobalState* global_state)
    : AllreduceOp(global_state), mpi_context_(mpi_context) {}

bool MPIAllreduce::Enabled(ParameterManager& param_manager,
                           std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

void MPIAllreduce::DoAllreduce(std::vector<TensorTableEntry>& entries,
                               const void* fused_input_data, void* buffer_data,
                               int64_t& num_elements, size_t& buffer_len) {
  global_state_->timeline.ActivityStartAll(entries, MPI_ALLREDUCE);
  DoMPIAllreduce(mpi_context_, entries, buffer_data, num_elements, buffer_len);
  global_state_->timeline.ActivityEndAll(entries);
}

#if HAVE_CUDA
MPI_CUDAAllreduce::MPI_CUDAAllreduce(MPIContext* mpi_context,
                                     HorovodGlobalState* global_state)
                                     : CUDAAllreduce(cuda_context, comm_context, global_state),
                                       mpi_context_(mpi_context) {}

void MPI_CUDAAllreduce::DoAllreduce(std::vector<TensorTableEntry>& entries,
                                    const void* fused_input_data, void* buffer_data,
                                    int64_t& num_elements, size_t& buffer_len) {
  global_state_->timeline.ActivityStartAll(entries, MPI_ALLREDUCE);
  DoMPIAllreduce(mpi_context_, entries, buffer_data, num_elements, buffer_len);
  global_state_->timeline.ActivityEndAll(entries);
}
#endif

MPIAllgather::MPIAllgather(MPIContext* mpi_context, HorovodGlobalState* global_state)
    : AllgatherOp(global_state), mpi_context_(mpi_context) {}

bool MPIAllgather::Enabled(ParameterManager& param_manager,
                           std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

void MPIAllgather::DoAllgather(std::vector<TensorTableEntry>& entries, int* recvcounts, int* displcmnts,
                               int64_t** entry_component_offsets, int64_t** entry_component_sizes,
                               int64_t total_size, int element_size) {
  // Data is at the CPU and hierarchical allgather is disabled, or
  // Data is at the GPU and HOROVOD_GPU_ALLGATHER == MPI
  auto& timeline = global_state_->timeline;
  auto& first_entry = entries[0];

  const void* sendbuf = nullptr;
  int64_t total_num_elements = 0;
  void* buffer_data;

  if (entries.size() > 1) {
    auto& buffer = global_state_->fusion_buffer.GetBuffer(
        first_entry.device, first_entry.context->framework());
    buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));

    // Copy memory into the fusion buffer. Then the input data of each
    // process is assumed to be in the area where that process would
    // receive its own contribution to the receive buffer.
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    int64_t offset = displcmnts[global_state_->rank] * element_size;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*) buffer_data + offset;
      std::memcpy(buffer_data_at_offset, e.tensor->data(),
                  (size_t) e.tensor->size());
      offset += e.tensor->size();
      total_num_elements += e.tensor->shape().num_elements();
    }
    timeline.ActivityEndAll(entries);
  } else {
    sendbuf = first_entry.tensor->data();
    total_num_elements = first_entry.tensor->shape().num_elements();
    buffer_data = (void*) first_entry.output->data();
  }

  global_state_->timeline.ActivityStartAll(entries, MPI_ALLGATHER);
  MPI_Allgatherv(*mpi_context_,
                 sendbuf,
                 (int) total_num_elements,
                 first_entry.tensor->dtype(),
                 buffer_data,
                 recvcounts,
                 displcmnts,
                 first_entry.tensor->dtype(),
                 Communicator::GLOBAL);
  global_state_->timeline.ActivityEndAll(entries);

  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    // Copy memory out of the fusion buffer.
    for (size_t ec = 0; ec < entries.size(); ++ec) {
      auto& e = entries[ec];
      int64_t copy_offset = 0;
      for (int rc = 0; rc < global_state_->size; ++rc) {
        std::memcpy((void*) ((uint8_t*) e.output->data() + copy_offset),
                    (void*) ((uint8_t*) buffer_data +
                             entry_component_offsets[ec][rc] * element_size),
                    (size_t) entry_component_sizes[ec][rc] * element_size);

        copy_offset += entry_component_sizes[ec][rc] * element_size;
      }
    }
    timeline.ActivityEndAll(entries);
  }

  delete[] recvcounts;
  delete[] displcmnts;

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    delete[] entry_component_sizes[ec];
    delete[] entry_component_offsets[ec];
  }
  delete[] entry_component_sizes;
  delete[] entry_component_offsets;
}

int MPIAllgather::GetElementSize(DataType dtype) const {
  int element_size;
  MPI_GetTypeSize(*mpi_context_, dtype, &element_size);
  return element_size;
}

MPIHierarchicalAllgather::MPIHierarchicalAllgather(MPIContext* mpi_context,
                                                   HorovodGlobalState* global_state)
    : MPIAllgather(mpi_context, global_state) {}

bool MPIHierarchicalAllgather::Enabled(ParameterManager& param_manager,
                                       std::vector<TensorTableEntry>& entries,
                                       const Response& response) const {
  return param_manager.HierarchicalAllgather();
}

void MPIHierarchicalAllgather::DoAllgather(std::vector<TensorTableEntry>& entries, int* recvcounts, int* displcmnts,
                                           int64_t** entry_component_offsets, int64_t** entry_component_sizes,
                                           int64_t total_size, int element_size) {
  auto& timeline = global_state_->timeline;

  // If shared buffer is not initialized or is not large enough, reallocate
  int64_t total_size_in_bytes = total_size * element_size;
  if (global_state_->shared_buffer == nullptr || global_state_->shared_buffer_size < total_size_in_bytes) {
    if (global_state_->shared_buffer != nullptr) {
      MPI_FreeSharedBuffer(*mpi_context_);
      global_state_->shared_buffer = nullptr;
    }

    // Allocate shared memory, give each rank their respective pointer
    timeline.ActivityStartAll(entries, ALLOCATE_SHARED_BUFFER);
    int64_t window_size = global_state_->local_rank == 0 ? total_size_in_bytes : 0;
    MPI_AllocateSharedBuffer(*mpi_context_, window_size, element_size, &global_state_->shared_buffer,
                             Communicator::LOCAL);
    if (global_state_->local_rank != 0) {
      MPI_QuerySharedBuffer(*mpi_context_, 0, &global_state_->shared_buffer);
    }
    global_state_->shared_buffer_size = total_size_in_bytes;
    timeline.ActivityEndAll(entries);
  }

  // Compute cross-node allgather displacements and recvcounts for
  // homogeneous/parallelized case
  auto* cross_recvcounts = new int[global_state_->cross_size]();
  auto* cross_displcmnts = new int[global_state_->cross_size]();

  if (global_state_->is_homogeneous) {
    for (int i = 0; i < global_state_->cross_size; ++i) {
      cross_recvcounts[i] = recvcounts[global_state_->local_size * i +
                                       global_state_->local_rank];
      cross_displcmnts[i] = displcmnts[global_state_->local_size * i +
                                       global_state_->local_rank];
    }
  } else if (global_state_->local_rank == 0) {
    // In this case local rank 0 will allgather with all local data
    int offset = 0;
    for (int i = 0; i < global_state_->cross_size; ++i) {
      for (int j = offset; j < offset + global_state_->local_sizes[i];
           ++j) {
        cross_recvcounts[i] += recvcounts[j];
      }
      cross_displcmnts[i] = displcmnts[offset];
      offset += global_state_->local_sizes[i];
    }
  }

  timeline.ActivityStartAll(entries, MEMCPY_IN_SHARED_BUFFER);
  for (size_t ec = 0; ec < entries.size(); ++ec) {
    auto& e = entries[ec];
    void* shared_buffer_at_offset =
        (uint8_t*) global_state_->shared_buffer +
        entry_component_offsets[ec][global_state_->rank] * element_size;

    // CPU copy to shared buffer
    memcpy(shared_buffer_at_offset, e.tensor->data(),
           (size_t) (entry_component_sizes[ec][global_state_->rank] *
                     element_size));
  }
  MPI_Barrier(*mpi_context_, Communicator::GLOBAL);
  timeline.ActivityEndAll(entries);

  // Perform the cross-node allgather. If the cluster is homogeneous all
  // local ranks participate, otherwise local rank 0 handles all data
  global_state_->timeline.ActivityStartAll(entries, MPI_CROSS_ALLGATHER);
  auto& first_entry = entries[0];
  if (global_state_->is_homogeneous || global_state_->local_rank == 0) {
    MPI_Allgatherv(*mpi_context_,
                   nullptr,
                   0,
                   DataType::HOROVOD_NULL,
                   global_state_->shared_buffer,
                   cross_recvcounts,
                   cross_displcmnts,
                   first_entry.tensor->dtype(),
                   Communicator::CROSS);
  }
  MPI_Barrier(*mpi_context_, Communicator::GLOBAL);
  global_state_->timeline.ActivityEndAll(entries);

  // Copy memory out of the fusion buffer.
  timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
  for (size_t ec = 0; ec < entries.size(); ++ec) {
    auto& e = entries[ec];
    int64_t copy_offset = 0;
    for (int rc = 0; rc < global_state_->size; ++rc) {
      auto entry_component_size = entry_component_sizes[ec][rc];
      std::memcpy((void*) ((uint8_t*) e.output->data() + copy_offset),
                  (void*) ((uint8_t*) global_state_->shared_buffer +
                           entry_component_size * element_size),
                  (size_t) entry_component_size * element_size);
      copy_offset += entry_component_size * element_size;
    }
  }
  MPI_Barrier(*mpi_context_, Communicator::GLOBAL);
  timeline.ActivityEndAll(entries);

  // Free the buffers
  delete[] cross_displcmnts;
  delete[] cross_recvcounts;
}

MPIBroadcast::MPIBroadcast(MPIContext* mpi_context, HorovodGlobalState* global_state)
    : BroadcastOp(global_state), mpi_context_(mpi_context) {}

bool MPIBroadcast::Enabled(ParameterManager& param_manager,
                           std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

void MPIBroadcast::DoBroadcast(std::vector<TensorTableEntry>& entries,
                               const void* buffer_data, int64_t num_elements,
                               DataType dtype, int root_rank) {
  global_state_->timeline.ActivityStartAll(entries, MPI_BCAST);
  MPI_Broadcast(*mpi_context_, buffer_data, num_elements, dtype, root_rank,
                Communicator::GLOBAL);
  global_state_->timeline.ActivityEndAll(entries);
}

} // namespace common
} // namespace horovod
