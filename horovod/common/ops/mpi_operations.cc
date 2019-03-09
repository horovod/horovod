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

MPIAllreduce::MPIAllreduce(MPIContext* mpi_context, HorovodGlobalState* global_state)
    : AllreduceOp(global_state), mpi_context_(mpi_context) {}

Status MPIAllreduce::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& first_entry = entries[0];

  void* buffer_data;
  size_t buffer_len;
  int64_t num_elements = NumElements(entries);

  // Copy memory into the fusion buffer.
  auto& timeline = global_state_->timeline;
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    const void* fused_input_data;
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);
    timeline.ActivityEndAll(entries);
  } else {
    buffer_data = (void*) first_entry.output->data();
    buffer_len = (size_t) first_entry.output->size();
  }

  // Do allreduce.
  timeline.ActivityStartAll(entries, MPI_ALLREDUCE);
  const void* sendbuf = entries.size() > 1 || first_entry.tensor->data() == first_entry.output->data()
                        ? MPI_IN_PLACE : first_entry.tensor->data();
  int op = MPI_Allreduce(sendbuf, buffer_data,
                         (int) num_elements,
                         mpi_context_->GetMPIDataType(first_entry.tensor),
                         mpi_context_->GetMPISumOp(first_entry.tensor->dtype()),
                         mpi_context_->GetMPICommunicator(Communicator::GLOBAL));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Allreduce failed, see MPI output for details.");
  }
  timeline.ActivityEndAll(entries);

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(buffer_data, entries);
    timeline.ActivityEndAll(entries);
  }

  return Status::OK();
}

bool MPIAllreduce::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

void MPIAllreduce::MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                             const TensorTableEntry& e, void* buffer_data_at_offset) {
  std::memcpy(buffer_data_at_offset, e.tensor->data(),
              (size_t) e.tensor->size());
}

void MPIAllreduce::MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                              const void* buffer_data_at_offset, TensorTableEntry& e) {
  std::memcpy((void*) e.output->data(), buffer_data_at_offset,
              (size_t) e.tensor->size());
}

MPIAllgather::MPIAllgather(MPIContext* mpi_context, HorovodGlobalState* global_state)
    : AllgatherOp(global_state), mpi_context_(mpi_context) {}

bool MPIAllgather::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

Status MPIAllgather::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& timeline = global_state_->timeline;

  // Sizes of subcomponents of each entry from all ranks
  auto** entry_component_sizes = new int64_t* [entries.size()];

  // Offset of each subcomponent of every entry in the final buffer after
  // allgatherv
  auto** entry_component_offsets = new int64_t* [entries.size()];

  auto* recvcounts = new int[global_state_->size]();
  auto* displcmnts = new int[global_state_->size]();

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    entry_component_sizes[ec] = new int64_t[global_state_->size]();
    entry_component_offsets[ec] = new int64_t[global_state_->size]();
  }

  auto& first_entry = entries[0];

  timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);
  Status status = AllocateOutput(entries, response, entry_component_sizes, recvcounts);
  if (!status.ok()) {
    return status;
  }
  timeline.ActivityEndAll(entries);

  SetDisplacements(recvcounts, displcmnts);
  SetEntryComponentOffsets(entries, entry_component_sizes, recvcounts, entry_component_offsets);

  int element_size = mpi_context_->GetMPITypeSize(first_entry.tensor->dtype());

  const void* sendbuf = nullptr;
  void* buffer_data;
  int64_t total_num_elements = NumElements(entries);

  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    MemcpyInFusionBuffer(entries, displcmnts, element_size, buffer_data);
    timeline.ActivityEndAll(entries);
  } else {
    sendbuf = first_entry.tensor->data();
    buffer_data = (void*) first_entry.output->data();
  }

  global_state_->timeline.ActivityStartAll(entries, MPI_ALLGATHER);
  auto dtype = mpi_context_->GetMPIDataType(first_entry.tensor->dtype());
  int op = MPI_Allgatherv(sendbuf != nullptr ? sendbuf : MPI_IN_PLACE,
                          (int) total_num_elements,
                          dtype,
                          buffer_data,
                          recvcounts,
                          displcmnts,
                          dtype,
                          mpi_context_->GetMPICommunicator(Communicator::GLOBAL));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Allgatherv failed, see MPI output for details.");
  }
  global_state_->timeline.ActivityEndAll(entries);

  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(entry_component_offsets, entry_component_sizes,
                          buffer_data, element_size, entries);
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

  return Status::OK();
}

MPIHierarchicalAllgather::MPIHierarchicalAllgather(MPIContext* mpi_context,
                                                   HorovodGlobalState* global_state)
    : MPIAllgather(mpi_context, global_state) {}

Status MPIHierarchicalAllgather::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& timeline = global_state_->timeline;

  // Sizes of subcomponents of each entry from all ranks
  auto** entry_component_sizes = new int64_t* [entries.size()];

  // Offset of each subcomponent of every entry in the final buffer after
  // allgatherv
  auto** entry_component_offsets = new int64_t* [entries.size()];

  auto* recvcounts = new int[global_state_->size]();
  auto* displcmnts = new int[global_state_->size]();

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    entry_component_sizes[ec] = new int64_t[global_state_->size]();
    entry_component_offsets[ec] = new int64_t[global_state_->size]();
  }

  auto& first_entry = entries[0];

  timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);
  Status status = AllocateOutput(entries, response, entry_component_sizes, recvcounts);
  if (!status.ok()) {
    return status;
  }
  timeline.ActivityEndAll(entries);

  SetDisplacements(recvcounts, displcmnts);
  SetEntryComponentOffsets(entries, entry_component_sizes, recvcounts, entry_component_offsets);

  int element_size = mpi_context_->GetMPITypeSize(first_entry.tensor->dtype());

  int64_t total_size = displcmnts[global_state_->size - 1] +
                       recvcounts[global_state_->size - 1];

  // If shared buffer is not initialized or is not large enough, reallocate
  int64_t total_size_in_bytes = total_size * element_size;
  if (global_state_->shared_buffer == nullptr || global_state_->shared_buffer_size < total_size_in_bytes) {
    if (global_state_->shared_buffer != nullptr) {
      MPI_Win_fence(0, mpi_context_->window);
      MPI_Win_free(&mpi_context_->window);
      global_state_->shared_buffer = nullptr;
    }

    // Allocate shared memory, give each rank their respective pointer
    timeline.ActivityStartAll(entries, ALLOCATE_SHARED_BUFFER);
    int64_t window_size = global_state_->local_rank == 0 ? total_size_in_bytes : 0;
    MPI_Win_allocate_shared(window_size,
                            element_size,
                            MPI_INFO_NULL,
                            mpi_context_->GetMPICommunicator(Communicator::LOCAL),
                            &global_state_->shared_buffer,
                            &mpi_context_->window);
    if (global_state_->local_rank != 0) {
      int disp_unit;
      MPI_Aint winsize;
      MPI_Win_shared_query(mpi_context_->window,
                           0,
                           &winsize,
                           &disp_unit,
                           &global_state_->shared_buffer);
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
  Barrier();
  timeline.ActivityEndAll(entries);

  // Perform the cross-node allgather. If the cluster is homogeneous all
  // local ranks participate, otherwise local rank 0 handles all data
  global_state_->timeline.ActivityStartAll(entries, MPI_CROSS_ALLGATHER);
  if (global_state_->is_homogeneous || global_state_->local_rank == 0) {
    int op = MPI_Allgatherv(MPI_IN_PLACE,
                            0,
                            MPI_DATATYPE_NULL,
                            global_state_->shared_buffer,
                            cross_recvcounts,
                            cross_displcmnts,
                            mpi_context_->GetMPIDataType(first_entry.tensor->dtype()),
                            mpi_context_->GetMPICommunicator(Communicator::CROSS));
    if (op != MPI_SUCCESS) {
      throw std::logic_error("MPI_Allgatherv failed, see MPI output for details.");
    }
  }
  Barrier();
  global_state_->timeline.ActivityEndAll(entries);

  // Copy memory out of the fusion buffer.
  timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
  MemcpyOutFusionBuffer(entry_component_offsets, entry_component_sizes,
                        global_state_->shared_buffer, element_size, entries);
  Barrier();
  timeline.ActivityEndAll(entries);

  // Free the buffers
  delete[] cross_displcmnts;
  delete[] cross_recvcounts;

  return Status::OK();
}

bool MPIHierarchicalAllgather::Enabled(const ParameterManager& param_manager,
                                       const std::vector<TensorTableEntry>& entries,
                                       const Response& response) const {
  return param_manager.HierarchicalAllgather();
}

void MPIHierarchicalAllgather::Barrier() {
  int op = MPI_Barrier(mpi_context_->GetMPICommunicator(Communicator::GLOBAL));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Barrier failed, see MPI output for details.");
  }
}

MPIBroadcast::MPIBroadcast(MPIContext* mpi_context, HorovodGlobalState* global_state)
    : BroadcastOp(global_state), mpi_context_(mpi_context) {}

Status MPIBroadcast::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  assert(entries.size() == 1);
  auto e = entries[0];

  // On root rank, MPI_Bcast sends data, on other ranks it receives data.
  void* data_ptr;
  if (global_state_->rank == e.root_rank) {
    data_ptr = (void*) e.tensor->data();
  } else {
    data_ptr = (void*) e.output->data();
  }

  global_state_->timeline.ActivityStartAll(entries, MPI_BCAST);
  int op = MPI_Bcast(data_ptr,
                     (int) e.tensor->shape().num_elements(),
                     mpi_context_->GetMPIDataType(e.tensor->dtype()),
                     e.root_rank,
                     mpi_context_->GetMPICommunicator(Communicator::GLOBAL));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Broadcast failed, see MPI output for details.");
  }
  global_state_->timeline.ActivityEndAll(entries);

  return Status::OK();
}

bool MPIBroadcast::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

} // namespace common
} // namespace horovod
