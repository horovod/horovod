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

#include "mpi_operations.h"

namespace horovod {
namespace common {

MPIAllreduce::MPIAllreduce(HorovodGlobalState* global_state)
    : AllreduceOp(global_state) {}

Status MPIAllreduce::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  assert(!entries.empty());
  WaitForData(entries);

  auto& first_entry = entries[0];
  auto& process_set =
      global_state_->process_set_table.Get(first_entry.process_set_id);
  const auto& mpi_context = process_set.mpi_context;

  const void* fused_input_data;
  void* buffer_data;
  size_t buffer_len;
  int64_t num_elements = NumElements(entries);

  // Copy memory into the fusion buffer.
  auto& timeline = global_state_->timeline;
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);
    timeline.ActivityEndAll(entries);
  } else {
    fused_input_data = first_entry.tensor->data();
    buffer_data = (void*) first_entry.output->data();
    buffer_len = (size_t) first_entry.output->size();
  }

  if (response.prescale_factor() != 1.0) {
    // Execute prescaling op
    ScaleBuffer(response.prescale_factor(), entries, fused_input_data, buffer_data, num_elements);
    fused_input_data = buffer_data; // for unfused, scale is done out of place
  }

  // Do allreduce.
  timeline.ActivityStartAll(entries, MPI_ALLREDUCE);
  const void* sendbuf = entries.size() > 1 || fused_input_data == buffer_data
                        ? MPI_IN_PLACE : fused_input_data;
  int op =
      MPI_Allreduce(sendbuf, buffer_data, (int)num_elements,
                    mpi_context.GetMPIDataType(first_entry.tensor),
                    mpi_context.GetMPISumOp(first_entry.tensor->dtype()),
                    mpi_context.GetMPICommunicator(Communicator::GLOBAL));
  if (op != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Allreduce failed, see MPI output for details.");
  }
  timeline.ActivityEndAll(entries);

  if (response.postscale_factor() != 1.0) {
    // Execute postscaling op
    ScaleBuffer(response.postscale_factor(), entries, buffer_data, buffer_data, num_elements);
  }

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

MPIAllgather::MPIAllgather(HorovodGlobalState* global_state)
    : AllgatherOp(global_state) {}

bool MPIAllgather::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

Status MPIAllgather::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  assert(!entries.empty());
  WaitForData(entries);

  auto& first_entry = entries[0];
  auto& process_set =
      global_state_->process_set_table.Get(first_entry.process_set_id);
  const auto& mpi_context = process_set.mpi_context;

  auto& timeline = global_state_->timeline;

  // Sizes of subcomponents of each entry from all ranks
  auto** entry_component_sizes = new int64_t* [entries.size()];

  // Offset of each subcomponent of every entry in the final buffer after
  // allgatherv
  auto** entry_component_offsets = new int64_t* [entries.size()];

  int global_size = process_set.controller->GetSize();
  auto* recvcounts = new int[global_size]();
  auto* displcmnts = new int[global_size]();

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    entry_component_sizes[ec] = new int64_t[global_size]();
    entry_component_offsets[ec] = new int64_t[global_size]();
  }

  timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);
  Status status = AllocateOutput(entries, response, entry_component_sizes, recvcounts);
  if (!status.ok()) {
    /* Cleanup */
    for (size_t ec = 0; ec < entries.size(); ++ec) {
      delete[] entry_component_sizes[ec];
      delete[] entry_component_offsets[ec];
    }
    delete[] entry_component_sizes;
    delete[] entry_component_offsets;
    delete[] recvcounts;
    delete[] displcmnts;
    return status;
  }
  timeline.ActivityEndAll(entries);

  SetDisplacements(recvcounts, displcmnts, global_size);
  SetEntryComponentOffsets(entries, entry_component_sizes, recvcounts, entry_component_offsets);

  int element_size = mpi_context.GetMPITypeSize(first_entry.tensor->dtype());

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
  auto dtype = mpi_context.GetMPIDataType(first_entry.tensor->dtype());
  int op = MPI_Allgatherv(sendbuf != nullptr ? sendbuf : MPI_IN_PLACE,
                          (int) total_num_elements,
                          dtype,
                          buffer_data,
                          recvcounts,
                          displcmnts,
                          dtype,
                          mpi_context.GetMPICommunicator(Communicator::GLOBAL));
  if (op != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Allgatherv failed, see MPI output for details.");
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

MPIHierarchicalAllgather::MPIHierarchicalAllgather(
    HorovodGlobalState* global_state)
    : MPIAllgather(global_state) {}

Status MPIHierarchicalAllgather::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  assert(!entries.empty());
  WaitForData(entries);

  auto& first_entry = entries[0];
  auto& process_set =
      global_state_->process_set_table.Get(first_entry.process_set_id);
  auto& mpi_context = process_set.mpi_context;

  auto& timeline = global_state_->timeline;

  // Sizes of subcomponents of each entry from all ranks
  auto** entry_component_sizes = new int64_t* [entries.size()];

  // Offset of each subcomponent of every entry in the final buffer after
  // allgatherv
  auto** entry_component_offsets = new int64_t* [entries.size()];

  int global_size = process_set.controller->GetSize();
  auto* recvcounts = new int[global_size]();
  auto* displcmnts = new int[global_size]();

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    entry_component_sizes[ec] = new int64_t[global_size]();
    entry_component_offsets[ec] = new int64_t[global_size]();
  }

  timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);
  Status status = AllocateOutput(entries, response, entry_component_sizes, recvcounts);
  if (!status.ok()) {
    /* Cleanup */
    for (size_t ec = 0; ec < entries.size(); ++ec) {
      delete[] entry_component_sizes[ec];
      delete[] entry_component_offsets[ec];
    }
    delete[] entry_component_sizes;
    delete[] entry_component_offsets;
    delete[] recvcounts;
    delete[] displcmnts;
    return status;
  }
  timeline.ActivityEndAll(entries);

  SetDisplacements(recvcounts, displcmnts, global_size);
  SetEntryComponentOffsets(entries, entry_component_sizes, recvcounts, entry_component_offsets);

  int element_size = mpi_context.GetMPITypeSize(first_entry.tensor->dtype());

  int64_t total_size = displcmnts[global_size - 1] +
                       recvcounts[global_size - 1];

  // If shared buffer is not initialized or is not large enough, reallocate
  int64_t total_size_in_bytes = total_size * element_size;
  if (process_set.shared_buffer == nullptr ||
      process_set.shared_buffer_size < total_size_in_bytes) {
    if (process_set.shared_buffer != nullptr) {
      MPI_Win_fence(0, mpi_context.window);
      MPI_Win_free(&mpi_context.window);
      process_set.shared_buffer = nullptr;
    }

    // Allocate shared memory, give each rank their respective pointer
    timeline.ActivityStartAll(entries, ALLOCATE_SHARED_BUFFER);
    int64_t window_size = process_set.controller->GetLocalRank() == 0 ? total_size_in_bytes : 0;
    MPI_Win_allocate_shared(window_size,
                            element_size,
                            MPI_INFO_NULL,
                            mpi_context.GetMPICommunicator(Communicator::LOCAL),
                            &process_set.shared_buffer,
                            &mpi_context.window);
    if (process_set.controller->GetLocalRank() != 0) {
      int disp_unit;
      MPI_Aint winsize;
      MPI_Win_shared_query(mpi_context.window,
                           0,
                           &winsize,
                           &disp_unit,
                           &process_set.shared_buffer);
    }
    process_set.shared_buffer_size = total_size_in_bytes;
    timeline.ActivityEndAll(entries);
  }

  // Compute cross-node allgather displacements and recvcounts for
  // homogeneous/parallelized case
  int cross_size = process_set.controller->GetCrossSize();
  int local_size = process_set.controller->GetLocalSize();
  int local_rank = process_set.controller->GetLocalRank();
  auto* cross_recvcounts = new int[cross_size]();
  auto* cross_displcmnts = new int[cross_size]();

  if (process_set.controller->IsHomogeneous()) {
    for (int i = 0; i < process_set.controller->GetCrossSize(); ++i) {
      cross_recvcounts[i] = recvcounts[local_size * i + local_rank];
      cross_displcmnts[i] = displcmnts[local_size * i + local_rank];
    }
  } else if (process_set.controller->GetLocalRank() == 0) {
    // In this case local rank 0 will allgather with all local data
    int offset = 0;
    for (int i = 0; i < cross_size; ++i) {
      for (int j = offset; j < offset + process_set.controller->GetLocalSizeAtCrossRank(i);
           ++j) {
        cross_recvcounts[i] += recvcounts[j];
      }
      cross_displcmnts[i] = displcmnts[offset];
      offset += process_set.controller->GetLocalSizeAtCrossRank(i);
    }
  }

  timeline.ActivityStartAll(entries, MEMCPY_IN_SHARED_BUFFER);

  int rank = process_set.controller->GetRank();
  for (size_t ec = 0; ec < entries.size(); ++ec) {
    auto& e = entries[ec];
    void* shared_buffer_at_offset =
        (uint8_t*)process_set.shared_buffer +
        entry_component_offsets[ec][rank] * element_size;

    // CPU copy to shared buffer
    memcpy(shared_buffer_at_offset, e.tensor->data(),
           (size_t) (entry_component_sizes[ec][rank] * element_size));
  }
  Barrier(process_set);
  timeline.ActivityEndAll(entries);

  // Perform the cross-node allgather. If the cluster is homogeneous all
  // local ranks participate, otherwise local rank 0 handles all data
  global_state_->timeline.ActivityStartAll(entries, MPI_CROSS_ALLGATHER);
  if (process_set.controller->IsHomogeneous() || process_set.controller->GetLocalRank() == 0) {
    int op = MPI_Allgatherv(MPI_IN_PLACE,
                            0,
                            MPI_DATATYPE_NULL,
                            process_set.shared_buffer,
                            cross_recvcounts,
                            cross_displcmnts,
                            mpi_context.GetMPIDataType(first_entry.tensor->dtype()),
                            mpi_context.GetMPICommunicator(Communicator::CROSS));
    if (op != MPI_SUCCESS) {
      throw std::runtime_error("MPI_Allgatherv failed, see MPI output for details.");
    }
  }
  Barrier(process_set);
  global_state_->timeline.ActivityEndAll(entries);

  // Copy memory out of the fusion buffer.
  timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
  MemcpyOutFusionBuffer(entry_component_offsets, entry_component_sizes,
                        process_set.shared_buffer, element_size, entries);
  Barrier(process_set);
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

void MPIHierarchicalAllgather::Barrier(const ProcessSet& process_set) {
  int op =
      MPI_Barrier(process_set.mpi_context.GetMPICommunicator(Communicator::GLOBAL));
  if (op != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Barrier failed, see MPI output for details.");
  }
}

MPIBroadcast::MPIBroadcast(HorovodGlobalState* global_state)
    : BroadcastOp(global_state) {}

Status MPIBroadcast::Execute(std::vector<TensorTableEntry>& entries,
                             const Response& response) {
  WaitForData(entries);

  assert(entries.size() == 1);
  auto e = entries[0];
  auto& process_set = global_state_->process_set_table.Get(e.process_set_id);
  const auto& mpi_context = process_set.mpi_context;

  // On root rank, MPI_Bcast sends data, on other ranks it receives data.
  void* data_ptr;
  if (process_set.controller->GetRank() == e.root_rank) {
    data_ptr = (void*) e.tensor->data();
  } else {
    data_ptr = (void*) e.output->data();
  }

  global_state_->timeline.ActivityStartAll(entries, MPI_BCAST);
  int op = MPI_Bcast(data_ptr,
                     (int) e.tensor->shape().num_elements(),
                     mpi_context.GetMPIDataType(e.tensor->dtype()),
                     e.root_rank,
                     mpi_context.GetMPICommunicator(Communicator::GLOBAL));
  if (op != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Broadcast failed, see MPI output for details.");
  }
  global_state_->timeline.ActivityEndAll(entries);

  return Status::OK();
}

bool MPIBroadcast::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

MPIAlltoall::MPIAlltoall(HorovodGlobalState* global_state)
    : AlltoallOp(global_state) {}

Status MPIAlltoall::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  WaitForData(entries);

  assert(entries.size() == 1);
  auto e = entries[0];
  auto& process_set = global_state_->process_set_table.Get(e.process_set_id);
  const auto& mpi_context = process_set.mpi_context;

  std::vector<int32_t> sdispls, rdispls;
  std::vector<int32_t> sendcounts, recvcounts;
  Status status = PrepareOutputAndParams(e, sdispls, rdispls, sendcounts, recvcounts);
  if (!status.ok()) {
    return status;
  }

  const void* sendbuf = e.tensor->data();
  void* buffer_data = (void*) e.output->data();
  global_state_->timeline.ActivityStartAll(entries, MPI_ALLTOALL);

  int op = MPI_Alltoallv(sendbuf, sendcounts.data(), sdispls.data(),
                         mpi_context.GetMPIDataType(e.tensor->dtype()),
                         buffer_data, recvcounts.data(), rdispls.data(),
                         mpi_context.GetMPIDataType(e.output->dtype()),
                         mpi_context.GetMPICommunicator(Communicator::GLOBAL));
  if (op != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Alltoallv failed, see MPI output for details.");
  }
  global_state_->timeline.ActivityEndAll(entries);

  return Status::OK();
}

bool MPIAlltoall::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

} // namespace common
} // namespace horovod
