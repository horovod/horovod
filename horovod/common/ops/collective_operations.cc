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

#include "collective_operations.h"

namespace horovod {
namespace common {

HorovodOp::HorovodOp(HorovodGlobalState* global_state) : global_state_(global_state) {}

// Allreduce
AllreduceOp::AllreduceOp(HorovodGlobalState* global_state) : HorovodOp(global_state) {}

Status AllreduceOp::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& first_entry = entries[0];

  Initialize(entries, response);

  const void* fused_input_data;
  void* buffer_data;
  size_t buffer_len;

  // Copy memory into the fusion buffer.
  if (entries.size() > 1) {
    // Access the fusion buffer.
    auto& buffer = global_state_->fusion_buffer.GetBuffer(
        first_entry.device, first_entry.context->framework());
    buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));

    StartMemcpyInFusionBuffer(entries);
    int64_t offset = 0;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*) buffer_data + offset;
      MemcpyInFusionBuffer(buffer_data_at_offset, e, entries);
      offset += e.tensor->size();
    }
    EndMemcpyInFusionBuffer(entries);

    buffer_len = (size_t) offset;

    // Set the input data to originate from the buffer.
    fused_input_data = buffer_data;
  } else {
    fused_input_data = first_entry.tensor->data();
    buffer_data = (void*) first_entry.output->data();
    buffer_len = (size_t) first_entry.output->size();
  }

  int64_t num_elements = 0;
  for (auto& e : entries) {
    num_elements += e.tensor->shape().num_elements();
  }

  // Do allreduce.
  DoAllreduce(entries, fused_input_data, buffer_data, num_elements, buffer_len);

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    StartMemcpyOutFusionBuffer(entries);
    int64_t offset = 0;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*) buffer_data + offset;
      MemcpyOutFusionBuffer(buffer_data_at_offset, e, entries);
      offset += e.tensor->size();
    }
    EndMemcpyOutFusionBuffer(entries);
  }

  return Finalize(entries);
}

void AllreduceOp::Initialize(std::vector<TensorTableEntry>& entries, const Response& response) {
}

Status AllreduceOp::Finalize(std::vector<TensorTableEntry>& entries) {
  return Status::OK();
}

void AllreduceOp::StartMemcpyInFusionBuffer(std::vector<TensorTableEntry>& entries) {
  global_state_->timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
}

void AllreduceOp::MemcpyInFusionBuffer(void* buffer_data_at_offset, TensorTableEntry& e,
                                       std::vector<TensorTableEntry>& entries) {
  std::memcpy(buffer_data_at_offset, e.tensor->data(),
              (size_t) e.tensor->size());
}

void AllreduceOp::EndMemcpyInFusionBuffer(std::vector<TensorTableEntry>& entries) {
  global_state_->timeline.ActivityEndAll(entries);
}

void AllreduceOp::StartMemcpyOutFusionBuffer(std::vector<TensorTableEntry>& entries) {
  global_state_->timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
}

void AllreduceOp::MemcpyOutFusionBuffer(void* buffer_data_at_offset, TensorTableEntry& e,
                                        std::vector<TensorTableEntry>& entries) {
  std::memcpy((void*) e.output->data(), buffer_data_at_offset,
              (size_t) e.tensor->size());
}

void AllreduceOp::EndMemcpyOutFusionBuffer(std::vector<TensorTableEntry>& entries) {
  global_state_->timeline.ActivityEndAll(entries);
}

// Allgather
AllgatherOp::AllgatherOp(HorovodGlobalState* global_state) : HorovodOp(global_state) {}

Status AllgatherOp::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
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
  for (size_t ec = 0; ec < entries.size(); ++ec) {
    auto& e = entries[ec];
    // Every tensor participating in Allgather operation may have different
    // first dimension size, but the rest of dimensions are same for all
    // tensors.  Here we get shape of tensor sliced by first dimension.
    TensorShape single_slice_shape;
    for (int i = 1; i < e.tensor->shape().dims(); ++i) {
      single_slice_shape.AddDim(e.tensor->shape().dim_size(i));
    }

    // Copy tensor sizes from the MPI response into a vector of int64_t
    // and compute total size.  This is size of first dimension.
    int64_t total_entry_dimension_size = 0;
    const auto& tensor_sizes = response.tensor_sizes();
    for (int rc = 0; rc < global_state_->size; ++rc) {
      auto component_size = tensor_sizes[ec * global_state_->size + rc];
      total_entry_dimension_size += component_size;
      recvcounts[rc] += component_size * single_slice_shape.num_elements();
      entry_component_sizes[ec][rc] =
          component_size * single_slice_shape.num_elements();
    }

    // Allgather output will have shape of:
    // (sum of first dimension of every tensor) x (tensor slice shape).
    TensorShape output_shape;
    output_shape.AddDim((int64_t) total_entry_dimension_size);
    output_shape.AppendShape(single_slice_shape);

    Status status = e.context->AllocateOutput(output_shape, &e.output);
    if (!status.ok()) {
      return status;
    }
  }
  timeline.ActivityEndAll(entries);

  for (int rc = 0; rc < global_state_->size; ++rc) {
    if (rc == 0) {
      displcmnts[rc] = 0;
    } else {
      displcmnts[rc] = displcmnts[rc - 1] + recvcounts[rc - 1];
    }
  }

  unsigned int rank_displacement = 0;
  for (int rc = 0; rc < global_state_->size; ++rc) {
    for (size_t ec = 0; ec < entries.size(); ++ec) {
      if (ec == 0) {
        entry_component_offsets[ec][rc] = rank_displacement;
      } else {
        entry_component_offsets[ec][rc] =
            entry_component_offsets[ec - 1][rc] +
            entry_component_sizes[ec - 1][rc];
      }
    }
    rank_displacement += recvcounts[rc];
  }

  int element_size = GetElementSize(first_entry.tensor->dtype());
  int64_t total_size = displcmnts[global_state_->size - 1] +
                       recvcounts[global_state_->size - 1];

  DoAllgather(entries, recvcounts, displcmnts,
              entry_component_offsets, entry_component_sizes,
              total_size, element_size);

  return Status::OK();
}

void AllgatherOp::DoAllgather(std::vector<TensorTableEntry>& entries, int* recvcounts, int* displcmnts,
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

  DoAllgatherv(entries, sendbuf, (int) total_num_elements,
               first_entry.tensor->dtype(),
               buffer_data, recvcounts, displcmnts,
               first_entry.tensor->dtype());

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

BroadcastOp::BroadcastOp(HorovodGlobalState* global_state) : HorovodOp(global_state) {}

Status BroadcastOp::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  assert(entries.size() == 1);
  auto e = entries[0];

  // On root rank, MPI_Bcast sends data, on other ranks it receives data.
  void* data_ptr;
  if (global_state_->rank == e.root_rank) {
    data_ptr = (void*) e.tensor->data();
  } else {
    data_ptr = (void*) e.output->data();
  }

  DoBroadcast(entries, data_ptr, (int) e.tensor->shape().num_elements(), e.tensor->dtype(), e.root_rank);

  return Status::OK();
}

bool BroadcastOp::Enabled(ParameterManager& param_manager,
                          std::vector<TensorTableEntry>& entries,
                          const Response& response) const {
  return true;
}

ErrorOp::ErrorOp(HorovodGlobalState* global_state) : HorovodOp(global_state) {}

Status ErrorOp::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  assert(entries.size() == 1);
  auto e = entries[0];
  return Status::PreconditionError(response.error_message());
}

HierarchicalAllgather::HierarchicalAllgather(HorovodGlobalState* global_state) : AllgatherOp(global_state) {}

void HierarchicalAllgather::DoAllgather(std::vector<TensorTableEntry>& entries, int* recvcounts, int* displcmnts,
                                        int64_t** entry_component_offsets, int64_t** entry_component_sizes,
                                        int64_t total_size, int element_size) {
  auto& timeline = global_state_->timeline;

  // If shared buffer is not initialized or is not large enough, reallocate
  int64_t total_size_in_bytes = total_size * element_size;
  if (global_state_->shared_buffer == nullptr || global_state_->shared_buffer_size < total_size_in_bytes) {
    FreeSharedBuffer();

    // Allocate shared memory, give each rank their respective pointer
    timeline.ActivityStartAll(entries, ALLOCATE_SHARED_BUFFER);
    AllocateSharedBuffer(total_size_in_bytes, element_size);
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

  auto& first_entry = entries[0];
  DoAllgatherv(entries,
               nullptr, 0, DataType::HOROVOD_NULL, global_state_->shared_buffer,
               cross_recvcounts, cross_displcmnts, first_entry.tensor->dtype());

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
  Barrier();
  timeline.ActivityEndAll(entries);

  // Free the buffers
  delete[] cross_displcmnts;
  delete[] cross_recvcounts;
}

} // namespace common
} // namespace horovod
