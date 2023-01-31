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

#include "collective_operations.h"
#include "../message.h"

namespace horovod {
namespace common {

HorovodOp::HorovodOp(HorovodGlobalState* global_state)
    : global_state_(global_state) {}

int64_t HorovodOp::NumElements(std::vector<TensorTableEntry>& entries) {
  int64_t num_elements = 0;
  for (auto& e : entries) {
    num_elements += e.tensor->shape().num_elements();
  }
  return num_elements;
}

void HorovodOp::WaitForData(std::vector<TensorTableEntry>& entries) {
  // On GPU data readiness is signalled by ready_event.
  auto& timeline = global_state_->timeline;
  std::vector<TensorTableEntry> waiting_tensors;
  for (auto& e : entries) {
    if (e.ready_event_list.size() != 0) {
      timeline.ActivityStart(e.tensor_name, WAIT_FOR_DATA);
      waiting_tensors.push_back(e);
    }
  }
  while (!waiting_tensors.empty()) {
    for (auto it = waiting_tensors.begin(); it != waiting_tensors.end();) {
      if (it->ready_event_list.Ready()) {
        timeline.ActivityEnd(it->tensor_name);
        timeline.ActivityStart(it->tensor_name, WAIT_FOR_OTHER_TENSOR_DATA);
        it = waiting_tensors.erase(it);
      } else {
        ++it;
      }
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(100));
  }
  for (auto& e : entries) {
    if (e.ready_event_list.size() != 0) {
      timeline.ActivityEnd(e.tensor_name);
    }
  }
}

void HorovodOp::ScaleBuffer(double scale_factor,
                            const std::vector<TensorTableEntry>& entries,
                            const void* fused_input_data, void* buffer_data,
                            int64_t num_elements) {
  DataType dtype = entries[0].tensor->dtype();
  switch (dtype) {
  case HOROVOD_UINT8:
    ScaleBufferCPUImpl((const uint8_t*)fused_input_data, (uint8_t*)buffer_data,
                       num_elements, scale_factor);
    break;
  case HOROVOD_INT8:
    ScaleBufferCPUImpl((const int8_t*)fused_input_data, (int8_t*)buffer_data,
                       num_elements, scale_factor);
    break;
  case HOROVOD_INT32:
    ScaleBufferCPUImpl((const int32_t*)fused_input_data, (int32_t*)buffer_data,
                       num_elements, scale_factor);
    break;
  case HOROVOD_INT64:
    ScaleBufferCPUImpl((const int64_t*)fused_input_data, (int64_t*)buffer_data,
                       num_elements, scale_factor);
    break;
  case HOROVOD_FLOAT16:
    ScaleBufferCPUImpl((const unsigned short*)fused_input_data,
                       (unsigned short*)buffer_data, num_elements,
                       (float)scale_factor);
    break;
  case HOROVOD_FLOAT32:
    ScaleBufferCPUImpl((const float*)fused_input_data, (float*)buffer_data,
                       num_elements, (float)scale_factor);
    break;
  case HOROVOD_FLOAT64:
    ScaleBufferCPUImpl((const double*)fused_input_data, (double*)buffer_data,
                       num_elements, scale_factor);
    break;
  default:
    throw std::logic_error("Type " + DataType_Name(dtype) +
                           " not supported by ScaleBufferCPUImpl.");
  }
}

// Allreduce
AllreduceOp::AllreduceOp(HorovodGlobalState* global_state)
    : HorovodOp(global_state) {}

void AllreduceOp::MemcpyInFusionBuffer(
    const std::vector<TensorTableEntry>& entries, const void*& fused_input_data,
    void*& buffer_data, size_t& buffer_len) {
  // Access the fusion buffer.
  auto& first_entry = entries[0];
  auto buffer = global_state_->fusion_buffer.GetBuffer(
      first_entry.device, first_entry.context->framework(), global_state_->current_nccl_stream);
  buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));

  int64_t offset = 0;
  for (auto& e : entries) {
    void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
    MemcpyEntryInFusionBuffer(entries, e, buffer_data_at_offset);
    offset += e.tensor->size();
  }

  buffer_len = (size_t)offset;

  // Set the input data to originate from the buffer.
  fused_input_data = buffer_data;
}

void AllreduceOp::MemcpyOutFusionBuffer(
    const void* buffer_data, std::vector<TensorTableEntry>& entries) {
  int64_t offset = 0;
  for (auto& e : entries) {
    void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
    MemcpyEntryOutFusionBuffer(entries, buffer_data_at_offset, e);
    offset += e.output->size();
  }
}

void AllreduceOp::MemcpyEntryInFusionBuffer(
    const std::vector<TensorTableEntry>& entries, const TensorTableEntry& e,
    void* buffer_data_at_offset) {
  std::memcpy(buffer_data_at_offset, e.tensor->data(),
              (size_t)e.tensor->size());
}

void AllreduceOp::MemcpyEntryOutFusionBuffer(
    const std::vector<TensorTableEntry>& entries,
    const void* buffer_data_at_offset, TensorTableEntry& e) {
  std::memcpy((void*)e.output->data(), buffer_data_at_offset,
              (size_t)e.output->size());
}

// Allgather
AllgatherOp::AllgatherOp(HorovodGlobalState* global_state)
    : HorovodOp(global_state) {}

Status AllgatherOp::AllocateOutput(std::vector<TensorTableEntry>& entries,
                                   const Response& response,
                                   int64_t**& entry_component_sizes) {
  for (size_t ec = 0; ec < entries.size(); ++ec) {
    auto& e = entries[ec];
    auto& process_set = global_state_->process_set_table.Get(e.process_set_id);
    int global_size = process_set.controller->GetSize();
    // Every tensor participating in Allgather operation may have different
    // first dimension size, but the rest of dimensions are same for all
    // tensors.  Here we get shape of tensor sliced by first dimension.
    TensorShape single_slice_shape;
    for (int i = 1; i < e.tensor->shape().dims(); ++i) {
      single_slice_shape.AddDim(e.tensor->shape().dim_size(i));
    }

    // Copy tensor sizes from the response into a vector of int64_t
    // and compute total size.  This is size of first dimension.
    int64_t total_entry_dimension_size = 0;
    const auto& tensor_sizes = response.tensor_sizes();
    for (int rc = 0; rc < global_size; ++rc) {
      auto component_size = tensor_sizes[ec * global_size + rc];
      total_entry_dimension_size += component_size;

      if (entry_component_sizes) {
        entry_component_sizes[ec][rc] =
            component_size * single_slice_shape.num_elements();
      }
    }

    // Allgather output will have shape of:
    // (sum of first dimension of every tensor) x (tensor slice shape).
    TensorShape output_shape;
    output_shape.AddDim((int64_t)total_entry_dimension_size);
    output_shape.AppendShape(single_slice_shape);

    Status status =
        e.context->AllocateOutput(e.output_index, output_shape, &e.output);
    if (!status.ok()) {
      LOG(WARNING) << "AllgatherOp::AllocateOutput failed: " << status.reason();
      return status;
    }
  }

  return Status::OK();
}

void AllgatherOp::SetRecvcounts(const int64_t* const* entry_component_sizes,
                                size_t num_entries, int global_size,
                                int*& recvcounts, int rank_padding_elements) {
  assert(num_entries > 0);
  for (int rc = 0; rc < global_size; ++rc) {
    recvcounts[rc] = (int)entry_component_sizes[0][rc];
    for (size_t ec = 1; ec < num_entries; ++ec) {
      recvcounts[rc] += (int)entry_component_sizes[ec][rc];
    }
    recvcounts[rc] =
        rank_padding_elements *
        ((recvcounts[rc] + rank_padding_elements - 1) / rank_padding_elements);
  }
}

void AllgatherOp::SetDisplacements(const int* recvcounts, int*& displcmnts,
                                   int global_size) {
  for (int rc = 0; rc < global_size; ++rc) {
    if (rc == 0) {
      displcmnts[rc] = 0;
    } else {
      displcmnts[rc] = displcmnts[rc - 1] + recvcounts[rc - 1];
    }
  }
}

void AllgatherOp::SetEntryComponentOffsets(
    const int64_t* const* entry_component_sizes, const int* recvcounts,
    size_t num_entries, int global_size, int64_t**& entry_component_offsets) {
  unsigned int rank_displacement = 0;
  for (int rc = 0; rc < global_size; ++rc) {
    for (size_t ec = 0; ec < num_entries; ++ec) {
      if (ec == 0) {
        entry_component_offsets[ec][rc] = rank_displacement;
      } else {
        entry_component_offsets[ec][rc] = entry_component_offsets[ec - 1][rc] +
                                          entry_component_sizes[ec - 1][rc];
      }
    }
    rank_displacement += recvcounts[rc];
  }
}

void AllgatherOp::MemcpyInFusionBuffer(
    const std::vector<TensorTableEntry>& entries, const int* displcmnts,
    int element_size, void*& buffer_data) {
  assert(!entries.empty());
  // Access the fusion buffer.
  auto& first_entry = entries[0];
  auto buffer = global_state_->fusion_buffer.GetBuffer(
      first_entry.device, first_entry.context->framework(), global_state_->current_nccl_stream);
  buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));

  auto& process_set =
      global_state_->process_set_table.Get(first_entry.process_set_id);
  int64_t offset = (int64_t)displcmnts[process_set.controller->GetRank()] * (int64_t)element_size;
  for (auto& e : entries) {
    void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
    MemcpyEntryInFusionBuffer(entries, e, buffer_data_at_offset);
    offset += e.tensor->size();
  }
}

void AllgatherOp::MemcpyOutFusionBuffer(
    const int64_t* const* entry_component_offsets,
    const int64_t* const* entry_component_sizes, const void* buffer_data,
    int element_size, std::vector<TensorTableEntry>& entries) {
  // Copy memory out of the fusion buffer.
  for (size_t ec = 0; ec < entries.size(); ++ec) {
    auto& e = entries[ec];
    auto& process_set = global_state_->process_set_table.Get(e.process_set_id);
    int global_size = process_set.controller->GetSize();
    int64_t copy_offset = 0;
    for (int rc = 0; rc < global_size; ++rc) {
      int64_t entry_offset = entry_component_offsets[ec][rc] * element_size;
      int64_t entry_size = entry_component_sizes[ec][rc] * element_size;
      const void* buffer_data_at_offset = (uint8_t*)buffer_data + entry_offset;
      MemcpyEntryOutFusionBuffer(entries, buffer_data_at_offset, e,
                                 copy_offset, entry_size);
      copy_offset += entry_size;
    }
  }
}

void AllgatherOp::MemcpyEntryInFusionBuffer(
    const std::vector<TensorTableEntry>& entries, const TensorTableEntry& e,
    void* buffer_data_at_offset) {
  std::memcpy(buffer_data_at_offset, e.tensor->data(),
              (size_t)e.tensor->size());
}

void AllgatherOp::MemcpyEntryOutFusionBuffer(
    const std::vector<TensorTableEntry>& entries,
    const void* buffer_data_at_offset, TensorTableEntry& e,
    int64_t entry_offset, size_t entry_size) {
  std::memcpy((uint8_t*)e.output->data() + entry_offset,
              buffer_data_at_offset, entry_size);
}

BroadcastOp::BroadcastOp(HorovodGlobalState* global_state)
    : HorovodOp(global_state) {}

AlltoallOp::AlltoallOp(HorovodGlobalState* global_state)
    : HorovodOp(global_state) {}

// Reducescatter
ReducescatterOp::ReducescatterOp(HorovodGlobalState* global_state)
    : HorovodOp(global_state) {}

TensorShape ReducescatterOp::ComputeOutputShapeForRank(
    const TensorShape& tensor_shape, int rank, int global_size) const {
  // If tensor_shape.dim_size(0) % global_size != 0, the first ranks 0, ..., tensor_shape.dim_size(0) % global_size - 1
  // may receive a slightly larger tensor.
  int64_t min_size = tensor_shape.dim_size(0) / global_size;
  int64_t max_size = tensor_shape.dim_size(0) / global_size + 1;
  int64_t component_size = rank < tensor_shape.dim_size(0) % global_size ? max_size : min_size;

  TensorShape output_shape;
  output_shape.AddDim(component_size);
  for (int i = 1; i < tensor_shape.dims(); ++i) {
    output_shape.AddDim(tensor_shape.dim_size(i));
  }

  return output_shape;
}

std::vector<std::vector<TensorShape>> ReducescatterOp::ComputeOutputShapes(
    const std::vector<TensorTableEntry>& entries, int global_size) const {
  std::vector<std::vector<TensorShape>> output_shapes(global_size);
  for (int rank = 0; rank < global_size; ++rank) {
    output_shapes[rank].reserve(entries.size());
    for (const auto& e : entries) {
      TensorShape shape =
          ComputeOutputShapeForRank(e.tensor->shape(), rank, global_size);
      output_shapes[rank].emplace_back(std::move(shape));
    }
  }

  return output_shapes;
}

std::vector<int> ReducescatterOp::ComputeReceiveCounts(
    const std::vector<std::vector<TensorShape>>& output_shapes) const {
  std::vector<int> recvcounts(output_shapes.size(), 0);
  for (size_t rank = 0; rank < output_shapes.size(); ++rank) {
    const auto& rank_shapes = output_shapes[rank];
    for (const TensorShape& shape : rank_shapes) {
      recvcounts[rank] += static_cast<int>(shape.num_elements());
    }
  }

  return recvcounts;
}

Status
ReducescatterOp::AllocateOutput(std::vector<TensorTableEntry>& entries,
                                const std::vector<TensorShape>& output_shapes) {
  for (size_t ec = 0; ec < entries.size(); ++ec) {
    auto& e = entries[ec];
    const auto& output_shape = output_shapes[ec];
    Status status =
        e.context->AllocateOutput(e.output_index, output_shape, &e.output);
    if (!status.ok()) {
      LOG(WARNING) << "ReducescatterOp::AllocateOutput failed: "
                   << status.reason();
      return status;
    }
  }

  return Status::OK();
}

void ReducescatterOp::MemcpyInFusionBuffer(
    const std::vector<TensorTableEntry>& entries,
    const std::vector<std::vector<TensorShape>>& output_shapes,
    std::size_t element_size, void*& buffer_data,
    size_t& buffer_len) {
  // Access the fusion buffer.
  auto& first_entry = entries[0];
  auto buffer = global_state_->fusion_buffer.GetBuffer(
      first_entry.device, first_entry.context->framework(),
      global_state_->current_nccl_stream);
  buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));

  size_t buffer_offset = 0;
  std::vector<size_t> entry_offsets(entries.size(), 0);

  for (const auto& rank_shapes : output_shapes) {
    for (size_t ec = 0; ec < entries.size(); ++ec) {
      auto& e = entries[ec];
      const auto& entry_shape = rank_shapes[ec];
      auto entry_offset = entry_offsets[ec];
      size_t entry_size = entry_shape.num_elements() * element_size;
      void* buffer_data_at_offset = (uint8_t*)buffer_data + buffer_offset;
      MemcpyEntryInFusionBuffer(e, entry_offset, entry_size,
                                buffer_data_at_offset);
      entry_offsets[ec] += entry_size;
      buffer_offset += entry_size;
    }
  }
  buffer_len = buffer_offset;
}

void ReducescatterOp::MemcpyOutFusionBuffer(
    const void* buffer_data, std::vector<TensorTableEntry>& entries) {
  int64_t offset = 0;
  for (auto& e : entries) {
    void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
    MemcpyEntryOutFusionBuffer(buffer_data_at_offset, e);
    offset += e.output->size();
  }
}

void ReducescatterOp::MemcpyEntryInFusionBuffer(const TensorTableEntry& e,
                                                size_t entry_offset,
                                                size_t entry_size,
                                                void* buffer_data_at_offset) {
  void* tensor_data_at_offset = (uint8_t*)e.tensor->data() + entry_offset;
  std::memcpy(buffer_data_at_offset, tensor_data_at_offset, entry_size);
}

void ReducescatterOp::MemcpyEntryOutFusionBuffer(
    const void* buffer_data_at_offset, TensorTableEntry& e) {
  std::memcpy((void*)e.output->data(), buffer_data_at_offset,
              (size_t)e.output->size());
}

// Join
JoinOp::JoinOp(HorovodGlobalState* global_state) : HorovodOp(global_state) {}

Status JoinOp::Execute(std::vector<TensorTableEntry>& entries,
                       const Response& response, ProcessSet& process_set) {
  WaitForData(entries);

  assert(entries.size() == 1);
  auto e = entries[0];
  auto output_ptr = (int*) e.output->data();
  *output_ptr = response.last_joined_rank();
  if (process_set.joined) {
    process_set.tensor_queue.RemoveJoinTensor();
    process_set.joined = false;
    process_set.last_joined_rank = -1;
  }
  return Status::OK();
}

// Barrier
BarrierOp::BarrierOp(HorovodGlobalState* global_state) : HorovodOp(global_state) {}

Status BarrierOp::Execute(std::vector<TensorTableEntry>& entries,
                       const Response& response) {
  assert(entries.size() == 1);
  int& process_set_id = entries[0].process_set_id;
  auto& process_set = global_state_->process_set_table.Get(process_set_id);


  process_set.controller->Barrier(Communicator::GLOBAL);
  LOG(TRACE, global_state_->global_controller->GetRank()) << "Released from barrier.";

  return Status::OK();
}

// Error
ErrorOp::ErrorOp(HorovodGlobalState* global_state) : HorovodOp(global_state) {}

Status ErrorOp::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  return Status::PreconditionError(response.error_message());
}

} // namespace common
} // namespace horovod
