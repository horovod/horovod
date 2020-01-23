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

#include "gpu_operations.h"

#include <thread>

namespace horovod {
namespace common {

GPUOpContext::GPUOpContext(GPUContext* context, HorovodGlobalState* global_state)
    : gpu_context_(context), global_state_(global_state) {}

void GPUOpContext::InitGPU(const std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  gpu_context_->SetDevice(first_entry.device);

  // Ensure stream is in the map before executing reduction.
  gpuStream_t& stream = gpu_context_->streams[global_state_->current_nccl_stream][first_entry.device];
  if (stream == nullptr) {
    gpu_context_->StreamCreate(&stream);
  }
}

void GPUOpContext::InitGPUQueue(const std::vector<TensorTableEntry>& entries, const Response& response) {
  event_queue = std::queue<std::pair<std::string, gpuEvent_t>>();
  stream = &gpu_context_->streams[global_state_->current_nccl_stream][entries[0].device];

  if (global_state_->timeline.Initialized()) {
    gpu_context_->RecordEvent(event_queue, QUEUE, *stream);
  }
}

Status GPUOpContext::FinalizeGPUQueue(const std::vector<TensorTableEntry>& entries, bool free_host_buffer /*= true*/) {
  // Use completion marker via event because it's faster than
  // blocking gpuStreamSynchronize() in this thread.
  gpu_context_->RecordEvent(event_queue, "", *stream);

  auto& first_entry = entries[0];
  void* cpu_buffer = host_buffer;
  auto& evt_queue = event_queue;
  auto& timeline = global_state_->timeline;
  auto& gpu_context = gpu_context_;

  // Claim a std::shared_ptr to the fusion buffer to prevent its memory from being reclaimed
  // during finalization.
  auto fusion_buffer = global_state_->fusion_buffer.GetBuffer(
      first_entry.device, first_entry.context->framework(), global_state_->current_nccl_stream);

  gpu_context_->finalizer_thread_pool.execute([entries, first_entry, cpu_buffer, fusion_buffer, free_host_buffer,
                                                evt_queue, &timeline, &gpu_context]() mutable {
    gpu_context->SetDevice(first_entry.device);

    gpu_context->WaitForEvents(evt_queue, entries, timeline);
    if (free_host_buffer && cpu_buffer != nullptr) {
      free(cpu_buffer);
    }

    for (auto& e : entries) {
      timeline.End(e.tensor_name, e.output);
      // Callback can be null if the rank sent Join request.
      if (e.callback != nullptr) {
        e.callback(Status::OK());
      }
    }
  });

  // Update current stream
  global_state_->current_nccl_stream = (global_state_->current_nccl_stream + 1) %
                                  global_state_->num_nccl_streams;

  return Status::InProgress();
}

GPUAllreduce::GPUAllreduce(GPUContext* context, HorovodGlobalState* global_state)
    : AllreduceOp(global_state), gpu_context_(context), gpu_op_context_(context, global_state) {}

bool GPUAllreduce::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}

void GPUAllreduce::MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                             const TensorTableEntry& e, void* buffer_data_at_offset) {
  auto& first_entry = entries[0];
  gpu_context_->MemcpyAsyncD2D(buffer_data_at_offset, e.tensor->data(), (size_t) e.tensor->size(),
                               gpu_context_->streams[global_state_->current_nccl_stream][first_entry.device]);
}

void GPUAllreduce::MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                               const void* buffer_data_at_offset, TensorTableEntry& e) {
  auto& first_entry = entries[0];
  gpu_context_->MemcpyAsyncD2D((void*) e.output->data(), buffer_data_at_offset, (size_t) e.tensor->size(),
                               gpu_context_->streams[global_state_->current_nccl_stream][first_entry.device]);
}

GPUAllgather::GPUAllgather(GPUContext* context, HorovodGlobalState* global_state)
    : AllgatherOp(global_state), gpu_context_(context), gpu_op_context_(context, global_state) {}

bool GPUAllgather::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}

void GPUAllgather::MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                             const TensorTableEntry& e, void* buffer_data_at_offset) {
  auto& first_entry = entries[0];
  gpu_context_->MemcpyAsyncD2D(buffer_data_at_offset, e.tensor->data(), (size_t) e.tensor->size(),
                               gpu_context_->streams[global_state_->current_nccl_stream][first_entry.device]);
}

void GPUAllgather::MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                              const void* buffer_data_at_offset, TensorTableEntry& e,
                                              int64_t entry_offset, size_t entry_size) {
  auto& first_entry = entries[0];
  gpu_context_->MemcpyAsyncD2D((int8_t*)e.output->data() + entry_offset, buffer_data_at_offset, entry_size,
                               gpu_context_->streams[global_state_->current_nccl_stream][first_entry.device]);
}

GPUBroadcast::GPUBroadcast(GPUContext* context,
                           HorovodGlobalState* global_state)
    : BroadcastOp(global_state), gpu_context_(context), gpu_op_context_(context, global_state) {}

bool GPUBroadcast::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}

} // namespace common
} // namespace horovod
