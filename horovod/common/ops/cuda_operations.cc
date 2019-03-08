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

#include "cuda_operations.h"

#include <thread>

namespace horovod {
namespace common {

cudaError_t CUDAContext::GetCudaEvent(cudaEvent_t* event) {
  int device;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }

  auto& mutex = cuda_events_mutex;
  {
    std::lock_guard<std::mutex> guard(mutex);
    auto& queue = cuda_events[device];
    if (!queue.empty()) {
      *event = queue.front();
      queue.pop();
      return cudaSuccess;
    }
  }

  return cudaEventCreateWithFlags(event, cudaEventBlockingSync |
                                         cudaEventDisableTiming);
}

cudaError_t CUDAContext::ReleaseCudaEvent(cudaEvent_t event) {
  int device;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }

  auto& mutex = cuda_events_mutex;
  {
    std::lock_guard<std::mutex> guard(mutex);
    auto& queue = cuda_events[device];
    queue.push(event);
  }

  return cudaSuccess;
}

void CUDAContext::ErrorCheck(std::string op_name, cudaError_t cuda_result) {
  if (cuda_result != cudaSuccess) {
    throw std::logic_error(std::string(op_name) + " failed: " + cudaGetErrorString(cuda_result));
  }
}

void CUDAContext::RecordEvent(std::queue<std::pair<std::string, cudaEvent_t>>& event_queue,
                              std::string name, cudaStream_t& stream) {
  cudaEvent_t event;
  ErrorCheck("GetCudaEvent", GetCudaEvent(&event));
  ErrorCheck("cudaEventRecord", cudaEventRecord(event, stream));
  event_queue.emplace(name, event);
}

void CUDAContext::WaitForEvents(std::queue<std::pair<std::string, cudaEvent_t>>& event_queue,
                                const std::vector<TensorTableEntry>& entries, Timeline& timeline) {
  while (!event_queue.empty()) {
    std::string name;
    cudaEvent_t event;
    std::tie(name, event) = event_queue.front();
    event_queue.pop();
    if (name != "") {
      timeline.ActivityStartAll(entries, name);
    }
    ErrorCheck("cudaEventSynchronize", cudaEventSynchronize(event));
    if (name != "") {
      timeline.ActivityEndAll(entries);
    }
    ErrorCheck("ReleaseCudaEvent", ReleaseCudaEvent(event));
  }
}

CUDAAllreduce::CUDAAllreduce(CUDAContext* context,
                             HorovodGlobalState* global_state)
    : AllreduceOp(global_state), cuda_context_(context) {}

bool CUDAAllreduce::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}

void CUDAAllreduce::MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                              const TensorTableEntry& e, void* buffer_data_at_offset) {
  auto& first_entry = entries[0];
  auto cuda_result = cudaMemcpyAsync(buffer_data_at_offset, e.tensor->data(),
                                     (size_t) e.tensor->size(), cudaMemcpyDeviceToDevice,
                                     cuda_context_->streams[first_entry.device]);
  cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
}

void CUDAAllreduce::MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                               const void* buffer_data_at_offset, TensorTableEntry& e) {
  auto& first_entry = entries[0];
  auto cuda_result = cudaMemcpyAsync((void*) e.output->data(), buffer_data_at_offset,
                                     (size_t) e.tensor->size(), cudaMemcpyDeviceToDevice,
                                     cuda_context_->streams[first_entry.device]);
  cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
}

void CUDAAllreduce::InitCUDA(const std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  cuda_context_->ErrorCheck("cudaSetDevice", cudaSetDevice(first_entry.device));

  // Ensure stream is in the map before executing reduction.
  cudaStream_t& stream = cuda_context_->streams[first_entry.device];
  if (stream == nullptr) {
    int greatest_priority;
    cuda_context_->ErrorCheck("cudaDeviceGetStreamPriorityRange",
                              cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
    cuda_context_->ErrorCheck("cudaStreamCreateWithPriority",
                              cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest_priority));
  }
}

void CUDAAllreduce::InitCUDAQueue(const std::vector<TensorTableEntry>& entries, const Response& response) {
  event_queue_ = std::queue<std::pair<std::string, cudaEvent_t>>();
  stream_ = &cuda_context_->streams[entries[0].device];
  host_buffer_ = nullptr;

  if (global_state_->timeline.Initialized()) {
    cuda_context_->RecordEvent(event_queue_, QUEUE, *stream_);
  }
}

Status CUDAAllreduce::FinalizeCUDAQueue(const std::vector<TensorTableEntry>& entries) {
  // Use completion marker via event because it's faster than
  // blocking cudaStreamSynchronize() in this thread.
  cuda_context_->RecordEvent(event_queue_, "", *stream_);

  auto& first_entry = entries[0];
  void* host_buffer = host_buffer_;
  auto& event_queue = event_queue_;
  auto& timeline = global_state_->timeline;
  auto& cuda_context = cuda_context_;

  // TODO: use thread pool or single thread for callbacks
  std::thread finalizer_thread([entries, first_entry, host_buffer,
                                event_queue, &timeline, &cuda_context]() mutable {
    auto cuda_result = cudaSetDevice(first_entry.device);
    cuda_context->ErrorCheck("cudaSetDevice", cuda_result);

    cuda_context->WaitForEvents(event_queue, entries, timeline);
    if (host_buffer != nullptr) {
      free(host_buffer);
    }

    for (auto& e : entries) {
      timeline.End(e.tensor_name, e.output);
      e.callback(Status::OK());
    }
  });

  finalizer_thread.detach();

  return Status::InProgress();
}

} // namespace common
} // namespace horovod
