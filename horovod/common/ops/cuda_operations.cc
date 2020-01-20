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

class GPUContext::impl {
public:
  cudaError_t GetGpuEvent(cudaEvent_t* event) {
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

    return cudaEventCreateWithFlags(event, cudaEventBlockingSync | cudaEventDisableTiming);
  }

  cudaError_t ReleaseGpuEvent(cudaEvent_t event) {
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

  void ErrorCheck(std::string op_name, cudaError_t cuda_result) {
    if (cuda_result != cudaSuccess) {
      throw std::logic_error(std::string(op_name) + " failed: " + cudaGetErrorString(cuda_result));
    }
  }

  void RecordEvent(std::queue<std::pair<std::string, cudaEvent_t>>& event_queue, std::string name, cudaStream_t& stream) {
    cudaEvent_t event;
    ErrorCheck("GetGpuEvent", GetGpuEvent(&event));
    ErrorCheck("cudaEventRecord", cudaEventRecord(event, stream));
    event_queue.emplace(name, event);
  }

  void WaitForEvents(std::queue<std::pair<std::string, cudaEvent_t>>& event_queue,
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
      ErrorCheck("ReleaseGpuEvent", ReleaseGpuEvent(event));
    }
  }

  void StreamCreate(cudaStream_t *stream) {
    int greatest_priority;
    ErrorCheck("cudaDeviceGetStreamPriorityRange",
        cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
    ErrorCheck("cudaStreamCreateWithPriority",
        cudaStreamCreateWithPriority(stream, cudaStreamNonBlocking, greatest_priority));
  }

  void StreamSynchronize(cudaStream_t stream) {
    ErrorCheck("cudaStreamSynchronize", cudaStreamSynchronize(stream));
  }

  int GetDevice() {
    int device;
    ErrorCheck("cudaGetDevice", cudaGetDevice(&device));
    return device;
  }

  void SetDevice(int device) {
    ErrorCheck("cudaSetDevice", cudaSetDevice(device));
  }

  void MemcpyAsyncD2D(void* dst, const void* src, size_t count, cudaStream_t stream) {
    ErrorCheck("cudaMemcpyAsync", cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream));
  }

  void MemcpyAsyncH2D(void* dst, const void* src, size_t count, cudaStream_t stream) {
    ErrorCheck("cudaMemcpyAsync", cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream));
  }

  void MemcpyAsyncD2H(void* dst, const void* src, size_t count, cudaStream_t stream) {
    ErrorCheck("cudaMemcpyAsync", cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream));
  }

private:
  // We reuse CUDA events as it appears that their creation carries non-zero cost.
  std::unordered_map<int, std::queue<cudaEvent_t>> cuda_events;
  std::mutex cuda_events_mutex;
};

#include "gpu_context_impl.cc"

} // namespace common
} // namespace horovod
