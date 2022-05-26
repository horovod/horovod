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

#include "../hashes.h"
#include "../message.h"
#include "gpu_operations.h"
#include "rocm/hip_kernels.h"

#include <thread>

namespace horovod {
namespace common {
class GPUContext::impl {
public:
  hipError_t GetGpuEvent(Event* event, hipStream_t stream) {
    int device;
    auto status = hipGetDevice(&device);
    if (status != hipSuccess) {
      return status;
    }

    auto& mutex = hip_events_mutex;
    {
      std::lock_guard<std::mutex> guard(mutex);
      auto key = std::make_pair(device, stream);
      auto& queue = hip_events[key];
      if (!prepopulated[key]) {
        // On first call for device and stream pair, prepopulate event queue.
        // This is to minimize event reuse of callback events passed to
        // framework.
        for (int i = 0; i < N_HIP_EVENTS_PREPOPULATE; ++i) {
          hipEvent_t ev;
          status = hipEventCreateWithFlags(&ev, hipEventDisableTiming);
          queue.emplace(std::make_shared<hipEvent_t>(ev), stream);
        }
        prepopulated[key] = true;
      }
      if (!queue.empty()) {
        *event = queue.front();
        event->event_idx = ++hip_event_idx[key];
        queue.pop();
        return hipSuccess;
      }
    }

    hipEvent_t ev;
    status = hipEventCreateWithFlags(&ev, hipEventDisableTiming);
    event->event = std::make_shared<hipEvent_t>(ev);
    event->stream = stream;
    auto key2 = std::make_pair(device, stream);
    event->event_idx = ++hip_event_idx[key2];

    return status;
  }

  hipError_t ReleaseGpuEvent(Event event) {
    int device;
    auto status = hipGetDevice(&device);
    if (status != hipSuccess) {
      return status;
    }

    auto& mutex = hip_events_mutex;
    {
      std::lock_guard<std::mutex> guard(mutex);
      auto& queue = hip_events[std::make_pair(device, event.stream)];
      queue.push(event);
    }

    return hipSuccess;
  }

  void ErrorCheck(std::string op_name, hipError_t hip_result) {
    if (hip_result != hipSuccess) {
      throw std::logic_error(std::string(op_name) +
                             " failed: " + hipGetErrorString(hip_result));
    }
  }

  void RecordEvent(std::queue<std::pair<std::string, Event>>& event_queue,
                   std::string name, hipStream_t& stream) {
    Event event;
    ErrorCheck("GetGpuEvent", GetGpuEvent(&event, stream));
    ErrorCheck("hipEventRecord", 
	       hipEventRecord(*(event.event), event.stream));
    event_queue.emplace(name, event);
  }

  Event RecordEvent(hipStream_t& stream) {
    Event event;
    ErrorCheck("GetGpuEvent", GetGpuEvent(&event, stream));
    ErrorCheck("hipEventRecord",
               hipEventRecord(*(event.event), event.stream));
    return event;
  }

  void WaitForEvents(std::queue<std::pair<std::string, Event>>& event_queue,
                const std::vector<TensorTableEntry>& entries,
                Timeline& timeline,
                const std::function<void()>& error_check_callback) {
    while (!event_queue.empty()) {
      std::string name;
      Event event;
      std::tie(name, event) = event_queue.front();
      event_queue.pop();
      if (name != "") {
        timeline.ActivityStartAll(entries, name);
      }

      hipError_t hip_result = hipEventSynchronize(*(event.event));
      if (hip_result != hipSuccess) {
        throw std::logic_error(std::string("cudaEventSynchronize failed: ") +
                               hipGetErrorString(hip_result));
      }
      if (error_check_callback) {
        error_check_callback();
      }

      if (name != "") {
        timeline.ActivityEndAll(entries);
      }
      ErrorCheck("ReleaseGpuEvent", ReleaseGpuEvent(event));
    }
  }

  void WaitForEventsElastic(
      std::queue<std::pair<std::string, Event>>& event_queue,
      const std::vector<TensorTableEntry>& entries, Timeline& timeline,
      const std::function<void()>& error_check_callback) {
    while (!event_queue.empty()) {
      std::string name;
      Event event;
      std::tie(name, event) = event_queue.front();
      event_queue.pop();
      if (name != "") {
        timeline.ActivityStartAll(entries, name);
      }

      // Check for async (networking) errors while waiting for the event to
      // complete
      hipError_t hip_result;
      while (true) {
        hip_result = hipEventQuery(*(event.event));
        if (hip_result == hipSuccess) {
          break;
        }

        if (hip_result != hipErrorNotReady) {
          throw std::logic_error(std::string("cudaEventQuery failed: ") +
                                 hipGetErrorString(hip_result));
        }

        if (error_check_callback) {
          error_check_callback();
        }
        std::this_thread::yield();
      }

      if (name != "") {
        timeline.ActivityEndAll(entries);
      }
      ErrorCheck("ReleaseGpuEvent", ReleaseGpuEvent(event));
    }
  }

  void ClearEvents(std::queue<std::pair<std::string, Event>>& event_queue,
                   const std::vector<TensorTableEntry>& entries,
                   Timeline& timeline,
                   const std::function<void()>& error_check_callback,
                   bool elastic) {
    while (!event_queue.empty()) {
      std::string name;
      Event event;
      std::tie(name, event) = event_queue.front();
      event_queue.pop();
      if (name != "") {
        timeline.ActivityStartAll(entries, name);
      }

      if (name != "") {
        timeline.ActivityEndAll(entries);
      }
      ErrorCheck("ReleaseGpuEvent", ReleaseGpuEvent(event));
    }
  }

  void StreamCreate(hipStream_t* stream) {
    int greatest_priority;
    ErrorCheck("hipDeviceGetStreamPriorityRange",
               hipDeviceGetStreamPriorityRange(NULL, &greatest_priority));
    ErrorCheck("hipStreamCreateWithPriority",
               hipStreamCreateWithPriority(stream, hipStreamNonBlocking,
                                           greatest_priority));
  }

  void StreamSynchronize(hipStream_t stream) {
    ErrorCheck("hipStreamSynchronize", hipStreamSynchronize(stream));
  }

  int GetDevice() {
    int device;
    ErrorCheck("hipGetDevice", hipGetDevice(&device));
    return device;
  }

  void SetDevice(int device) {
    ErrorCheck("hipSetDevice", hipSetDevice(device));
  }

  void MemcpyAsyncD2D(void* dst, const void* src, size_t count,
                      hipStream_t stream) {
    ErrorCheck(
        "hipMemcpyAsync",
        hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToDevice, stream));
  }

  void MemcpyAsyncH2D(void* dst, const void* src, size_t count,
                      hipStream_t stream) {
    ErrorCheck("hipMemcpyAsync",
               hipMemcpyAsync(dst, src, count, hipMemcpyHostToDevice, stream));
  }

  void MemcpyAsyncD2H(void* dst, const void* src, size_t count,
                      hipStream_t stream) {
    ErrorCheck("hipMemcpyAsync",
               hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToHost, stream));
  }

  void ScaleBufferImpl(const void* fused_input_data, void* buffer_data,
                       int64_t num_elements, double scale_factor,
                       DataType dtype, hipStream_t stream) {
    throw std::logic_error("ScaleBuffer not implemented for AMD GPUs.");
  }

private:
  // We reuse HIP events as it appears that their creation carries non-zero
  // cost.
  std::unordered_map<std::pair<int, hipStream_t>, std::queue<Event>>
      hip_events;
  std::unordered_map<std::pair<int, hipStream_t>, bool> prepopulated;
  std::unordered_map<std::pair<int, hipStream_t>, std::atomic<uint64_t>> hip_event_idx;
  std::mutex hip_events_mutex;
  static constexpr int N_HIP_EVENTS_PREPOPULATE = 128;
};

#include "gpu_context_impl.cc"

} // namespace common
} // namespace horovod
