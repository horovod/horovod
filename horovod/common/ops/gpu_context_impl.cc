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
#ifdef HAVE_CUDA
#include "cuda/cuda_kernels.h"
#elif HAVE_ROCM
#include "hip/hip_kernels.h"
#endif
#include "../message.h"
#include "../hashes.h"

#include <thread>

namespace horovod {
namespace common {
class GPUContext::impl {
public:
  gpuError_t GetGpuEvent(Event* event, gpuStream_t stream) {
    int device;
#ifdef HAVE_CUDA
    auto status = cudaGetDevice(&device);
#elif HAVE_RCOM
    auto status = hipGetDevice(&device);
#endif
    if (status != gpuSuccess) {
      return status;
    }

    auto& mutex = gpu_events_mutex;
    {
      std::lock_guard<std::mutex> guard(mutex);
      auto key = std::make_pair(device, stream);
      auto& queue = cuda_events[key];
      if (!prepopulated[key]) {
        // On first call for device and stream pair, prepopulate event queue.
        // This is to minimize event reuse of callback events passed to framework.
        for (int i = 0; i < N_CUDA_EVENTS_PREPOPULATE; ++i) {
          gpuEvent_t ev;
#ifdef HAVE_CUDA
          status = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
#elif HAVE_ROCM
          status = hipEventCreateWithFlags(&ev, hipEventDisableTiming);
#endif
          queue.emplace(std::make_shared<gpuEvent_t>(ev), stream);
        }
        prepopulated[key] = true;
      }
      if (!queue.empty()) {
        *event = queue.front();
        queue.pop();
        return gpuSuccess;
      }
    }

    gpuEvent_t ev;
#ifdef HAVE_CUDA
    status = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
#elif HAVE_RCOM
    status = hipEventCreateWithFlags(&ev, hipEventDisableTiming);
#endif
    event->event = std::make_shared<gpuEvent_t>(ev);
    event->stream = stream;

    return status;
  }

  gpuError_t ReleaseGpuEvent(Event event) {
    int device;
#if HAVE_CUDA
    auto status = cudaGetDevice(&device);
#elif HAVE_ROCM
    auto status = hipGetDevice(&device);
#endif
    if (status != gpuSuccess) {
      return status;
    }

    auto& mutex = gpu_events_mutex;
    {
      std::lock_guard<std::mutex> guard(mutex);
      auto& queue = gpu_events[std::make_pair(device, event.stream)];
      queue.push(event);
    }

    return gpuSuccess;
  }

  void ErrorCheck(std::string op_name, gpuError_t gpu_result) {
    if (gpu_result != gpuSuccess) {
#ifdef HAVE_CUDA
      throw std::logic_error(std::string(op_name) + " failed: " + cudaGetErrorString(gpu_result));
#elif HAVE_RCOM
      throw std::logic_error(std::string(op_name) + " failed: " + hipGetErrorString(gpu_result));
#endif
    }
  }

  void RecordEvent(std::queue<std::pair<std::string, Event>>& event_queue, std::string name, gpuStream_t& stream) {
    Event event;
    ErrorCheck("GetGpuEvent", GetGpuEvent(&event, stream));
#ifdef HAVE_CUDA
    ErrorCheck("cudaEventRecord", cudaEventRecord(*(event.event), event.stream));
#elif HAVE_ROCM
    ErrorCheck("cudaEventRecord", hipEventRecord(*(event.event), event.stream));
#endif
    event_queue.emplace(name, event);
  }

  Event RecordEvent(gpuStream_t& stream) {
    Event event;
    ErrorCheck("GetGpuEvent", GetGpuEvent(&event, stream));
#ifdef HAVE_CUDA
    ErrorCheck("cudaEventRecord", cudaEventRecord(*(event.event), event.stream));
#elif HAVE_RCOM
    ErrorCheck("cudaEventRecord", hipEventRecord(*(event.event), event.stream));
#endif
    return event;
  }

  void WaitForEvents(std::queue<std::pair<std::string, Event>>& event_queue,
      const std::vector<TensorTableEntry>& entries, Timeline& timeline,
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

      // Check for async (networking) errors while waiting for the event to complete
      if (elastic) {
        gpuError_t gpu_result;
        while (true) {
#ifdef HAVE_CUDA
          gpu_result = cudaEventQuery(*(event.event));
#elif HAVE_ROCM
          gpu_result = hipEventQuery(*(event.event));
#endif
          if (gpu_result == gpuSuccess) {
            break;
          }

#ifdef HAVE_CUDA
          if (gpu_result != cudaErrorNotReady) {
            throw std::logic_error(std::string("gpuEventQuery failed: ") + cudaGetErrorString(gpu_result));
#elif HAVE_ROCM
          if (gpu_result != hipErrorNotReady) {
            throw std::logic_error(std::string("gpuEventQuery failed: ") + hipGetErrorString(gpu_result));
#endif
          }

          if (error_check_callback) {
            error_check_callback();
          }
          std::this_thread::yield();
        }
      } else {
        gpuError_t gpu_result;
#ifdef HAVE_CUDA
        gpu_result= cudaEventSynchronize(*(event.event));
#elif HAVE_RCOM
        gpu_result= hipEventSynchronize(*(event.event));
#endif
        if (gpu_result != gpuSuccess){ 
#ifdef HAVE_CUDA
          throw std::logic_error(std::string("gpuEventSynchronize failed: ") + cudaGetErrorString(gpu_result));
#elif HAVE_RCOM
          throw std::logic_error(std::string("gpuEventSynchronize failed: ") + hipGetErrorString(gpu_result));
#endif 
        }
        if (error_check_callback){ 
          error_check_callback();
        }
        
      }

      if (name != "") {
        timeline.ActivityEndAll(entries);
      }
      ErrorCheck("ReleaseGpuEvent", ReleaseGpuEvent(event));
    }
  }

  void ClearEvents(std::queue<std::pair<std::string, Event>>& event_queue,
      const std::vector<TensorTableEntry>& entries, Timeline& timeline,
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

  void StreamCreate(gpuStream_t *stream) {
    int greatest_priority;
#ifdef HAVE_CUDA
    ErrorCheck("GpuDeviceGetStreamPriorityRange",
        cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
    ErrorCheck("GpuStreamCreateWithPriority",
        cudaStreamCreateWithPriority(stream, cudaStreamNonBlocking, greatest_priority));
#elif HAVE_CROM
    ErrorCheck("GpuDeviceGetStreamPriorityRange",
        hipDeviceGetStreamPriorityRange(NULL, &greatest_priority));
    ErrorCheck("GpuStreamCreateWithPriority",
        hipStreamCreateWithPriority(stream, cudaStreamNonBlocking, greatest_priority));
#endif
  }

  void StreamSynchronize(hipStream_t stream) {
#ifdef HAVE_CUDA
    ErrorCheck("gpuStreamSynchronize", cudaStreamSynchronize(stream));
#elif HAVE_ROCM
    ErrorCheck("gpuStreamSynchronize", hipStreamSynchronize(stream));
#endif
  }

  int GetDevice() {
    int device;
#ifdef HAVE_CUDA
    ErrorCheck("gpuGetDevice", cudaGetDevice(&device));
#elif HAVE_RCOM
    ErrorCheck("gpuGetDevice", hipGetDevice(&device));
#endif
    return device;
  }

  void SetDevice(int device) {
#ifdef HAVE_CUDA
    ErrorCheck("gpuSetDevice", cudaSetDevice(device));
#elif HAVE_RCOM
    ErrorCheck("gpuSetDevice", hipSetDevice(device));
#endif
  }

  void MemcpyAsyncD2D(void* dst, const void* src, size_t count, hipStream_t stream) {
#ifdef HAVE_CUDA
    ErrorCheck("gpuMemcpyAsync", cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream));
#elif HAVE_ROCM
    ErrorCheck("gpuMemcpyAsync", hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToDevice, stream));
#endif
  }

  void MemcpyAsyncH2D(void* dst, const void* src, size_t count, hipStream_t stream) {
#ifdef HAVE_CUDA
    ErrorCheck("gpuMemcpyAsync", cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream));
#elif HAVE_RCOM
    ErrorCheck("gpuMemcpyAsync", hipMemcpyAsync(dst, src, count, hipMemcpyHostToDevice, stream));
#endif
  }

  void MemcpyAsyncD2H(void* dst, const void* src, size_t count, cudaStream_t stream) {
#ifdef HAVE_CUDA
    ErrorCheck("gpuMemcpyAsync", cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream));
#elif HAVE_RCOM
    ErrorCheck("gpuMemcpyAsync", hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToHost, stream));
#endif
  }

  void ScaleBufferImpl(const void* fused_input_data, void* buffer_data, int64_t num_elements,
                       double scale_factor, DataType dtype, gpuStream_t stream) {
    ScaleBufferGPUImpl(fused_input_data, buffer_data, num_elements, scale_factor, dtype, stream);

    // TODO: https://github.com/horovod/horovod/issues/2230
    //ErrorCheck("ScaleBufferCudaImpl", cudaGetLastError());
  }

private:
  // We reuse CUDA events as it appears that their creation carries non-zero cost.
  std::unordered_map<std::pair<int, gpuStream_t>, std::queue<Event>> gpu_events;
  std::unordered_map<std::pair<int, gpuStream_t>, bool> prepopulated;
  std::mutex gpu_events_mutex;

  static constexpr int N_GPU_EVENTS_PREPOPULATE = 128;
};

#include "gpu_context.cc"

} // namespace common
} // namespace horovod
