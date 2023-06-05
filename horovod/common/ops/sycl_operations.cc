// Copyright (C) 2023 Intel CORPORATION. All rights reserved.
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

#include <algorithm>
#include <thread>

namespace horovod {
namespace common {

template <typename T, typename TS> class ScaleBufferSyclKernel;

class GPUContext::impl {
public:
  int GetGpuEvent(Event* event, gpuStream_t& stream) {
    auto& mutex = sycl_events_mutex;
    {
      std::lock_guard<std::mutex> guard(mutex);

      auto& queue = sycl_events[*stream];
      if (!queue.empty()) {
        *event = queue.front();
        queue.pop();
        return 0;
      }
    }

    event->event = std::make_shared<gpuEvent_t>();
    event->stream = stream;

    return 0;
  }

  gpuError_t ReleaseGpuEvent(Event event) {
    auto& mutex = sycl_events_mutex;
    {
      std::lock_guard<std::mutex> guard(mutex);
      auto& queue = sycl_events[*event.stream];
      queue.push(event);
    }
    return sycl::errc::success;
  }

  void ErrorCheck(std::string op_name, gpuError_t sycl_result) {
    throw std::logic_error("Not supported by SYCL.");
  }

  void RecordEvent(std::queue<std::pair<std::string, Event>>& event_queue,
                   std::string name, gpuStream_t& stream) {
    Event event;
    GetGpuEvent(&event, stream);
    *(event.event) = stream->ext_oneapi_submit_barrier();
    event_queue.emplace(name, event);
  }

  Event RecordEvent(gpuStream_t& stream) {
    Event event;
    GetGpuEvent(&event, stream);
    *(event.event) = stream->ext_oneapi_submit_barrier();
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
      event.event->wait();
      if (error_check_callback) {
        error_check_callback();
      }

      if (name != "") {
        timeline.ActivityEndAll(entries);
      }
      ReleaseGpuEvent(event);
    }
  }

  void
  WaitForEventsElastic(std::queue<std::pair<std::string, Event>>& event_queue,
                       const std::vector<TensorTableEntry>& entries,
                       Timeline& timeline,
                       const std::function<void()>& error_check_callback) {
    throw std::runtime_error("Not implemented yet!");
  }

  void ClearEvents(std::queue<std::pair<std::string, Event>>& event_queue,
                   const std::vector<TensorTableEntry>& entries,
                   Timeline& timeline,
                   const std::function<void()>& error_check_callback = nullptr,
                   bool elastic = false) {
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
      ReleaseGpuEvent(event);
    }
  }

  template <typename T, typename TS>
  void ScaleBufferSyclImpl(const T* input, T* output, int64_t num_elements,
                           TS scale_factor, gpuStream_t& stream) {
    const int wg_size =
        stream->get_device()
            .get_info<sycl::info::device::max_work_group_size>();
    const int num_workgroups = (num_elements + wg_size - 1) / wg_size;
    stream->submit([&](sycl::handler& h) {
      sycl::range<1> global(num_workgroups * wg_size);
      sycl::range<1> local(wg_size);
      h.parallel_for<ScaleBufferSyclKernel<T, TS>>(
          sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> it) {
            auto id = it.get_global_linear_id();
            if (id >= num_elements)
              return;

            output[id] = scale_factor * input[id];
          });
    });
  }

  void StreamCreate(const TensorTableEntry& e, gpuStream_t& stream) {
    auto org_q = e.context->SYCLQueue();
    auto property_list = sycl::property_list{sycl::property::queue::in_order()};
    stream.reset(new sycl::queue(org_q.get_context(), org_q.get_device(),
                                 property_list));
  }

  void StreamSynchronize(gpuStream_t& stream) { stream->wait(); }

  int GetDevice() { throw std::logic_error("Not supported by SYCL."); }

  void SetDevice(int device) {
    // SYCL does not support SetDevice
    return;
  }

  void MemcpyAsyncD2D(void* dst, const void* src, size_t count,
                      gpuStream_t& stream) {
    stream->memcpy(dst, src, count);
  }

  void MemcpyAsyncH2D(void* dst, const void* src, size_t count,
                      gpuStream_t stream) {
    throw std::runtime_error("Not implemented yet!");
  }

  void MemcpyAsyncD2H(void* dst, const void* src, size_t count,
                      gpuStream_t stream) {
    throw std::runtime_error("Not implemented yet!");
  }

  void ScaleBufferImpl(const void* fused_input_data, void* buffer_data,
                       int64_t num_elements, double scale_factor,
                       DataType dtype, gpuStream_t& stream) {
    auto float_scale_factor = (float)scale_factor;
    switch (dtype) {
    case HOROVOD_UINT8:
      ScaleBufferSyclImpl((const uint8_t*)fused_input_data,
                          (uint8_t*)buffer_data, num_elements,
                          float_scale_factor, stream);
      break;
    case HOROVOD_INT8:
      ScaleBufferSyclImpl((const int8_t*)fused_input_data, (int8_t*)buffer_data,
                          num_elements, float_scale_factor, stream);
      break;
    case HOROVOD_INT32:
      ScaleBufferSyclImpl((const int32_t*)fused_input_data,
                          (int32_t*)buffer_data, num_elements,
                          float_scale_factor, stream);
      break;
    case HOROVOD_INT64:
      ScaleBufferSyclImpl((const int64_t*)fused_input_data,
                          (int64_t*)buffer_data, num_elements,
                          float_scale_factor, stream);
      break;
    case HOROVOD_FLOAT16:
      ScaleBufferSyclImpl((const sycl::half*)fused_input_data,
                          (sycl::half*)buffer_data, num_elements,
                          (sycl::half)float_scale_factor, stream);
      break;
    case HOROVOD_FLOAT32:
      ScaleBufferSyclImpl((const float*)fused_input_data, (float*)buffer_data,
                          num_elements, float_scale_factor, stream);
      break;
    case HOROVOD_FLOAT64:
      ScaleBufferSyclImpl((const double*)fused_input_data, (double*)buffer_data,
                          num_elements, scale_factor, stream);
      break;
    default:
      throw std::logic_error("Type " + DataType_Name(dtype) +
                             " not supported by ScaleBufferSyclImpl.");
    }
  }

private:
  std::unordered_map<sycl::queue, std::queue<Event>> sycl_events;
  std::mutex sycl_events_mutex;
};

#include "gpu_context_impl.cc"

} // namespace common
} // namespace horovod
