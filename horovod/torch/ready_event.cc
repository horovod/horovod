// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

#if HAVE_CUDA
#include <cassert>
#include <THC/THC.h>

#include "ready_event.h"

extern THCState* state;

namespace horovod {
namespace torch {

struct ReadyEventRegistry {
  std::unordered_map<int, std::queue<cudaEvent_t>> cuda_events;
  std::mutex mutex;
};

static ReadyEventRegistry ready_event_registry;

template <class T>
TorchReadyEvent<T>::TorchReadyEvent(int device) : device_(device) {
  assert(device_ != CPU_DEVICE_ID);

  int restoreDevice;
  THCudaCheck(cudaGetDevice(&restoreDevice));
  THCudaCheck(cudaSetDevice(device_));
  {
    std::lock_guard<std::mutex> guard(ready_event_registry.mutex);
    auto& queue = ready_event_registry.cuda_events[device_];
    if (!queue.empty()) {
      cuda_event_ = queue.front();
      queue.pop();
    } else {
      THCudaCheck(cudaEventCreateWithFlags(
          &cuda_event_, cudaEventBlockingSync | cudaEventDisableTiming));
    }
  }
  auto stream = THCState_getCurrentStreamOnDevice(state, device_);
  THCudaCheck(cudaEventRecord(cuda_event_, stream));
  THCudaCheck(cudaSetDevice(restoreDevice));
}

template <class T> TorchReadyEvent<T>::~TorchReadyEvent() {
  {
    std::lock_guard<std::mutex> guard(ready_event_registry.mutex);
    auto& queue = ready_event_registry.cuda_events[device_];
    queue.push(cuda_event_);
  }
}

template <class T> bool TorchReadyEvent<T>::Ready() const {
  auto status = cudaEventQuery(cuda_event_);
  if (status == cudaErrorNotReady) {
    return false;
  }
  THCudaCheck(status);
  return true;
}

READY_EVENT_DEFINE_TYPE(THCudaByteTensor)
READY_EVENT_DEFINE_TYPE(THCudaCharTensor)
READY_EVENT_DEFINE_TYPE(THCudaShortTensor)
READY_EVENT_DEFINE_TYPE(THCudaIntTensor)
READY_EVENT_DEFINE_TYPE(THCudaLongTensor)
READY_EVENT_DEFINE_TYPE(THCudaTensor)
READY_EVENT_DEFINE_TYPE(THCudaDoubleTensor)

} // namespace torch
} // namespace horovod
#endif