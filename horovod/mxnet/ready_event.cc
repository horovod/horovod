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

#include <mxnet/base.h>

#if HAVE_CUDA
#include <cassert>

#include "ready_event.h"

namespace horovod {
namespace MX {

/*struct ReadyEventRegistry {
  std::unordered_map<int, std::queue<cudaEvent_t>> cuda_events;
  std::mutex mutex;
};

static ReadyEventRegistry ready_event_registry;*/

template <class T>
MXReadyEvent<T>::MXReadyEvent(NDArray* tensor) : tensor_(tensor) {
  int device = tensor->ctx().real_dev_id();
  assert(device != CPU_DEVICE_ID);

  /*int restoreDevice;
  CUDA_CALL(cudaGetDevice(&restoreDevice));
  CUDA_CALL(cudaSetDevice(device_));
  {
    std::lock_guard<std::mutex> guard(ready_event_registry.mutex);
    auto& queue = ready_event_registry.cuda_events[device_];
    if (!queue.empty()) {
      cuda_event_ = queue.front();
      queue.pop();
    } else {
      CUDA_CALL(cudaEventCreateWithFlags(
          &cuda_event_, cudaEventBlockingSync | cudaEventDisableTiming));
    }
  }
  
  auto stream = MXGetCurrentStreamOnDevice(ctx_);
  CUDA_CALL(cudaEventRecord(cuda_event_, stream));
  CUDA_CALL(cudaSetDevice(restoreDevice));*/
}

template <class T> MXReadyEvent<T>::~MXReadyEvent() {
  /*{
    std::lock_guard<std::mutex> guard(ready_event_registry.mutex);
    auto& queue = ready_event_registry.cuda_events[device_];
    queue.push(cuda_event_);
  }*/
}

template <class T> bool MXReadyEvent<T>::Ready() const {
  /*auto status = cudaEventQuery(cuda_event_);
  if (status == cudaErrorNotReady) {
    return false;
  }
  CUDA_CALL(status);*/
  //tensor_->WaitToRead();
  return true;
}

template class MXReadyEvent<NDArray>;

} // namespace MX
} // namespace horovod
#endif
