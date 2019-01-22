// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
namespace mxnet {

template <class T>
MXReadyEvent<T>::MXReadyEvent(NDArray* tensor) : tensor_(tensor) {
  assert(tensor->ctx().real_dev_id() != CPU_DEVICE_ID);
}

template <class T> MXReadyEvent<T>::~MXReadyEvent() {}

template <class T> bool MXReadyEvent<T>::Ready() const { return true; }

template class MXReadyEvent<NDArray>;

} // namespace mxnet
} // namespace horovod
#endif
