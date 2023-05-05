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

#include <stdexcept>

#if HAVE_CUDA
#include "cuda_runtime.h"
#include <mxnet/base.h>
#endif

#include "../common/common.h"
#include "cuda_util.h"
#include "util.h"

namespace horovod {
namespace mxnet {

with_device::with_device(int device) {
  if (device == CPU_DEVICE_ID) {
    restore_device_ = CPU_DEVICE_ID;
  } else {
#if HAVE_CUDA
    CUdevice cudev;
    auto err = cuCtxGetDevice(&cudev);
    if (err == CUDA_ERROR_NOT_INITIALIZED ||
        err == CUDA_ERROR_INVALID_CONTEXT) {
       // If device has never been set on this thread,
       // restore to supplied device.
       restore_device_ = device;
     } else if (err == CUDA_SUCCESS) {
       restore_device_ = static_cast<int>(cudev);
     } else {
       HVD_GPU_DRIVER_CHECK(err);
     }
     CUDA_CALL(cudaSetDevice(device));
#else
    throw std::logic_error("Internal error. Requested device context manager "
                           "with GPU device but not compiled with CUDA.");
#endif
  }
}

with_device::~with_device() {
#if HAVE_CUDA
  if (restore_device_ != CPU_DEVICE_ID) {
    CUDA_CALL(cudaSetDevice(restore_device_));
  }
#endif
}

} // namespace mxnet
} // namespace horovod
