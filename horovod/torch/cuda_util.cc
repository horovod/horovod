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

#if HAVE_GPU
#include "cuda_runtime.h"
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#else
#include <stdexcept>
#endif

#include "../common/common.h"
#include "cuda_util.h"

namespace horovod {
namespace torch {

with_device::with_device(int device) {
  if (device == CPU_DEVICE_ID) {
    restore_device_ = CPU_DEVICE_ID;
  } else {
#if HAVE_GPU
    C10_CUDA_CHECK(cudaGetDevice(&restore_device_));
    C10_CUDA_CHECK(cudaSetDevice(device));
#else
    throw std::logic_error("Internal error. Requested device context manager "
                           "with GPU device but not compiled with CUDA.");
#endif
  }
}

with_device::~with_device() {
#if HAVE_GPU
  if (restore_device_ != CPU_DEVICE_ID) {
    C10_CUDA_CHECK(cudaSetDevice(restore_device_));
  }
#endif
}

} // namespace torch
} // namespace horovod
