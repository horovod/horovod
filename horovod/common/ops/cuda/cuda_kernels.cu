// Copyright (C) 2020 NVIDIA CORPORATION. All rights reserved.
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

#include "cuda_kernels.h"

#include <stdexcept>
#include <cuda_fp16.h>

namespace horovod {
namespace common {

template<typename T, typename TS>
__global__ void scale_buffer_k(const T* input, T* output, int64_t num_elements, const TS scale_factor) {

  const size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;

  for (size_t i = idx; i < num_elements; i += gridDim.x * blockDim.x) {
    output[i] = scale_factor * input[i];
  }
}

// Specialization for half2
__global__ void scale_buffer_half2_k(const __half* input, __half* output, int64_t num_elements, const __half scale_factor) {

  const size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;

#if __CUDA_ARCH__ > 530
  const __half2* input_h2 = reinterpret_cast<const __half2 *>(input);
  __half2* output_h2 = reinterpret_cast<__half2 *>(output);
  const __half2 scale_factor_h2 = __halves2half2(scale_factor, scale_factor);

  for (size_t i = idx; i < num_elements / 2; i += gridDim.x * blockDim.x) {
    output_h2[i] = __hmul2(scale_factor_h2, input_h2[i]);
  }

  // Deal with last element if num_elements is odd
  if (idx == 0 && num_elements % 2) {
    output[num_elements - 1] = __hmul(scale_factor, input[num_elements - 1]);
  }
#else
  for (size_t i = idx; i < num_elements; i += gridDim.x * blockDim.x) {
    output[i] = __float2half(__half2float(scale_factor) * __half2float(input[i]));
  }
#endif
}

// Specialization for architectures without __half compute
template<>
__global__ void scale_buffer_k(const __half* input, __half* output, int64_t num_elements, const __half scale_factor) {

  const size_t idx = static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;

#if __CUDA_ARCH__ > 530
  for (size_t i = idx; i < num_elements; i += gridDim.x * blockDim.x) {
    output[i] = scale_factor * input[i];
  }
#else
  for (size_t i = idx; i < num_elements; i += gridDim.x * blockDim.x) {
    output[i] = __float2half(__half2float(scale_factor) * __half2float(input[i]));
  }
#endif
}

#define NTHREADS_SCALE_BUFFER_KERNEL 512
void ScaleBufferCudaImpl(const void* fused_input_data, void* buffer_data, const int64_t num_elements, double scale_factor,
                         DataType dtype, cudaStream_t stream) {
  const int64_t blocks = (num_elements + NTHREADS_SCALE_BUFFER_KERNEL - 1) / NTHREADS_SCALE_BUFFER_KERNEL;
  const int threads = NTHREADS_SCALE_BUFFER_KERNEL;
  switch (dtype) {
    case HOROVOD_UINT8:
      scale_buffer_k<<<blocks, threads, 0, stream>>>((const uint8_t*) fused_input_data, (uint8_t*) buffer_data,
                                                     num_elements, scale_factor);
      break;
    case HOROVOD_INT8:
      scale_buffer_k<<<blocks, threads, 0, stream>>>((const int8_t*) fused_input_data, (int8_t*) buffer_data,
                                                     num_elements, scale_factor);
      break;
    case HOROVOD_INT32:
      scale_buffer_k<<<blocks, threads, 0, stream>>>((const int32_t*) fused_input_data, (int32_t*) buffer_data,
                                                     num_elements, scale_factor);
      break;
    case HOROVOD_INT64:
      scale_buffer_k<<<blocks, threads, 0, stream>>>((const int64_t*) fused_input_data, (int64_t*) buffer_data,
                                                     num_elements, scale_factor);
      break;
    case HOROVOD_FLOAT16:
    {
      __half scale_factor_half = __float2half((float) scale_factor);
      if ((size_t) fused_input_data % 4 == 0 && (size_t) buffer_data % 4 == 0) {
        // If alignment allows, use half2 specialized kernel
        int64_t num_elements_h2 = (num_elements + 1) / 2;
        int64_t blocks_h2 = (num_elements_h2 + NTHREADS_SCALE_BUFFER_KERNEL - 1) / NTHREADS_SCALE_BUFFER_KERNEL;
        scale_buffer_half2_k<<<blocks_h2, threads, 0, stream>>>((const __half*) fused_input_data, (__half*) buffer_data,
                                                          num_elements, scale_factor_half);
      } else {
        scale_buffer_k<<<blocks, threads, 0, stream>>>((const __half*) fused_input_data, (__half*) buffer_data,
                                                       num_elements, scale_factor_half);
     }
      break;
    }
    case HOROVOD_FLOAT32:
      scale_buffer_k<<<blocks, threads, 0, stream>>>((const float*) fused_input_data, (float*) buffer_data,
                                                     num_elements, (float) scale_factor);
      break;
    case HOROVOD_FLOAT64:
      scale_buffer_k<<<blocks, threads, 0, stream>>>((const double*) fused_input_data, (double*) buffer_data,
                                                     num_elements, scale_factor);
      break;
    default:
      throw std::logic_error("Type " + DataType_Name(dtype) +
                             " not supported by ScaleBufferCudaImpl.");
  }
}

} // namespace common
} // namespace horovod

