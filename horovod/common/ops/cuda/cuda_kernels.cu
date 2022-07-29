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

// ATTENTION: Any change here might obsolete hip_kernels.cu in rocm folder.
//            Please keep this file synced with hip_kernels.cu.

#include "cuda_kernels.h"

#include <stdexcept>
#include <cuda_fp16.h>

namespace horovod {
namespace common {

template<typename T, int blocks_per_copy>
__device__ void batched_memcpy_d(size_t idx, const void* in, void* out, size_t size) {

  const T* input = reinterpret_cast<const T *>(in);
  T* output = reinterpret_cast<T *>(out);
  const size_t num_elements = size / sizeof(T);

  for (size_t i = idx; i < num_elements; i += blockDim.x * blocks_per_copy) {
    output[i] = input[i];
  }

  // Deal with any remaining bytes
  size_t remainder = size % sizeof(T);
  if (remainder > 0 && idx < remainder) {
    const unsigned char* input_r = reinterpret_cast<const unsigned char *>(input + num_elements);
    unsigned char* output_r = reinterpret_cast<unsigned char *>(output + num_elements);
    output_r[idx] = input_r[idx];
  }
}

template<int blocks_per_copy>
__global__ void batched_memcpy_k(BatchedD2DParams params) {
  const size_t idx = blockDim.x * (blockIdx.x % blocks_per_copy) + threadIdx.x;

  const size_t size = params.sizes[blockIdx.x / blocks_per_copy];
  const void* input = params.in[blockIdx.x / blocks_per_copy];
  void* output = params.out[blockIdx.x / blocks_per_copy];

  // Check alignment relative to 16 bytes
  size_t align_in = reinterpret_cast<size_t>(input) % BATCHED_D2D_PADDING;
  size_t align_out = reinterpret_cast<size_t>(output) % BATCHED_D2D_PADDING;

  // Select load/store size based on the misaligned buffer
  size_t align = (align_out == 0) ? align_in : align_out;
  if (align_in && align_out) {
    // If both are misaligned, use unsigned char (this should not occur for
    // Allreduces as fusion buffer locations should be aligned by applying
    // BATCHED_D2D_PADDING during construction.)
    align = 1;
  }

  if (align % 16 == 0) {
    batched_memcpy_d<ulonglong2, blocks_per_copy>(idx, input, output, size);
  } else if (align % 8 == 0) {
    batched_memcpy_d<unsigned long long, blocks_per_copy>(idx, input, output, size);
  } else if (align % 4 == 0) {
    batched_memcpy_d<unsigned int, blocks_per_copy>(idx, input, output, size);
  } else if (align % 2 == 0) {
    batched_memcpy_d<unsigned short, blocks_per_copy>(idx, input, output, size);
  } else {
    batched_memcpy_d<unsigned char, blocks_per_copy>(idx, input, output, size);
  }
}

#define NTHREADS_D2D_KERNEL 1024
#define BLOCKS_PER_COPY_D2D_KERNEL 8
void BatchedD2DMemcpyCudaImpl(BatchedD2DParams& params, int num_copies, cudaStream_t stream)
{
   batched_memcpy_k<BLOCKS_PER_COPY_D2D_KERNEL><<<num_copies * BLOCKS_PER_COPY_D2D_KERNEL,
                                                  NTHREADS_D2D_KERNEL, 0, stream>>>(params);
}

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

template<typename TL, int blocks_per_copy, typename T, typename TS>
__device__ void batched_scaled_memcpy_d(size_t idx, const T* input, T* output, size_t size, const TS scale_factor) {

  const int64_t num_words = size / sizeof(TL);
  const TL* read_ptr = reinterpret_cast<const TL*>(input);
  TL* write_ptr = reinterpret_cast<TL*>(output);
  for (size_t i = idx; i < num_words; i += blockDim.x * blocks_per_copy) {
    // Load word
    TL word = read_ptr[i];
    T* val = reinterpret_cast<T*>(&word);

    // Scale elements in word
    for (int j = 0; j < sizeof(TL) / sizeof(T); ++j) {
      val[j] *= scale_factor;
    }

    // Write word
    write_ptr[i] = word;
  }

  // Deal with any remaining elements
  size_t remainder = (size % sizeof(TL)) / sizeof(T);
  if (remainder > 0 && idx < remainder) {
    const T* input_r = reinterpret_cast<const T*>(read_ptr + num_words);
    T* output_r = reinterpret_cast<T*>(write_ptr + num_words);
    output_r[idx] = scale_factor * input_r[idx];
  }
}

// Specialization for architectures without __half compute
template<typename TL, int blocks_per_copy>
__device__ void batched_scaled_memcpy_d(size_t idx, const __half* input, __half* output, size_t size, const __half scale_factor) {

  const int64_t num_words = size / sizeof(TL);
  const TL* read_ptr = reinterpret_cast<const TL*>(input);
  TL* write_ptr = reinterpret_cast<TL*>(output);
  for (size_t i = idx; i < num_words; i += blockDim.x * blocks_per_copy) {
    // Load word
    TL word = read_ptr[i];
    __half* val = reinterpret_cast<__half*>(&word);

    // Scale elements in word
    for (int j = 0; j < sizeof(TL) / sizeof(__half); ++j) {
#if __CUDA_ARCH__ > 530
      val[j] *= scale_factor;
#else
      val[j] = __float2half(__half2float(scale_factor) * __half2float(val[j]));
#endif
    }

    // Write word
    write_ptr[i] = word;
  }

  // Deal with any remaining elements
  size_t remainder = (size % sizeof(TL)) / sizeof(__half);
  if (remainder > 0 && idx < remainder) {
    const __half* input_r = reinterpret_cast<const __half*>(read_ptr + num_words);
    __half* output_r = reinterpret_cast<__half*>(write_ptr + num_words);
#if __CUDA_ARCH__ > 530
    output_r[idx] = scale_factor * input_r[idx];
#else
    output_r[idx] = __float2half(__half2float(scale_factor) * __half2float(input_r[idx]));
#endif
  }
}

template<typename T, int blocks_per_copy, typename TS>
__global__ void batched_scaled_memcpy_k(BatchedD2DParams params, TS scale_factor) {
  const size_t idx = blockDim.x * (blockIdx.x % blocks_per_copy) + threadIdx.x;

  const size_t size = params.sizes[blockIdx.x / blocks_per_copy];
  const T* input = reinterpret_cast<const T*>(params.in[blockIdx.x / blocks_per_copy]);
  T* output = reinterpret_cast<T*>(params.out[blockIdx.x / blocks_per_copy]);

  // Check alignment relative to 16 bytes
  size_t align_in = reinterpret_cast<size_t>(input) % BATCHED_D2D_PADDING;
  size_t align_out = reinterpret_cast<size_t>(output) % BATCHED_D2D_PADDING;

  // Select load/store size based on the misaligned buffer
  size_t align = (align_out == 0) ? align_in : align_out;
  if (align_in && align_out) {

    // If both are misaligned, use datatype size
    align = sizeof(T);
  }

  if (align % 16 == 0) {
    batched_scaled_memcpy_d<ulonglong2, blocks_per_copy>(idx, input, output, size, scale_factor);
  } else if (align % 8 == 0) {
    batched_scaled_memcpy_d<unsigned long long, blocks_per_copy>(idx, input, output, size, scale_factor);
  } else if (align % 4 == 0) {
    batched_scaled_memcpy_d<unsigned int, blocks_per_copy>(idx, input, output, size, scale_factor);
  } else if (align % 2 == 0) {
    batched_scaled_memcpy_d<unsigned short, blocks_per_copy>(idx, input, output, size, scale_factor);
  } else {
    batched_scaled_memcpy_d<unsigned char, blocks_per_copy>(idx, input, output, size, scale_factor);
  }
}

void BatchedScaledD2DMemcpyCudaImpl(BatchedD2DParams& params, int num_copies, double scale_factor,
                                    DataType dtype, cudaStream_t stream) {
  const int64_t blocks = num_copies * BLOCKS_PER_COPY_D2D_KERNEL;
  const int threads = NTHREADS_D2D_KERNEL;
  switch (dtype) {
   case HOROVOD_UINT8:
     batched_scaled_memcpy_k<uint8_t, BLOCKS_PER_COPY_D2D_KERNEL><<<blocks, threads, 0, stream>>>(params, scale_factor);
     break;
   case HOROVOD_INT8:
     batched_scaled_memcpy_k<int8_t, BLOCKS_PER_COPY_D2D_KERNEL><<<blocks, threads, 0, stream>>>(params, scale_factor);
     break;
   case HOROVOD_INT32:
     batched_scaled_memcpy_k<int32_t, BLOCKS_PER_COPY_D2D_KERNEL><<<blocks, threads, 0, stream>>>(params, scale_factor);
     break;
   case HOROVOD_INT64:
     batched_scaled_memcpy_k<int64_t, BLOCKS_PER_COPY_D2D_KERNEL><<<blocks, threads, 0, stream>>>(params, scale_factor);
     break;
   case HOROVOD_FLOAT16: {
     __half scale_factor_half = __float2half((float) scale_factor);
     batched_scaled_memcpy_k<__half, BLOCKS_PER_COPY_D2D_KERNEL><<<blocks, threads, 0, stream>>>(params, scale_factor_half);
     break;
   }
   case HOROVOD_FLOAT32:
     batched_scaled_memcpy_k<float, BLOCKS_PER_COPY_D2D_KERNEL><<<blocks, threads, 0, stream>>>(params, (float) scale_factor);
     break;
   case HOROVOD_FLOAT64:
     batched_scaled_memcpy_k<double, BLOCKS_PER_COPY_D2D_KERNEL><<<blocks, threads, 0, stream>>>(params, scale_factor);
     break;
   default:
     throw std::logic_error("Type " + DataType_Name(dtype) +
                            " not supported by BatchedScaledD2DMemcpyCudaImpl.");
  }
}

} // namespace common
} // namespace horovod

