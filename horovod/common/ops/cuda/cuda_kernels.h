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

#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <stdint.h>
#include <cuda_runtime.h>

#include "../../message.h"

#define BATCHED_D2D_CAPACITY 160
#define BATCHED_D2D_PADDING 16

namespace horovod {
namespace common {

struct BatchedD2DParams {
  void* out[BATCHED_D2D_CAPACITY];
  void* in[BATCHED_D2D_CAPACITY];
  size_t sizes[BATCHED_D2D_CAPACITY];
};

// Performs a batched d2d memcopy
void BatchedD2DMemcpyCudaImpl(BatchedD2DParams& params, int num_copies, cudaStream_t stream);

// Scales buffer by scalar
void ScaleBufferCudaImpl(const void* fused_input_data, void* buffer_data, const int64_t num_elements,
                         double scale_factor, DataType dtype, cudaStream_t stream);

void BatchedScaledD2DMemcpyCudaImpl(BatchedD2DParams& params, int num_copies, double scale_factor,
                                    DataType dtype, cudaStream_t stream);

// Performs adasum operator logic for double
void CudaSingleAdasumImpl(int count, double* device_a, const double* device_b,
						double* device_vals, cudaStream_t stream);

// Performs adasum operator logic for float
void CudaSingleAdasumImpl(int count, float* device_a, const float* device_b,
						double* device_vals, cudaStream_t stream);

// Performs adasum operator logic for uint16_t
void CudaSingleAdasumImpl(int count, uint16_t* device_a, const uint16_t* device_b,
						double* device_vals, cudaStream_t stream);

// Performs a fused dot product kernel for double
void CudaDotProductImpl(int count, const double* device_a, const double* device_b,
						double* device_vals, double& host_normsq_a, double& host_normsq_b, double& host_dot);

// Performs a fused dot product kernel for float
void CudaDotProductImpl(int count, const float* device_a, const float* device_b,
						double* device_vals, double& host_normsq_a, double& host_normsq_b, double& host_dot);

// Performs a fused dot product kernel for uint16_t
void CudaDotProductImpl(int count, const uint16_t* device_a, const uint16_t* device_b,
						double* device_vals, double& host_normsq_a, double& host_normsq_b, double& host_dot);

// Performs a fused scale and add kernel for double
void CudaScaleAddImpl(int count, double* a_device, const double* b_device, double host_a_coeff, double host_b_coeff);

// Performs a fused scale and add kernel for float
void CudaScaleAddImpl(int count, float* a_device, const float* b_device, double host_a_coeff, double host_b_coeff);

// Performs a fused scale and add kernel for uint16_t
void CudaScaleAddImpl(int count, uint16_t* a_device, const uint16_t* b_device, double host_a_coeff, double host_b_coeff);

} // namespace common
} // namespace horovod

#endif // CUDA_KERNELS_H
