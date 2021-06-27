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

} // namespace common
} // namespace horovod

#endif // CUDA_KERNELS_H
