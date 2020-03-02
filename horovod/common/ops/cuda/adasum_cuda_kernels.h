// Copyright 2019 Microsoft. All Rights Reserved.
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

#include <stdint.h>
#include <cuda_runtime.h>
void CudaSingleAdasumImpl(int count, double* device_a, const double* device_b, 
						double* device_vals, cudaStream_t stream);

void CudaSingleAdasumImpl(int count, float* device_a, const float* device_b, 
						double* device_vals, cudaStream_t stream);

void CudaSingleAdasumImpl(int count, uint16_t* device_a, const uint16_t* device_b, 
						double* device_vals, cudaStream_t stream);

void CudaDotProductImpl(int count, const double* device_a, const double* device_b, 
						double* device_vals, double& host_normsq_a, double& host_normsq_b, double& host_dot);

void CudaDotProductImpl(int count, const float* device_a, const float* device_b, 
						double* device_vals, double& host_normsq_a, double& host_normsq_b, double& host_dot);

void CudaDotProductImpl(int count, const uint16_t* device_a, const uint16_t* device_b, 
						double* device_vals, double& host_normsq_a, double& host_normsq_b, double& host_dot);

void CudaScaleAddImpl(int count, double* a_device, const double* b_device, double host_a_coeff, double host_b_coeff);

void CudaScaleAddImpl(int count, float* a_device, const float* b_device, double host_a_coeff, double host_b_coeff);

void CudaScaleAddImpl(int count, uint16_t* a_device, const uint16_t* b_device, double host_a_coeff, double host_b_coeff);