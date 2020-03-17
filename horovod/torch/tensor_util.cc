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

#include "tensor_util.h"

namespace horovod {
namespace torch {

// Define all types for TensorUtil.
TENSOR_UTIL_DEFINE_CPU_TYPE(DataType::HOROVOD_UINT8, THByteTensor,
                            THByteStorage)
TENSOR_UTIL_DEFINE_CPU_TYPE(DataType::HOROVOD_INT8, THCharTensor,
                            THCharStorage)
TENSOR_UTIL_DEFINE_CPU_TYPE(DataType::HOROVOD_INT16, THShortTensor,
                            THShortStorage)
TENSOR_UTIL_DEFINE_CPU_TYPE(DataType::HOROVOD_INT32, THIntTensor,
                            THIntStorage)
TENSOR_UTIL_DEFINE_CPU_TYPE(DataType::HOROVOD_INT64, THLongTensor,
                            THLongStorage)
TENSOR_UTIL_DEFINE_CPU_TYPE(DataType::HOROVOD_FLOAT32, THFloatTensor,
                            THFloatStorage)
TENSOR_UTIL_DEFINE_CPU_TYPE(DataType::HOROVOD_FLOAT64, THDoubleTensor,
                            THDoubleStorage)

#if HAVE_GPU
TENSOR_UTIL_DEFINE_CUDA_TYPE(DataType::HOROVOD_UINT8, THCudaByteTensor,
                             THByteTensor, THCudaByteStorage)
TENSOR_UTIL_DEFINE_CUDA_TYPE(DataType::HOROVOD_INT8, THCudaCharTensor,
                             THCharTensor, THCudaCharStorage)
TENSOR_UTIL_DEFINE_CUDA_TYPE(DataType::HOROVOD_INT16, THCudaShortTensor,
                             THShortTensor, THCudaShortStorage)
TENSOR_UTIL_DEFINE_CUDA_TYPE(DataType::HOROVOD_INT32, THCudaIntTensor,
                             THIntTensor, THCudaIntStorage)
TENSOR_UTIL_DEFINE_CUDA_TYPE(DataType::HOROVOD_INT64, THCudaLongTensor,
                             THLongTensor, THCudaLongStorage)
TENSOR_UTIL_DEFINE_CUDA_TYPE(DataType::HOROVOD_FLOAT32, THCudaTensor,
                             THFloatTensor, THCudaStorage)
TENSOR_UTIL_DEFINE_CUDA_TYPE(DataType::HOROVOD_FLOAT64, THCudaDoubleTensor,
                             THDoubleTensor, THCudaDoubleStorage)
#endif

} // namespace torch
} // namespace horovod