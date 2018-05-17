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
TENSOR_UTIL_DEFINE_TYPE(THByteTensor, THByteStorage, MPIDataType::HOROVOD_UINT8)
TENSOR_UTIL_DEFINE_TYPE(THCharTensor, THCharStorage, MPIDataType::HOROVOD_INT8)
TENSOR_UTIL_DEFINE_TYPE(THShortTensor, THShortStorage,
                        MPIDataType::HOROVOD_INT16)
TENSOR_UTIL_DEFINE_TYPE(THIntTensor, THIntStorage, MPIDataType::HOROVOD_INT32)
TENSOR_UTIL_DEFINE_TYPE(THLongTensor, THLongStorage, MPIDataType::HOROVOD_INT64)
TENSOR_UTIL_DEFINE_TYPE(THFloatTensor, THFloatStorage,
                        MPIDataType::HOROVOD_FLOAT32)
TENSOR_UTIL_DEFINE_TYPE(THDoubleTensor, THDoubleStorage,
                        MPIDataType::HOROVOD_FLOAT64)

#if HAVE_CUDA
TENSOR_UTIL_DEFINE_CUDA_TYPE(THCudaByteTensor, THByteTensor, THCudaByteStorage,
                             MPIDataType::HOROVOD_UINT8)
TENSOR_UTIL_DEFINE_CUDA_TYPE(THCudaCharTensor, THCharTensor, THCudaCharStorage,
                             MPIDataType::HOROVOD_INT8)
TENSOR_UTIL_DEFINE_CUDA_TYPE(THCudaShortTensor, THShortTensor,
                             THCudaShortStorage, MPIDataType::HOROVOD_INT16)
TENSOR_UTIL_DEFINE_CUDA_TYPE(THCudaIntTensor, THIntTensor, THCudaIntStorage,
                             MPIDataType::HOROVOD_INT32)
TENSOR_UTIL_DEFINE_CUDA_TYPE(THCudaLongTensor, THLongTensor, THCudaLongStorage,
                             MPIDataType::HOROVOD_INT64)
TENSOR_UTIL_DEFINE_CUDA_TYPE(THCudaTensor, THFloatTensor, THCudaStorage,
                             MPIDataType::HOROVOD_FLOAT32)
TENSOR_UTIL_DEFINE_CUDA_TYPE(THCudaDoubleTensor, THDoubleTensor,
                             THCudaDoubleStorage, MPIDataType::HOROVOD_FLOAT64)
#endif

} // namespace torch
} // namespace horovod