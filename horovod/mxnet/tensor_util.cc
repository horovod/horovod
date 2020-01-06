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

#include <mxnet/c_api.h>

#include "tensor_util.h"

namespace horovod {
namespace mxnet {

// Define all types for TensorUtil.
const DataType TensorUtil::GetDType(NDArray* tensor) {
  switch (tensor->dtype()) {
  case mshadow::kFloat32:
    return DataType::HOROVOD_FLOAT32;
  case mshadow::kFloat64:
    return DataType::HOROVOD_FLOAT64;
  case mshadow::kFloat16:
    return DataType::HOROVOD_FLOAT16;
  case mshadow::kUint8:
    return DataType::HOROVOD_UINT8;
  case mshadow::kInt32:
    return DataType::HOROVOD_INT32;
  case mshadow::kInt8:
    return DataType::HOROVOD_INT8;
  case mshadow::kInt64:
    return DataType::HOROVOD_INT64;
  default:
    throw std::logic_error("GetDType: Type " + std::to_string(tensor->dtype()) +
                           " is not supported in MPI mode.");
  }
}

// Return shape of tensor (similar to TShape)
const TensorShape TensorUtil::GetShape(NDArray* tensor) {
  TensorShape shape;
  TShape mx_shape = tensor->shape();
  for (int idx = 0; idx < (int)mx_shape.ndim(); idx++) {
    shape.AddDim(mx_shape[idx]);
  }
  return shape;
}

// Return data of tensor
const void* TensorUtil::GetData(NDArray* tensor) {
  // The following returns an error:
  // return tensor->data().dptr<void>();
  switch (tensor->dtype()) {
  case mshadow::kFloat32:
    return static_cast<void*>(tensor->data().dptr<float>());
  case mshadow::kFloat64:
    return static_cast<void*>(tensor->data().dptr<double>());
  case mshadow::kFloat16:
    return static_cast<void*>(tensor->data().dptr<mshadow::half::half_t>());
  case mshadow::kUint8:
    return static_cast<void*>(tensor->data().dptr<uint8_t>());
  case mshadow::kInt32:
    return static_cast<void*>(tensor->data().dptr<int32_t>());
  case mshadow::kInt8:
    return static_cast<void*>(tensor->data().dptr<int8_t>());
  case mshadow::kInt64:
    return static_cast<void*>(tensor->data().dptr<int64_t>());
  default:
    throw std::logic_error("Type " + std::to_string(tensor->dtype()) +
                           " is not supported in MPI mode.");
  }
}

// Return size of tensor in bytes
int64_t TensorUtil::GetSize(NDArray* tensor) {
  int64_t element_size = 0;
  switch (tensor->dtype()) {
  case mshadow::kFloat32:
    element_size = kFloat32Size;
    break;
  case mshadow::kFloat64:
    element_size = kFloat64Size;
    break;
  case mshadow::kFloat16:
    element_size = kFloat16Size;
    break;
  case mshadow::kUint8:
    element_size = kUInt8Size;
    break;
  case mshadow::kInt32:
    element_size = kInt32Size;
    break;
  case mshadow::kInt8:
    element_size = kInt8Size;
    break;
  case mshadow::kInt64:
    element_size = kInt64Size;
    break;
  default:
    throw std::logic_error("Type " + std::to_string(tensor->dtype()) +
                           " is not supported in MPI mode.");
  }
  return (int64_t)(tensor->shape().Size()) * element_size;
}

// If Tensor on GPU, return device id
// Otherwise return CPU_DEVICE_ID (-1)
int TensorUtil::GetDevice(NDArray* tensor) {
  int dev_mask = tensor->ctx().dev_mask();
  if (dev_mask == gpu::kDevMask)
    return tensor->ctx().real_dev_id();
  return CPU_DEVICE_ID;
}

// Resize tensor to ndim with length dims[i] in dimension i
void TensorUtil::ResizeNd(NDArray *tensor, int ndim, int64_t* dims) {
  TShape shape(dims, dims + ndim);
  tensor->ReshapeAndAlloc(shape);
}

// Copy from tensor to output
void TensorUtil::Copy(NDArray* output, NDArray* tensor) {
  if (tensor->shape() != output->shape())
    output->ReshapeAndAlloc(tensor->shape());
  CopyFromTo(*tensor, output, 0);
}

// Elementwise division of tensor by value in-place
void TensorUtil::DivideTensorInPlace(NDArray* tensor, int value) {
  *tensor /= value;
}

#if HAVE_CUDA
void TensorUtil::AsyncCopyCPUToCuda(NDArray* cpu, NDArray* cuda) {
  TensorUtil::Copy(cuda, cpu);
}

void TensorUtil::AsyncCopyCudaToCPU(NDArray* cuda, NDArray* cpu) {
  TensorUtil::Copy(cpu, cuda);
}
#endif

} // namespace mxnet
} // namespace horovod
