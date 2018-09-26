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
namespace MX {

// Define all types for TensorUtil.
const MPIDataType TensorUtil::GetDType(NDArray* tensor) {
  switch (tensor->dtype()) {
    case 0:
      return MPIDataType::HOROVOD_FLOAT32;
    case 1:
      return MPIDataType::HOROVOD_FLOAT64;
    // TODO(@ctcyang): restore after fp16 grad branch is ready
    //case 2:
    //  return MPIDataType::HOROVOD_FLOAT16;
    case 3:
      return MPIDataType::HOROVOD_UINT8;
    case 4:
      return MPIDataType::HOROVOD_INT32;
    case 5:
      return MPIDataType::HOROVOD_INT8;
    case 6:
      return MPIDataType::HOROVOD_INT64;
    default:
      throw std::logic_error("GetDType: Type " + std::to_string(tensor->dtype()) +
                             " is not supported in MPI mode.");
  }
}

// Return shape of tensor (similar to TShape)
const TensorShape TensorUtil::GetShape(NDArray* tensor) {
  TensorShape shape;
  TShape mx_shape = tensor->shape();
  for (unsigned idx = 0; idx < mx_shape.ndim(); idx++) {
    shape.AddDim(mx_shape[idx]);
  }
  return shape;
}

// Return data of tensor
const void* TensorUtil::GetData(NDArray* tensor) {
  // The following returns an error:
  // return tensor->data().dptr<void>();
  switch (tensor->dtype()) {
    case 0:
      return static_cast<void*>(tensor->data().dptr<float>());
    case 1:
      return static_cast<void*>(tensor->data().dptr<double>());
    // TODO(@ctcyang): for fp16 support when branch is merged
    //case 2:
    //  return static_cast<void*>(tensor->data().dptr<mshadow::half::half_t>());
    case 3:
      return static_cast<void*>(tensor->data().dptr<uint8_t>());
    case 4:
      return static_cast<void*>(tensor->data().dptr<int32_t>());
    case 5:
      return static_cast<void*>(tensor->data().dptr<int8_t>());
    case 6:
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
    case 0:
      element_size = kFloat32Size;
      break;
    case 1:
      element_size = kFloat64Size;
      break;
    // TODO(@ctcyang): for fp16 support when that branch is merged
    //case 2:
    //  element_size = kFloat16Size;
    //  break;
    case 3:
      element_size = kUInt8Size;
      break;
    case 4:
      element_size = kInt32Size;
      break;
    case 5:
      element_size = kInt8Size;
      break;
    case 6:
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

// Returns pointer to newly created NDArray
// If dev_id equal to CPU_DEVICE_ID, construct Tensor on CPU
// Otherwise construct on GPU
NDArray* TensorUtil::New(int device, int dtype) {
  if (device == CPU_DEVICE_ID) {
    NDArray* my_array = new NDArray(TShape(), Context::CPU(0), false, dtype);
    return my_array;
  } else {
    NDArray* my_array = new NDArray(TShape(), Context::GPU(device), false, dtype);
    return my_array;
  }
}

void TensorUtil::Free(NDArray* tensor) {
  delete tensor;
}

// Resize tensor to nDimension with length size[i] in dimension i
void TensorUtil::ResizeNd(NDArray* tensor, int nDimension, int64_t* size) {
  TShape mx_shape(nDimension);
  for (int idx = 0; idx < nDimension; ++idx) {
    mx_shape[idx] = size[idx];
  }
  tensor->Reshape(mx_shape);
}

// Copy from tensor to output
// TODO(ctcyang): Is priority 0 okay?
void TensorUtil::Copy(NDArray* output, NDArray* tensor) {
  if (tensor->shape() != output->shape())
    output->ReshapeAndAlloc(tensor->shape());
  CopyFromTo(*tensor, output, 0);
}

// Elementwise division of tensor by value in-place
void TensorUtil::DivideTensorInPlace(NDArray* tensor, int value) {
  *tensor /= value;
}

#ifdef HAVE_CUDA
void TensorUtil::CopyCPUToCuda(NDArray* cpu, NDArray* cuda) {
  TensorUtil::Copy(cuda, cpu);
}

void TensorUtil::AsyncCopyCudaToCPU(NDArray* cuda, NDArray* cpu) {
  TensorUtil::Copy(cpu, cuda);
}
#endif

} // namespace MX
} // namespace horovod
