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
namespace mxnet {

// Define all types for TensorUtil.
const MPIDataType TensorUtil::GetDType(NDArray* tensor) {
  switch (tensor->dtype()) {
    case 0:
      return MPIDataType::HOROVOD_FLOAT32;
    case 1:
      return MPIDataType::HOROVOD_FLOAT64;
    case 2:
      return MPIDataType::HOROVOD_FLOAT16;
    case 3:
      return MPIDataType::HOROVOD_UINT8;
    case 4:
      return MPIDataType::HOROVOD_INT32;
    case 5:
      return MPIDataType::HOROVOD_INT8;
    case 6:
      return MPIDataType::HOROVOD_INT64;
    default:
      throw std::logic_error("Type " + tensor->dtype() +
                             " is not supported in MPI mode.");
  }
}

// Return shape of tensor (similar to TShape)
const TensorShape TensorUtil::GetShape(NDArray* tensor) {
  TensorShape shape;
  TShape mx_shape = tensor->shape();
  for (int idx = 0; idx < mx_shape.ndim(); idx++) {
    shape.AddDim(mx_shape[idx]);
  }
  return shape;
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
    case 2:
      element_size = kFloat16Size;
      break;
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
      throw std::logic_error("Type " + tensor->dtype() +
                             " is not supported in MPI mode.");
  }
  return (int64_t)(tensor->shape().Size() * element_size);
}

// If Tensor on GPU, return device id
// Otherwise return CPU_DEVICE_ID (-1)
int TensorUtil::GetDevice(NDArray* tensor) {
  int dev_mask = tensor->ctx().dev_mask();
  if (dev_mask == cpu::kDevMask)
    return CPU_DEVICE_ID;
  else if (dev_mask == gpu::kDevMask)
    return tensor->ctx().real_dev_id();
}

// Returns pointer to newly created NDArray
// If dev_id equal to CPU_DEVICE_ID, construct Tensor on CPU
// Otherwise construct on GPU
NDArray* TensorUtil::New(int device) {
  if (device == CPU_DEVICE_ID)
    return &NDArray(TShape(), Context::CPU(0));
  else
    // TODO(ctcyang): Test whether MXNet integration works fine without this
    // line that PyTorch requires
    //with_device device_context(device);
    return &NDArray(TShape(), Context::GPU(device));
}

void TensorUtil::Free(NDArray* tensor) {
  // TODO(ctcyang): Does this way of destroying NDArray work?
  delete *tensor;
}

// Resize tensor to nDimension with length size[i] in dimension i
void TensorUtil::ResizeNd(NDArray* tensor, int nDimension, 
                                   int64_t* size) {
  TShape mx_shape(nDimension);
  for (int idx = 0; idx < nDimension; ++idx) {
    mx_shape[idx] = size[idx];
  }
  tensor->Reshape(mx_shape);
}

// Copy from tensor to output
// TODO(ctcyang): Is priority 0 okay?
void TensorUtil::Copy(NDArray* output, NDArray* tensor) {
  CopyFromTo(tensor, output, 0)
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

} // namespace mxnet
} // namespace horovod
