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

#if HAVE_CUDA
#include <THC/THC.h>
#endif

#include "adapter_v2.h"
#include "cuda_util.h"

#if HAVE_CUDA
extern THCState* state;
#endif

namespace horovod {
namespace torch {

// This class intentionally does not have destructor at the moment.
//
// Unfortunately, by the time this destructor would be called in normal
// circumstances (application shutdown), CUDA context would already be destroyed
// and cudaFree() operations would print nasty errors in the log - in a pretty
// normal termination scenario.
//
// If we add functionality to terminate Horovod without terminating the
// application, we should revisit this logic.
TorchPersistentBuffer::TorchPersistentBuffer(int device, int64_t size)
    : device_(device) {
  with_device device_context(device_);
  if (device_ == CPU_DEVICE_ID) {
    buffer_ = new char[size];
  } else {
#if HAVE_CUDA
    buffer_ = THCudaMalloc(state, size);
#else
    throw std::logic_error("Internal error. Requested TorchPersistentBuffer "
                           "with GPU device but not compiled with CUDA.");
#endif
  }
}

const void*
TorchPersistentBuffer::AccessData(std::shared_ptr<OpContext> context) const {
  return buffer_;
}

TorchTensor::TorchTensor(at::Tensor tensor) : tensor_(tensor) {}

const MPIDataType TorchTensor::dtype() const {
  switch (tensor_.dtype()) {
  case at::ScalarType::Byte:
    return common::HOROVOD_UINT8;
  case at::ScalarType::Char:
    return common::HOROVOD_INT8;
  case at::ScalarType::Short:
    return common::HOROVOD_INT16;
  case at::ScalarType::Int:
    return common::HOROVOD_INT32;
  case at::ScalarType::Long:
    return common::HOROVOD_INT64;
  case at::ScalarType::Float:
    return common::HOROVOD_FLOAT32;
  case at::ScalarType::Double:
    return common::HOROVOD_FLOAT64;
  default:
    throw std::logic_error("Invalid tensor type.");
  }
}

const TensorShape TorchTensor::shape() const {
  TensorShape shape;
  for (int idx = 0; idx < tensor_.dim(); idx++) {
    shape.AddDim(tensor_.size(idx));
  }
  if (shape.dims() == 0) {
    // Tensor with empty shape is a Tensor with no values in PyTorch, unlike a
    // constant in TensorFlow. So, we inject a dummy zero dimension to make sure
    // that the number-of-elements calculation is correct.
    shape.AddDim(0);
  }
  return shape;
}

const void* TorchTensor::data() const { return tensor_.data_ptr(); }

int64_t TorchTensor::size() const {
  return tensor_.type().elementSizeInBytes() * tensor_.numel();
}

TorchOpContext::TorchOpContext(int device, at::Tensor output)
    : device_(device), output_(output) {}

Status
TorchOpContext::AllocatePersistent(int64_t size,
                                   std::shared_ptr<PersistentBuffer>* tensor) {
  // Allocation errors are handled using PyTorch exceptions.
  *tensor = std::make_shared<TorchPersistentBuffer>(device_, size);
  return Status::OK();
}

Status TorchOpContext::AllocateOutput(TensorShape shape,
                                      std::shared_ptr<Tensor>* tensor) {
  std::vector<int64_t> shape_vector;
  for (int idx = 0; idx < shape.dims(); idx++) {
    shape_vector.push_back(shape.dim_size(idx));
  }
  output_.resize_(shape_vector);
  *tensor = std::make_shared<TorchTensor>(output_);
  return Status::OK();
}

Framework TorchOpContext::framework() const {
  return Framework::PYTORCH;
}

void ThrowIfError(Status status) {
  switch (status.type()) {
  case StatusType::OK:
    return;
  case StatusType::PRECONDITION_ERROR:
    throw std::logic_error(status.reason());
  case StatusType::ABORTED:
    throw std::runtime_error(status.reason());
  default: // Includes UNKNOWN_ERROR
    throw std::runtime_error(status.reason());
  }
}

} // namespace torch
} // namespace horovod