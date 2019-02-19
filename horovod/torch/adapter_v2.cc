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

#include "adapter_v2.h"
#include "cuda_util.h"

namespace horovod {
namespace torch {

TorchPersistentBuffer::TorchPersistentBuffer(int device, int64_t size)
    : device_(device) {
  with_device device_context(device_);
  if (device_ == CPU_DEVICE_ID) {
    tensor_ = ::torch::empty(size, ::torch::device(::torch::kCPU).dtype(::torch::kByte));
  } else {
    tensor_ = ::torch::empty(size, ::torch::device(::torch::kCUDA).dtype(::torch::kByte));
  }
}

const void*
TorchPersistentBuffer::AccessData(std::shared_ptr<OpContext> context) const {
  return tensor_.data_ptr();
}

TorchTensor::TorchTensor(::torch::Tensor tensor) : tensor_(tensor) {}

const DataType TorchTensor::dtype() const {
  switch (tensor_.scalar_type()) {
  case ::torch::kByte:
    return common::HOROVOD_UINT8;
  case ::torch::kChar:
    return common::HOROVOD_INT8;
  case ::torch::kShort:
    return common::HOROVOD_INT16;
  case ::torch::kInt:
    return common::HOROVOD_INT32;
  case ::torch::kLong:
    return common::HOROVOD_INT64;
  case ::torch::kHalf:
    return common::HOROVOD_FLOAT16;
  case ::torch::kFloat:
    return common::HOROVOD_FLOAT32;
  case ::torch::kDouble:
    return common::HOROVOD_FLOAT64;
  default:
    throw std::logic_error("Invalid tensor type.");
  }
}

const TensorShape TorchTensor::shape() const {
  TensorShape shape;
  for (int idx = 0; idx < tensor_.dim(); ++idx) {
    shape.AddDim(tensor_.size(idx));
  }
  return shape;
}

const void* TorchTensor::data() const { return tensor_.data_ptr(); }

int64_t TorchTensor::size() const {
  return tensor_.type().elementSizeInBytes() * tensor_.numel();
}

TorchOpContext::TorchOpContext(int device, ::torch::Tensor output)
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
  shape_vector.reserve(shape.dims());
  for (int idx = 0; idx < shape.dims(); ++idx) {
    shape_vector.push_back(shape.dim_size(idx));
  }
  with_device device_context(device_);
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
  case StatusType::INVALID_ARGUMENT:
    throw std::invalid_argument(status.reason());
  default: // Includes UNKNOWN_ERROR
    throw std::runtime_error(status.reason());
  }
}

} // namespace torch
} // namespace horovod
