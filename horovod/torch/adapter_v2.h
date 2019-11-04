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

#ifndef HOROVOD_TORCH_ADAPTER_V2_H
#define HOROVOD_TORCH_ADAPTER_V2_H

#include <torch/extension.h>
#include <torch/torch.h>

#include "../common/common.h"

namespace horovod {
namespace torch {

using namespace horovod::common;

::torch::ScalarType GetTorchDataType(DataType dtype);

class TorchPersistentBuffer : public PersistentBuffer {
public:
  TorchPersistentBuffer(int device, int64_t size);
  virtual const void*
  AccessData(std::shared_ptr<OpContext> context) const override;

private:
  int device_ = CPU_DEVICE_ID;
  ::torch::Tensor tensor_;
};

class TorchTensor : public Tensor {
public:
  TorchTensor(::torch::Tensor tensor);
  virtual const DataType dtype() const override;
  virtual const TensorShape shape() const override;
  virtual const void* data() const override;
  virtual int64_t size() const override;

protected:
  ::torch::Tensor tensor_;
};

class TorchOpContext : public OpContext {
public:
  TorchOpContext(int device, ::torch::Tensor output);
  virtual Status
  AllocatePersistent(int64_t size,
                     std::shared_ptr<PersistentBuffer>* tensor) override;
  virtual Status AllocateOutput(TensorShape shape,
                                std::shared_ptr<Tensor>* tensor) override;
  virtual Status AllocateZeros(int64_t num_elements, DataType dtype,
                                std::shared_ptr<Tensor>* tensor) override;
  virtual Framework framework() const override;

private:
  int device_ = CPU_DEVICE_ID;
  ::torch::Tensor output_;
};

void ThrowIfError(Status status);

} // namespace torch
} // namespace horovod

#endif // HOROVOD_TORCH_ADAPTER_V2_H
