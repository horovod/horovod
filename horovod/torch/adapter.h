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

#ifndef HOROVOD_TORCH_ADAPTER_H
#define HOROVOD_TORCH_ADAPTER_H

#include "../common/common.h"

namespace horovod {
namespace torch {

using namespace horovod::common;

class TorchPersistentBuffer : public PersistentBuffer {
public:
  TorchPersistentBuffer(int device, int64_t size);
  virtual const void*
  AccessData(std::shared_ptr<OpContext> context) const override;

private:
  int device_ = CPU_DEVICE_ID;
  void* buffer_ = nullptr;
};

template <DataType DT, DeviceType Dev, class T>
class TorchTensor : public Tensor {
public:
  TorchTensor(T* tensor);
  virtual const DataType dtype() const override;
  virtual const TensorShape shape() const override;
  virtual const void* data() const override;
  virtual int64_t size() const override;

protected:
  T* tensor_ = nullptr;
};

template <DataType DT, DeviceType Dev, class T>
class TorchTemporaryBuffer : public TorchTensor<DT, Dev, T> {
public:
  TorchTemporaryBuffer(int device);
  ~TorchTemporaryBuffer();
  virtual T* tensor() const;
};

template <DataType DT, DeviceType Dev, class T>
class TorchOpContext : public OpContext {
public:
  TorchOpContext(int device, T* output);
  virtual Status
  AllocatePersistent(int64_t size,
                     std::shared_ptr<PersistentBuffer>* tensor) override;
  virtual Status AllocateOutput(TensorShape shape,
                                std::shared_ptr<Tensor>* tensor) override;
  virtual Framework framework() const override;

private:
  int device_ = CPU_DEVICE_ID;
  T* output_ = nullptr;
};

void ThrowIfError(Status status);

#define ADAPTER_DEFINE_TYPE(HorovodType, DeviceType, THTensor)                     \
  template class TorchTensor<HorovodType, DeviceType, THTensor>;                   \
  template class TorchTemporaryBuffer<HorovodType, DeviceType, THTensor>;          \
  template class TorchOpContext<HorovodType, DeviceType, THTensor>;

} // namespace torch
} // namespace horovod

#endif // HOROVOD_TORCH_ADAPTER_H
