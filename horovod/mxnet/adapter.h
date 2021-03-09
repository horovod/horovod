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

#ifndef HOROVOD_MXNET_ADAPTER_H
#define HOROVOD_MXNET_ADAPTER_H

#include <mxnet/base.h>
#include "../common/common.h"

namespace horovod {
namespace mxnet {

using namespace horovod::common;

typedef ::mxnet::NDArray NDArray;

class MXPersistentBuffer : public PersistentBuffer {
public:
  MXPersistentBuffer(int device, int64_t size);
  virtual const void*
  AccessData(std::shared_ptr<OpContext> context) const override;

private:
  int device_;
  void* buffer_;
};

class MXTensor : public Tensor {
public:
  MXTensor(NDArray *tensor);
  virtual const DataType dtype() const override;
  virtual const TensorShape shape() const override;
  virtual const void* data() const override;
  virtual int64_t size() const override;

protected:
  NDArray* tensor_;
};

class MXOpContext : public OpContext {
public:
  MXOpContext(int device, NDArray* principal_output);
  void AddOutput(NDArray* output);
  virtual Status
  AllocatePersistent(int64_t size,
                     std::shared_ptr<PersistentBuffer>* tensor) override;
  virtual Status AllocateOutput(TensorShape shape,
                                std::shared_ptr<Tensor>* tensor) override;
  virtual Status AllocateOutput(int output_index, TensorShape shape,
                                std::shared_ptr<Tensor>* tensor) override;
  virtual Status AllocateZeros(int64_t num_elements, DataType dtype,
                               std::shared_ptr<Tensor>* tensor) override;
  virtual Framework framework() const override;

private:
  int device_;
  std::vector<NDArray*> outputs_;
};

void ThrowIfError(const Status& status);

} // namespace mxnet
} // namespace horovod

#endif // HOROVOD_MXNET_ADAPTER_H
