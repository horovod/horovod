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

#include <TH/TH.h>

#if HAVE_GPU
#include <THC/THC.h>
#endif

#include "adapter.h"
#include "cuda_util.h"
#include "tensor_util.h"

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
#if HAVE_GPU
#if TORCH_VERSION >= 4001000
    buffer_ = THCudaMalloc(state, size);
#else
    THCudaCheck(THCudaMalloc(state, (void**)&buffer_, size));
#endif
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

template <DataType DT, DeviceType Dev, class T>
TorchTensor<DT, Dev, T>::TorchTensor(T* tensor) : tensor_(tensor) {}

template <DataType DT, DeviceType Dev, class T>
const DataType TorchTensor<DT, Dev, T>::dtype() const {
  return DT;
}

template <DataType DT, DeviceType Dev, class T>
const TensorShape TorchTensor<DT, Dev, T>::shape() const {
  auto shape = TensorUtil::GetShape<DT, Dev>(tensor_);
  if (shape.dims() == 0) {
    // Tensor with empty shape is a Tensor with no values in PyTorch, unlike a
    // constant in TensorFlow. So, we inject a dummy zero dimension to make sure
    // that the number-of-elements calculation is correct.
    shape.AddDim(0);
  }
  return shape;
}

template <DataType DT, DeviceType Dev, class T>
const void* TorchTensor<DT, Dev, T>::data() const {
  return TensorUtil::GetData<DT, Dev>(tensor_);
}

template <DataType DT, DeviceType Dev, class T>
int64_t TorchTensor<DT, Dev, T>::size() const {
  return TensorUtil::GetSize<DT, Dev>(tensor_);
}

template <DataType DT, DeviceType Dev, class T>
TorchTemporaryBuffer<DT, Dev, T>::TorchTemporaryBuffer(int device)
    : TorchTensor<DT, Dev, T>(nullptr) {
  this->tensor_ = TensorUtil::New<DT, Dev, T>(device);
}

template <DataType DT, DeviceType Dev, class T>
TorchTemporaryBuffer<DT, Dev, T>::~TorchTemporaryBuffer() {
  TensorUtil::Free<DT, Dev>(this->tensor_);
}

template <DataType DT, DeviceType Dev, class T>
T* TorchTemporaryBuffer<DT, Dev, T>::tensor() const {
  return this->tensor_;
}

template <DataType DT, DeviceType Dev, class T>
TorchOpContext<DT, Dev, T>::TorchOpContext(int device, T* output)
    : device_(device), output_(output) {}

template <DataType DT, DeviceType Dev, class T>
Status TorchOpContext<DT, Dev, T>::AllocatePersistent(
    int64_t size, std::shared_ptr<PersistentBuffer>* tensor) {
  // Allocation errors are handled using PyTorch exceptions.
  *tensor = std::make_shared<TorchPersistentBuffer>(device_, size);
  return Status::OK();
}

template <DataType DT, DeviceType Dev, class T>
Status
TorchOpContext<DT, Dev, T>::AllocateOutput(TensorShape shape,
                                           std::shared_ptr<Tensor>* tensor) {
  int64_t* shape_array = new int64_t[shape.dims()];
  for (int idx = 0; idx < shape.dims(); ++idx) {
    shape_array[idx] = shape.dim_size(idx);
  }
  TensorUtil::ResizeNd<DT, Dev>(output_, shape.dims(), shape_array, nullptr);
  delete[] shape_array;
  *tensor = std::make_shared<TorchTensor<DT, Dev, T>>(output_);
  return Status::OK();
}

template <DataType DT, DeviceType Dev, class T>
Status
TorchOpContext<DT, Dev, T>::AllocateZeros(int64_t num_elements, DataType dtype,
                                          std::shared_ptr<Tensor>* tensor) {
  return Status::PreconditionError(
      "AllocateZeros is not supported for PyTorch < 1.0");
}

template <DataType DT, DeviceType Dev, class T>
Framework TorchOpContext<DT, Dev, T>::framework() const {
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

ADAPTER_DEFINE_TYPE(DataType::HOROVOD_UINT8, DeviceType::CPU, THByteTensor)
ADAPTER_DEFINE_TYPE(DataType::HOROVOD_INT8, DeviceType::CPU, THCharTensor)
ADAPTER_DEFINE_TYPE(DataType::HOROVOD_INT16, DeviceType::CPU, THShortTensor)
ADAPTER_DEFINE_TYPE(DataType::HOROVOD_INT32, DeviceType::CPU, THIntTensor)
ADAPTER_DEFINE_TYPE(DataType::HOROVOD_INT64, DeviceType::CPU, THLongTensor)
ADAPTER_DEFINE_TYPE(DataType::HOROVOD_FLOAT32, DeviceType::CPU,
                    THFloatTensor)
ADAPTER_DEFINE_TYPE(DataType::HOROVOD_FLOAT64, DeviceType::CPU,
                    THDoubleTensor)

#if HAVE_GPU
ADAPTER_DEFINE_TYPE(DataType::HOROVOD_UINT8, DeviceType::GPU,
                    THCudaByteTensor)
ADAPTER_DEFINE_TYPE(DataType::HOROVOD_INT8, DeviceType::GPU,
                    THCudaCharTensor)
ADAPTER_DEFINE_TYPE(DataType::HOROVOD_INT16, DeviceType::GPU,
                    THCudaShortTensor)
ADAPTER_DEFINE_TYPE(DataType::HOROVOD_INT32, DeviceType::GPU,
                    THCudaIntTensor)
ADAPTER_DEFINE_TYPE(DataType::HOROVOD_INT64, DeviceType::GPU,
                    THCudaLongTensor)
ADAPTER_DEFINE_TYPE(DataType::HOROVOD_FLOAT32, DeviceType::GPU,
                    THCudaTensor)
ADAPTER_DEFINE_TYPE(DataType::HOROVOD_FLOAT64, DeviceType::GPU,
                    THCudaDoubleTensor)
#endif

} // namespace torch
} // namespace horovod
