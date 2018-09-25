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
#include "cuda.h"
#endif

#include <mxnet/base.h>

#include "adapter.h"
#include "cuda_util.h"
#include "tensor_util.h"

namespace horovod {
namespace MX {

// This class intentionally does not have destructor at the moment.
//
// Unfortunately, by the time this destructor would be called in normal
// circumstances (application shutdown), CUDA context would already be destroyed
// and cudaFree() operations would print nasty errors in the log - in a pretty
// normal termination scenario.
//
// If we add functionality to terminate Horovod without terminating the
// application, we should revisit this logic.
MXPersistentBuffer::MXPersistentBuffer(int device, int64_t size)
    : device_(device) {
  with_device device_context(device_);
  if (device_ == CPU_DEVICE_ID) {
    buffer_ = new char[size];
  } else {
#if HAVE_CUDA
    CUDA_CALL(cudaMalloc((void**)&buffer_, size));
#else
    throw std::logic_error("Internal error. Requested MXPersistentBuffer "
                           "with GPU device but not compiled with CUDA.");
#endif
  }
}

const void*
MXPersistentBuffer::AccessData(std::shared_ptr<OpContext> context) const {
  return buffer_;
}

template <class T> MXTensor<T>::MXTensor(T* tensor) : tensor_(tensor) {}

template <class T> const MPIDataType MXTensor<T>::dtype() const {
  return TensorUtil::GetDType(tensor_);
}

template <class T> const TensorShape MXTensor<T>::shape() const {
  auto shape = TensorUtil::GetShape(tensor_);
  if (shape.dims() == 0) {
    // Tensor with empty shape is a Tensor with no values in MXNet, unlike a
    // constant in TensorFlow. So, we inject a dummy zero dimension to make sure
    // that the number-of-elements calculation is correct.
    shape.AddDim(0);
  }
  return shape;
}

template <class T> const void* MXTensor<T>::data() const {
  return TensorUtil::GetData(tensor_);
}

template <class T> int64_t MXTensor<T>::size() const {
  return TensorUtil::GetSize(tensor_);
}

template <class T>
MXTemporaryBuffer<T>::MXTemporaryBuffer(int device, int dtype)
    : MXTensor<T>(nullptr) {
  this->tensor_ = TensorUtil::New(device, dtype);
}

template <class T> MXTemporaryBuffer<T>::~MXTemporaryBuffer() {
  TensorUtil::Free(this->tensor_);
}

template <class T> T* MXTemporaryBuffer<T>::tensor() const {
  return this->tensor_;
}

template <class T>
MXOpContext<T>::MXOpContext(int device, T* output)
    : device_(device), output_(output) {}

template <class T>
Status MXOpContext<T>::AllocatePersistent(
    int64_t size, std::shared_ptr<PersistentBuffer>* tensor) {
  // Allocation errors are handled using PyMX exceptions.
  *tensor = std::make_shared<MXPersistentBuffer>(device_, size);
  return Status::OK();
}

template <class T>
Status MXOpContext<T>::AllocateOutput(TensorShape shape,
                                      std::shared_ptr<Tensor>* tensor) {
  int64_t* shape_array = new int64_t[shape.dims()];
  for (int idx = 0; idx < shape.dims(); idx++) {
    shape_array[idx] = shape.dim_size(idx);
  }
  TensorUtil::ResizeNd(output_, shape.dims(), shape_array);
  delete[] shape_array;
  *tensor = std::make_shared<MXTensor<T>>(output_);
  return Status::OK();
}

template <class T> Framework MXOpContext<T>::framework() const {
  return Framework::MXNET;
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

template class MXTensor<NDArray>;
template class MXTemporaryBuffer<NDArray>;
template class MXOpContext<NDArray>;

} // namespace MX
} // namespace horovod
