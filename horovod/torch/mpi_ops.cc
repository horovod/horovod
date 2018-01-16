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

#include <atomic>
#include <cassert>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

#include "../common/operations.h"
#include "mpi_ops.h"

#if HAVE_CUDA
extern THCState* state;
#endif

namespace horovod {
namespace torch {

using namespace horovod::common;

namespace {

struct TorchGlobalState {
  std::atomic_int handle;

  std::unordered_map<int, std::shared_ptr<Status>> results;

  std::mutex mutex;
};

static TorchGlobalState hvd_state;

class TorchPersistentBuffer : public PersistentBuffer {
public:
  TorchPersistentBuffer(int device, int64_t size);
  virtual const char*
  AccessData(std::shared_ptr<OpContext> context) const override;

private:
  int device_;
  char* buffer_;
};

template <class T> class TorchTensor : public Tensor {
public:
  TorchTensor(T* tensor);
  virtual const MPIDataType dtype() const override;
  virtual const TensorShape shape() const override;
  virtual const char* data() const override;
  virtual int64_t size() const override;
  virtual int device() const;
  virtual void DivideBySizeInPlace();

protected:
  T* tensor_;
};

template <class T> class TorchOpContext : public OpContext {
public:
  TorchOpContext(int device, T* output);
  virtual Status
  AllocatePersistent(int64_t size,
                     std::shared_ptr<PersistentBuffer>* tensor) override;
  virtual Status AllocateOutput(TensorShape shape,
                                std::shared_ptr<Tensor>* tensor) override;
  virtual Framework framework() const override;
  void ResizeNd(T* tensor, int nDimension, int64_t* size, int64_t* stride);
  void Copy(T* output, T* tensor);

private:
  int device_;
  T* output_;
};

TorchPersistentBuffer::TorchPersistentBuffer(int device, int64_t size)
    : device_(device) {
  if (device_ == CPU_DEVICE_ID) {
    buffer_ = new char[size];
  } else {
#if HAVE_CUDA
    int restoreDevice;
    THCudaCheck(cudaGetDevice(&restoreDevice));
    THCudaCheck(cudaSetDevice(device_));
    THCudaCheck(THCudaMalloc(state, (void**)&buffer_, size));
    THCudaCheck(cudaSetDevice(restoreDevice));
#else
    assert(false);
#endif
  }
}

const char*
TorchPersistentBuffer::AccessData(std::shared_ptr<OpContext> context) const {
  return (const char*)buffer_;
}

template <class T> TorchTensor<T>::TorchTensor(T* tensor) : tensor_(tensor) {}

template <class T>
TorchOpContext<T>::TorchOpContext(int device, T* output)
    : device_(device), output_(output) {}

template <class T>
Status TorchOpContext<T>::AllocatePersistent(
    int64_t size, std::shared_ptr<PersistentBuffer>* tensor) {
  // Allocation errors are handled using PyTorch exceptions.
  *tensor = std::make_shared<TorchPersistentBuffer>(device_, size);
  return Status::OK();
}

template <class T>
Status TorchOpContext<T>::AllocateOutput(TensorShape shape,
                                         std::shared_ptr<Tensor>* tensor) {
  int64_t* shape_array = new int64_t[shape.dims()];
  for (int idx = 0; idx < shape.dims(); idx++) {
    shape_array[idx] = shape.dim_size(idx);
  }
  ResizeNd(output_, shape.dims(), shape_array, nullptr);
  delete[] shape_array;
  *tensor = std::make_shared<TorchTensor<T>>(output_);
  return Status::OK();
}

template <class T> Framework TorchOpContext<T>::framework() const {
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

#define DEFINE_TYPE(THTensor, THStorage, HorovodType)                          \
  template <> const MPIDataType TorchTensor<THTensor>::dtype() const {         \
    return HorovodType;                                                        \
  }                                                                            \
                                                                               \
  template <> const TensorShape TorchTensor<THTensor>::shape() const {         \
    TensorShape shape;                                                         \
    for (int idx = 0; idx < THTensor##_nDimension(tensor_); idx++) {           \
      shape.AddDim(THTensor##_size(tensor_, idx));                             \
    }                                                                          \
    return shape;                                                              \
  }                                                                            \
                                                                               \
  template <> const char* TorchTensor<THTensor>::data() const {                \
    return (const char*)THTensor##_data(tensor_);                              \
  }                                                                            \
                                                                               \
  template <> int64_t TorchTensor<THTensor>::size() const {                    \
    return (int64_t)(THStorage##_size(tensor_->storage) *                      \
                     THStorage##_elementSize());                               \
  }                                                                            \
                                                                               \
  template <> int TorchTensor<THTensor>::device() const {                      \
    return CPU_DEVICE_ID;                                                      \
  }                                                                            \
                                                                               \
  template <> void TorchTensor<THTensor>::DivideBySizeInPlace() {              \
    THTensor##_div(tensor_, tensor_, horovod_size());                          \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TorchOpContext<THTensor>::ResizeNd(THTensor* tensor, int nDimension,    \
                                          int64_t* size, int64_t* stride) {    \
    THTensor##_resizeNd(tensor, nDimension, size, stride);                     \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TorchOpContext<THTensor>::Copy(THTensor* output, THTensor* tensor) {    \
    THTensor##_copy(output, tensor);                                           \
  }

DEFINE_TYPE(THByteTensor, THByteStorage, MPIDataType::HOROVOD_UINT8)
DEFINE_TYPE(THCharTensor, THCharStorage, MPIDataType::HOROVOD_INT8)
DEFINE_TYPE(THShortTensor, THShortStorage, MPIDataType::HOROVOD_INT16)
DEFINE_TYPE(THIntTensor, THIntStorage, MPIDataType::HOROVOD_INT32)
DEFINE_TYPE(THLongTensor, THLongStorage, MPIDataType::HOROVOD_INT64)
DEFINE_TYPE(THFloatTensor, THFloatStorage, MPIDataType::HOROVOD_FLOAT32)
DEFINE_TYPE(THDoubleTensor, THDoubleStorage, MPIDataType::HOROVOD_FLOAT64)

#if HAVE_CUDA
#define DEFINE_CUDA_TYPE(THTensor, THStorage, HorovodType)                     \
  template <> const MPIDataType TorchTensor<THTensor>::dtype() const {         \
    return HorovodType;                                                        \
  }                                                                            \
                                                                               \
  template <> const TensorShape TorchTensor<THTensor>::shape() const {         \
    TensorShape shape;                                                         \
    for (int idx = 0; idx < THTensor##_nDimension(state, tensor_); idx++) {    \
      shape.AddDim(THTensor##_size(state, tensor_, idx));                      \
    }                                                                          \
    return shape;                                                              \
  }                                                                            \
                                                                               \
  template <> const char* TorchTensor<THTensor>::data() const {                \
    return (const char*)THTensor##_data(state, tensor_);                       \
  }                                                                            \
                                                                               \
  template <> int64_t TorchTensor<THTensor>::size() const {                    \
    return (int64_t)(THStorage##_size(state, tensor_->storage) *               \
                     THStorage##_elementSize(state));                          \
  }                                                                            \
                                                                               \
  template <> int TorchTensor<THTensor>::device() const {                      \
    return THTensor##_getDevice(state, tensor_);                               \
  }                                                                            \
                                                                               \
  template <> void TorchTensor<THTensor>::DivideBySizeInPlace() {              \
    THTensor##_div(state, tensor_, tensor_, horovod_size());                   \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TorchOpContext<THTensor>::ResizeNd(THTensor* tensor, int nDimension,    \
                                          int64_t* size, int64_t* stride) {    \
    THTensor##_resizeNd(state, tensor, nDimension, size, stride);              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TorchOpContext<THTensor>::Copy(THTensor* output, THTensor* tensor) {    \
    THTensor##_copy(state, output, tensor);                                    \
  }

DEFINE_CUDA_TYPE(THCudaByteTensor, THCudaByteStorage,
                 MPIDataType::HOROVOD_UINT8)
DEFINE_CUDA_TYPE(THCudaCharTensor, THCudaCharStorage, MPIDataType::HOROVOD_INT8)
DEFINE_CUDA_TYPE(THCudaShortTensor, THCudaShortStorage,
                 MPIDataType::HOROVOD_INT16)
DEFINE_CUDA_TYPE(THCudaIntTensor, THCudaIntStorage, MPIDataType::HOROVOD_INT32)
DEFINE_CUDA_TYPE(THCudaLongTensor, THCudaLongStorage,
                 MPIDataType::HOROVOD_INT64)
DEFINE_CUDA_TYPE(THCudaTensor, THCudaStorage, MPIDataType::HOROVOD_FLOAT32)
DEFINE_CUDA_TYPE(THCudaDoubleTensor, THCudaDoubleStorage,
                 MPIDataType::HOROVOD_FLOAT64)
#endif

int AllocateHandle() {
  int handle = hvd_state.handle.fetch_add(1) + 1;
  std::lock_guard<std::mutex> guard(hvd_state.mutex);
  hvd_state.results[handle] = nullptr;
  return handle;
}

void MarkDone(int handle, const Status& status) {
  std::lock_guard<std::mutex> guard(hvd_state.mutex);
  hvd_state.results[handle] = std::make_shared<Status>(status);
}

bool PollHandle(int handle) {
  std::lock_guard<std::mutex> guard(hvd_state.mutex);
  if (hvd_state.results.find(handle) == hvd_state.results.end()) {
    throw std::invalid_argument("Handle " + std::to_string(handle) +
                                " was not created or has been cleared.");
  }
  return hvd_state.results[handle] != nullptr;
}

std::shared_ptr<Status> ReleaseHandle(int handle) {
  std::lock_guard<std::mutex> guard(hvd_state.mutex);
  auto status = hvd_state.results[handle];
  hvd_state.results.erase(handle);
  return status;
}

template <class T>
int DoAllreduce(T* tensor, T* output, int average, char* name) {
  ThrowIfError(common::CheckInitialized());
  auto handle = AllocateHandle();
  auto hvd_tensor = std::make_shared<TorchTensor<T>>(tensor);
  auto hvd_context =
      std::make_shared<TorchOpContext<T>>(hvd_tensor->device(), output);
  auto hvd_output = std::make_shared<TorchTensor<T>>(output);
  auto name_or_handle =
      name != nullptr ? std::string(name) : "noname." + std::to_string(handle);
  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_tensor, hvd_output, nullptr,
      "allreduce." + name_or_handle, hvd_tensor->device(),
      [handle, average, hvd_output](const Status& status) {
        if (average) {
          hvd_output->DivideBySizeInPlace();
        }
        MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);
  return handle;
}

template <class T> int DoAllgather(T* tensor, T* output, char* name) {
  ThrowIfError(common::CheckInitialized());
  auto handle = AllocateHandle();
  auto hvd_tensor = std::make_shared<TorchTensor<T>>(tensor);
  auto hvd_context =
      std::make_shared<TorchOpContext<T>>(hvd_tensor->device(), output);
  auto name_or_handle =
      name != nullptr ? std::string(name) : "noname." + std::to_string(handle);
  auto enqueue_result = EnqueueTensorAllgather(
      hvd_context, hvd_tensor, nullptr, "allgather." + name_or_handle,
      hvd_tensor->device(),
      [handle](const Status& status) { MarkDone(handle, status); });
  ThrowIfError(enqueue_result);
  return handle;
}

template <class T>
int DoBroadcast(T* tensor, T* output, int root_rank, char* name) {
  ThrowIfError(common::CheckInitialized());
  auto handle = AllocateHandle();
  auto hvd_tensor = std::make_shared<TorchTensor<T>>(tensor);
  auto hvd_context =
      std::make_shared<TorchOpContext<T>>(hvd_tensor->device(), output);
  std::shared_ptr<Tensor> hvd_output = nullptr;
  if (horovod_rank() == root_rank) {
    if (tensor != output) {
      hvd_context->Copy(output, tensor);
    }
  } else {
    hvd_output = std::make_shared<TorchTensor<T>>(output);
  }
  auto name_or_handle =
      name != nullptr ? std::string(name) : "noname." + std::to_string(handle);
  auto enqueue_result = EnqueueTensorBroadcast(
      hvd_context, hvd_tensor, hvd_output, root_rank, nullptr,
      "broadcast." + name_or_handle, hvd_tensor->device(),
      [handle](const Status& status) { MarkDone(handle, status); });
  ThrowIfError(enqueue_result);
  return handle;
}

} // namespace

#define ALLREDUCE(torch_Tensor, THTensor)                                      \
  extern "C" int horovod_torch_allreduce_async_##torch_Tensor(                 \
      THTensor* tensor, THTensor* output, int average, char* name) {           \
    return DoAllreduce<THTensor>(tensor, output, average, name);               \
  }

ALLREDUCE(torch_IntTensor, THIntTensor)
ALLREDUCE(torch_LongTensor, THLongTensor)
ALLREDUCE(torch_FloatTensor, THFloatTensor)
ALLREDUCE(torch_DoubleTensor, THDoubleTensor)

#if HOROVOD_GPU_ALLREDUCE
ALLREDUCE(torch_cuda_IntTensor, THCudaIntTensor)
ALLREDUCE(torch_cuda_LongTensor, THCudaLongTensor)
ALLREDUCE(torch_cuda_FloatTensor, THCudaTensor)
ALLREDUCE(torch_cuda_DoubleTensor, THCudaDoubleTensor)
#endif

#define ALLGATHER(torch_Tensor, THTensor)                                      \
  extern "C" int horovod_torch_allgather_async_##torch_Tensor(                 \
      THTensor* tensor, THTensor* output, char* name) {                        \
    return DoAllgather<THTensor>(tensor, output, name);                        \
  }

ALLGATHER(torch_ByteTensor, THByteTensor)
ALLGATHER(torch_CharTensor, THCharTensor)
ALLGATHER(torch_ShortTensor, THShortTensor)
ALLGATHER(torch_IntTensor, THIntTensor)
ALLGATHER(torch_LongTensor, THLongTensor)
ALLGATHER(torch_FloatTensor, THFloatTensor)
ALLGATHER(torch_DoubleTensor, THDoubleTensor)

#if HOROVOD_GPU_ALLGATHER
ALLGATHER(torch_cuda_ByteTensor, THCudaByteTensor)
ALLGATHER(torch_cuda_CharTensor, THCudaCharTensor)
ALLGATHER(torch_cuda_ShortTensor, THCudaShortTensor)
ALLGATHER(torch_cuda_IntTensor, THCudaIntTensor)
ALLGATHER(torch_cuda_LongTensor, THCudaLongTensor)
ALLGATHER(torch_cuda_FloatTensor, THCudaTensor)
ALLGATHER(torch_cuda_DoubleTensor, THCudaDoubleTensor)
#endif

#define BROADCAST(torch_Tensor, THTensor)                                      \
  extern "C" int horovod_torch_broadcast_async_##torch_Tensor(                 \
      THTensor* tensor, THTensor* output, int root_rank, char* name) {         \
    return DoBroadcast<THTensor>(tensor, output, root_rank, name);             \
  }

BROADCAST(torch_ByteTensor, THByteTensor)
BROADCAST(torch_CharTensor, THCharTensor)
BROADCAST(torch_ShortTensor, THShortTensor)
BROADCAST(torch_IntTensor, THIntTensor)
BROADCAST(torch_LongTensor, THLongTensor)
BROADCAST(torch_FloatTensor, THFloatTensor)
BROADCAST(torch_DoubleTensor, THDoubleTensor)

#if HOROVOD_GPU_BROADCAST
BROADCAST(torch_cuda_ByteTensor, THCudaByteTensor)
BROADCAST(torch_cuda_CharTensor, THCudaCharTensor)
BROADCAST(torch_cuda_ShortTensor, THCudaShortTensor)
BROADCAST(torch_cuda_IntTensor, THCudaIntTensor)
BROADCAST(torch_cuda_LongTensor, THCudaLongTensor)
BROADCAST(torch_cuda_FloatTensor, THCudaTensor)
BROADCAST(torch_cuda_DoubleTensor, THCudaDoubleTensor)
#endif

extern "C" int horovod_torch_poll(int handle) {
  return PollHandle(handle) ? 1 : 0;
}

extern "C" void horovod_torch_wait_and_clear(int handle) {
  while (!PollHandle(handle)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  auto status = ReleaseHandle(handle);
  ThrowIfError(*status);
}

} // namespace torch
} // namespace horovod