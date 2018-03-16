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

// TODO: remove
//#define HAVE_CUDA 1
//#include "cuda_runtime.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

#include "../common/operations.h"
#include "handle_manager.h"
#include "mpi_ops.h"

#if HAVE_CUDA
extern THCState* state;
#endif

namespace horovod {
namespace torch {

using namespace horovod::common;

namespace {

#if HAVE_CUDA
template <class T> class TorchReadyEvent : public ReadyEvent {
public:
  TorchReadyEvent(int device);
  ~TorchReadyEvent();
  virtual bool Ready() const override;

private:
  int device_;
  cudaEvent_t cuda_event_;
  static std::unordered_map<int, std::queue<cudaEvent_t>> cuda_events_;
  static std::mutex mutex_;
};
#endif

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

protected:
  T* tensor_;
};

template <class T> class TorchTemporaryBuffer : public TorchTensor<T> {
public:
  TorchTemporaryBuffer();
  ~TorchTemporaryBuffer();
  virtual T* tensor() const;
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

private:
  int device_;
  T* output_;
};

// Utility functions.
class TensorUtil {
public:
  template <class T> static T* New();
  template <class T> static void Free(T* tensor);
  template <class T>
  static void ResizeNd(T* tensor, int nDimension, int64_t* size,
                       int64_t* stride);
  template <class T> static void Copy(T* output, T* tensor);
  template <class T> static int GetDevice(T* tensor);
  template <class T> static void DivideTensorBySizeInPlace(T* tensor);
#if HAVE_CUDA
  template <class T, class TC> static void CopyCPUToCuda(T* cpu, TC* cuda);
  template <class TC, class T> static void AsyncCopyCudaToCPU(TC* cuda, T* cpu);
#endif
};

#if HAVE_CUDA
template <class T>
TorchReadyEvent<T>::TorchReadyEvent(int device) : device_(device) {
  assert(device_ != CPU_DEVICE_ID);

  int restoreDevice;
  THCudaCheck(cudaGetDevice(&restoreDevice));
  THCudaCheck(cudaSetDevice(device_));
  {
    std::lock_guard<std::mutex> guard(mutex_);
    auto& queue = cuda_events_[device_];
    if (!queue.empty()) {
      cuda_event_ = queue.front();
      queue.pop();
    } else {
      THCudaCheck(cudaEventCreateWithFlags(
          &cuda_event_, cudaEventBlockingSync | cudaEventDisableTiming));
    }
  }
  auto stream = THCState_getCurrentStreamOnDevice(state, device_);
  THCudaCheck(cudaEventRecord(cuda_event_, stream));
  THCudaCheck(cudaSetDevice(restoreDevice));
}

template <class T> TorchReadyEvent<T>::~TorchReadyEvent() {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    auto& queue = cuda_events_[device_];
    queue.push(cuda_event_);
  }
}

template <class T> bool TorchReadyEvent<T>::Ready() const {
  auto status = cudaEventQuery(cuda_event_);
  if (status == cudaErrorNotReady) {
    return false;
  }
  THCudaCheck(status);
  return true;
}
#endif

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
TorchTemporaryBuffer<T>::TorchTemporaryBuffer()
    : TorchTensor<T>(TensorUtil::New<T>()) {}

template <class T> TorchTemporaryBuffer<T>::~TorchTemporaryBuffer() {
  TensorUtil::Free<T>(this->tensor_);
}

template <class T> T* TorchTemporaryBuffer<T>::tensor() const {
  return this->tensor_;
}

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
  TensorUtil::ResizeNd(output_, shape.dims(), shape_array, nullptr);
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
  template <> THTensor* TensorUtil::New<THTensor>() {                          \
    return THTensor##_new();                                                   \
  }                                                                            \
                                                                               \
  template <> void TensorUtil::Free<THTensor>(THTensor * tensor) {             \
    THTensor##_free(tensor);                                                   \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::ResizeNd<THTensor>(THTensor * tensor, int nDimension,       \
                                      int64_t* size, int64_t* stride) {        \
    THTensor##_resizeNd(tensor, nDimension, size, stride);                     \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::Copy<THTensor>(THTensor * output, THTensor * tensor) {      \
    THTensor##_copy(output, tensor);                                           \
  }                                                                            \
                                                                               \
  template <> int TensorUtil::GetDevice<THTensor>(THTensor * tensor) {         \
    return CPU_DEVICE_ID;                                                      \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::DivideTensorBySizeInPlace<THTensor>(THTensor * tensor) {    \
    THTensor##_div(tensor, tensor, horovod_size());                            \
  }

DEFINE_TYPE(THByteTensor, THByteStorage, MPIDataType::HOROVOD_UINT8)
DEFINE_TYPE(THCharTensor, THCharStorage, MPIDataType::HOROVOD_INT8)
DEFINE_TYPE(THShortTensor, THShortStorage, MPIDataType::HOROVOD_INT16)
DEFINE_TYPE(THIntTensor, THIntStorage, MPIDataType::HOROVOD_INT32)
DEFINE_TYPE(THLongTensor, THLongStorage, MPIDataType::HOROVOD_INT64)
DEFINE_TYPE(THFloatTensor, THFloatStorage, MPIDataType::HOROVOD_FLOAT32)
DEFINE_TYPE(THDoubleTensor, THDoubleStorage, MPIDataType::HOROVOD_FLOAT64)

#if HAVE_CUDA
#define DEFINE_CUDA_TYPE(THCTensor, THTensor, THCStorage, HorovodType)         \
  template <> const MPIDataType TorchTensor<THCTensor>::dtype() const {        \
    return HorovodType;                                                        \
  }                                                                            \
                                                                               \
  template <> const TensorShape TorchTensor<THCTensor>::shape() const {        \
    TensorShape shape;                                                         \
    for (int idx = 0; idx < THCTensor##_nDimension(state, tensor_); idx++) {   \
      shape.AddDim(THCTensor##_size(state, tensor_, idx));                     \
    }                                                                          \
    return shape;                                                              \
  }                                                                            \
                                                                               \
  template <> const char* TorchTensor<THCTensor>::data() const {               \
    return (const char*)THCTensor##_data(state, tensor_);                      \
  }                                                                            \
                                                                               \
  template <> int64_t TorchTensor<THCTensor>::size() const {                   \
    return (int64_t)(THCStorage##_size(state, tensor_->storage) *              \
                     THCStorage##_elementSize(state));                         \
  }                                                                            \
                                                                               \
  template <> THCTensor* TensorUtil::New() { return THCTensor##_new(); }       \
                                                                               \
  template <> void TensorUtil::Free<THCTensor>(THCTensor * tensor) {           \
    THCTensor##_free(tensor);                                                  \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::ResizeNd<THCTensor>(THCTensor * tensor, int nDimension,     \
                                       int64_t* size, int64_t* stride) {       \
    THCTensor##_resizeNd(state, tensor, nDimension, size, stride);             \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::Copy<THCTensor>(THCTensor * output, THCTensor * tensor) {   \
    THCTensor##_copy(state, output, tensor);                                   \
  }                                                                            \
                                                                               \
  template <> int TensorUtil::GetDevice<THCTensor>(THCTensor * tensor) {       \
    return THCTensor##_getDevice(state, tensor);                               \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::DivideTensorBySizeInPlace<THCTensor>(THCTensor * tensor) {  \
    THCTensor##_div(state, tensor, tensor, horovod_size());                    \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::CopyCPUToCuda<THTensor, THCTensor>(THTensor * cpu,          \
                                                      THCTensor * cuda) {      \
    THLongStorage* size = THTensor##_newSizeOf(cpu);                           \
    THCTensor##_resize(state, cuda, size, NULL);                               \
    THLongStorage_free(size);                                                  \
    THCTensor##_copyCPU(state, cuda, cpu);                                     \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::AsyncCopyCudaToCPU<THCTensor, THTensor>(THCTensor * cuda,   \
                                                           THTensor * cpu) {   \
    THLongStorage* size = THCTensor##_newSizeOf(state, cuda);                  \
    THTensor##_resize(cpu, size, NULL);                                        \
    THLongStorage_free(size);                                                  \
    THCTensor##_copyAsyncCuda(state, cpu, cuda);                               \
  }

DEFINE_CUDA_TYPE(THCudaByteTensor, THByteTensor, THCudaByteStorage,
                 MPIDataType::HOROVOD_UINT8)
DEFINE_CUDA_TYPE(THCudaCharTensor, THCharTensor, THCudaCharStorage,
                 MPIDataType::HOROVOD_INT8)
DEFINE_CUDA_TYPE(THCudaShortTensor, THShortTensor, THCudaShortStorage,
                 MPIDataType::HOROVOD_INT16)
DEFINE_CUDA_TYPE(THCudaIntTensor, THIntTensor, THCudaIntStorage,
                 MPIDataType::HOROVOD_INT32)
DEFINE_CUDA_TYPE(THCudaLongTensor, THLongTensor, THCudaLongStorage,
                 MPIDataType::HOROVOD_INT64)
DEFINE_CUDA_TYPE(THCudaTensor, THFloatTensor, THCudaStorage,
                 MPIDataType::HOROVOD_FLOAT32)
DEFINE_CUDA_TYPE(THCudaDoubleTensor, THDoubleTensor, THCudaDoubleStorage,
                 MPIDataType::HOROVOD_FLOAT64)
#endif

template <class T>
int DoAllreduce(T* tensor, T* output, int average, char* name) {
  ThrowIfError(common::CheckInitialized());

  auto handle = HandleManager::AllocateHandle();
  auto name_or_handle =
      name != nullptr ? std::string(name) : "noname." + std::to_string(handle);

  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<TorchTensor<T>>(tensor);
  auto hvd_context = std::make_shared<TorchOpContext<T>>(device, output);
  auto hvd_output = std::make_shared<TorchTensor<T>>(output);

  auto enqueue_result =
      EnqueueTensorAllreduce(hvd_context, hvd_tensor, hvd_output, nullptr,
                             "allreduce." + name_or_handle, device,
                             [handle, average, output](const Status& status) {
                               if (average) {
                                 TensorUtil::DivideTensorBySizeInPlace(output);
                               }
                               HandleManager::MarkDone(handle, status);
                             });
  ThrowIfError(enqueue_result);

  return handle;
}

#if HAVE_CUDA
template <class TC, class T>
int DoAllreduceCudaOnCPU(TC* tensor, TC* output, int average, char* name) {
  ThrowIfError(common::CheckInitialized());

  auto handle = HandleManager::AllocateHandle();
  auto name_or_handle =
      name != nullptr ? std::string(name) : "noname." + std::to_string(handle);

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto hvd_cpu_buffer = std::make_shared<TorchTemporaryBuffer<T>>();
  TensorUtil::AsyncCopyCudaToCPU<TC, T>(tensor, hvd_cpu_buffer->tensor());
  auto ready_event = std::make_shared<TorchReadyEvent<TC>>(tensor);

  auto hvd_context = std::make_shared<TorchOpContext<T>>(
      CPU_DEVICE_ID, hvd_cpu_buffer->tensor());

  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, ready_event,
      "allreduce." + name_or_handle, CPU_DEVICE_ID,
      [handle, average, output](const Status& status) {
        TensorUtil::CopyCPUToCuda(hvd_cpu_buffer->tensor(), output);
        if (average) {
          TensorUtil::DivideTensorBySizeInPlace(output);
        }
        HandleManager::MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}
#endif

template <class T> int DoAllgather(T* tensor, T* output, char* name) {
  ThrowIfError(common::CheckInitialized());

  auto handle = HandleManager::AllocateHandle();
  auto name_or_handle =
      name != nullptr ? std::string(name) : "noname." + std::to_string(handle);

  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<TorchTensor<T>>(tensor);
  auto hvd_context = std::make_shared<TorchOpContext<T>>(device, output);

  auto enqueue_result = EnqueueTensorAllgather(
      hvd_context, hvd_tensor, nullptr, "allgather." + name_or_handle, device,
      [handle](const Status& status) {
        HandleManager::MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}

#if HAVE_CUDA
template <class TC, class T>
int DoAllgatherCudaOnCPU(TC* tensor, TC* output, char* name) {
  ThrowIfError(common::CheckInitialized());

  auto handle = HandleManager::AllocateHandle();
  auto name_or_handle =
      name != nullptr ? std::string(name) : "noname." + std::to_string(handle);

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto hvd_cpu_tensor = std::make_shared<TorchTemporaryBuffer<T>>();
  TensorUtil::AsyncCopyCudaToCPU<TC, T>(tensor, hvd_cpu_tensor->tensor());
  auto ready_event = std::make_shared<TorchReadyEvent<TC>>(tensor);

  auto hvd_cpu_output = std::make_shared<TorchTemporaryBuffer<T>>();
  auto hvd_context = std::make_shared<TorchOpContext<T>>(
      CPU_DEVICE_ID, hvd_cpu_output->tensor());

  auto enqueue_result = EnqueueTensorAllgather(
      hvd_context, hvd_cpu_tensor, ready_event, "allgather." + name_or_handle,
      CPU_DEVICE_ID, [handle, hvd_cpu_output, output](const Status& status) {
        TensorUtil::CopyCPUToCuda<T, TC>(hvd_cpu_output->tensor(), output);
        HandleManager::MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}
#endif

template <class T>
int DoBroadcast(T* tensor, T* output, int root_rank, char* name) {
  ThrowIfError(common::CheckInitialized());

  auto handle = HandleManager::AllocateHandle();
  auto name_or_handle =
      name != nullptr ? std::string(name) : "noname." + std::to_string(handle);

  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<TorchTensor<T>>(tensor);
  auto hvd_context = std::make_shared<TorchOpContext<T>>(device, output);
  std::shared_ptr<Tensor> hvd_output = nullptr;
  if (horovod_rank() == root_rank) {
    if (tensor != output) {
      TensorUtil::Copy(output, tensor);
    }
  } else {
    hvd_output = std::make_shared<TorchTensor<T>>(output);
  }

  auto enqueue_result = EnqueueTensorBroadcast(
      hvd_context, hvd_tensor, hvd_output, root_rank, nullptr,
      "broadcast." + name_or_handle, device, [handle](const Status& status) {
        HandleManager::MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}

#if HAVE_CUDA
template <class TC, class T>
int DoBroadcastCudaOnCPU(TC* tensor, TC* output, int root_rank, char* name) {
  ThrowIfError(common::CheckInitialized());

  auto handle = HandleManager::AllocateHandle();
  auto name_or_handle =
      name != nullptr ? std::string(name) : "noname." + std::to_string(handle);

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto hvd_cpu_buffer = std::make_shared<TorchTemporaryBuffer<T>>();
  TensorUtil::AsyncCopyCudaToCPU<TC, T>(tensor, hvd_cpu_buffer->tensor());
  auto ready_event = std::make_shared<TorchReadyEvent<TC>>(tensor);

  auto hvd_context = std::make_shared<TorchOpContext<T>>(
      CPU_DEVICE_ID, hvd_cpu_buffer->tensor());

  auto enqueue_result = EnqueueTensorBroadcast(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, root_rank, ready_event,
      "broadcast." + name_or_handle, CPU_DEVICE_ID,
      [handle, hvd_cpu_buffer, output](const Status& status) {
        TensorUtil::CopyCPUToCuda<T, TC>(hvd_cpu_buffer->tensor(), output);
        HandleManager::MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}
#endif

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

#define ALLREDUCE_CUDA_ON_CPU(torch_Tensor, THCTensor, THTensor)               \
  extern "C" int horovod_torch_allreduce_async_##torch_Tensor(                 \
      THCTensor* tensor, THCTensor* output, int average, char* name) {         \
    return DoAllreduceCudaOnCPU<THCTensor>(tensor, output, average, name);     \
  }

#if !HOROVOD_GPU_ALLREDUCE && HAVE_CUDA
ALLREDUCE_CUDA_ON_CPU(torch_cuda_IntTensor, THCudaIntTensor, THIntTensor)
ALLREDUCE_CUDA_ON_CPU(torch_cuda_LongTensor, THCudaLongTensor, THLongTensor)
ALLREDUCE_CUDA_ON_CPU(torch_cuda_FloatTensor, THCudaTensor, THFloatTensor)
ALLREDUCE_CUDA_ON_CPU(torch_cuda_DoubleTensor, THCudaDoubleTensor,
                      THDoubleTensor)
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

#define ALLGATHER_CUDA_ON_CPU(torch_Tensor, THCTensor, THTensor)               \
  extern "C" int horovod_torch_allgather_async_##torch_Tensor(                 \
      THTensor* tensor, THTensor* output, char* name) {                        \
    return DoAllgatherCudaOnCPU<THCTensor, THTensor>(tensor, output, name);    \
  }

#if !HOROVOD_GPU_ALLGATHER && HAVE_CUDA
ALLGATHER_CUDA_ON_CPU(torch_cuda_ByteTensor, THCudaByteTensor, THByteTensor)
ALLGATHER_CUDA_ON_CPU(torch_cuda_CharTensor, THCudaCharTensor, THCharTensor)
ALLGATHER_CUDA_ON_CPU(torch_cuda_ShortTensor, THCudaShortTensor, THShortTensor)
ALLGATHER_CUDA_ON_CPU(torch_cuda_IntTensor, THCudaIntTensor, THIntTensor)
ALLGATHER_CUDA_ON_CPU(torch_cuda_LongTensor, THCudaLongTensor, THLongTensor)
ALLGATHER_CUDA_ON_CPU(torch_cuda_FloatTensor, THCudaTensor, THFloatTensor)
ALLGATHER_CUDA_ON_CPU(torch_cuda_DoubleTensor, THCudaDoubleTensor,
                      THDoubleTensor)
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

#define BROADCAST_CUDA_ON_CPU(torch_Tensor, THCTensor, THTensor)               \
  extern "C" int horovod_torch_broadcast_async_##torch_Tensor(                 \
      THCTensor* tensor, THCTensor* output, int root_rank, char* name) {       \
    return DoBroadcastCudaOnCPU<THCTensor, THTensor>(tensor, output,           \
                                                     root_rank, name);         \
  }

#if !HOROVOD_GPU_BROADCAST && HAVE_CUDA
BROADCAST_CUDA_ON_CPU(torch_cuda_ByteTensor, THCudaByteTensor, THByteTensor)
BROADCAST_CUDA_ON_CPU(torch_cuda_CharTensor, THCudaCharTensor, THCharTensor)
BROADCAST_CUDA_ON_CPU(torch_cuda_ShortTensor, THCudaShortTensor, THShortTensor)
BROADCAST_CUDA_ON_CPU(torch_cuda_IntTensor, THCudaIntTensor, THIntTensor)
BROADCAST_CUDA_ON_CPU(torch_cuda_LongTensor, THCudaLongTensor, THLongTensor)
BROADCAST_CUDA_ON_CPU(torch_cuda_FloatTensor, THCudaTensor, THFloatTensor)
BROADCAST_CUDA_ON_CPU(torch_cuda_DoubleTensor, THCudaDoubleTensor,
                      THDoubleTensor)
#endif

extern "C" int horovod_torch_poll(int handle) {
  return HandleManager::PollHandle(handle) ? 1 : 0;
}

extern "C" void horovod_torch_wait_and_clear(int handle) {
  while (!HandleManager::PollHandle(handle)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  auto status = HandleManager::ReleaseHandle(handle);
  ThrowIfError(*status);
}

} // namespace torch
} // namespace horovod