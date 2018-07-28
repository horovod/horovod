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

#include <chrono>
#include <memory>
#include <thread>

#include "../common/operations.h"
#include "adapter.h"
#include "cuda_util.h"
#include "handle_manager.h"
#include "mpi_ops.h"
#include "ready_event.h"
#include "tensor_util.h"

namespace horovod {
namespace torch {

static HandleManager handle_manager;

namespace {

std::string GetOpName(std::string prefix, char* name, int handle) {
  if (name != nullptr) {
    return prefix + "." + std::string(name);
  }
  return prefix + ".noname." + std::to_string(handle);
}

} // namespace

template <class T, MPIDataType DT>
int DoAllreduce(T* tensor, T* output, int average, char* name) {
  ThrowIfError(common::CheckInitialized());

  auto handle = handle_manager.AllocateHandle();
  auto device = TensorUtil::GetDevice<T, DT>(tensor);
  auto ready_event = RecordReadyEvent(device);
  auto hvd_tensor = std::make_shared<TorchTensor<T, DT>>(tensor);
  auto hvd_context = std::make_shared<TorchOpContext<T, DT>>(device, output);
  auto hvd_output = std::make_shared<TorchTensor<T, DT>>(output);

  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_tensor, hvd_output, ready_event,
      GetOpName("allreduce", name, handle), device,
      [handle, average, output](const Status& status) {
        if (average) {
          TensorUtil::DivideTensorInPlace<T, DT>(output, horovod_size());
        }
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}

#if HAVE_CUDA
template <class TC, class T, MPIDataType DT>
int DoAllreduceCudaOnCPU(TC* tensor, TC* output, int average, char* name) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto device = TensorUtil::GetDevice<TC, DT>(tensor);
  auto hvd_cpu_buffer =
      std::make_shared<TorchTemporaryBuffer<T, DT>>(CPU_DEVICE_ID);
  TensorUtil::AsyncCopyCudaToCPU<TC, T, DT>(tensor, hvd_cpu_buffer->tensor());
  auto ready_event = RecordReadyEvent(device);

  auto hvd_context = std::make_shared<TorchOpContext<T, DT>>(
      CPU_DEVICE_ID, hvd_cpu_buffer->tensor());

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, ready_event,
      GetOpName("allreduce", name, handle), CPU_DEVICE_ID,
      [handle, average, hvd_cpu_buffer, output](const Status& status) {
        TensorUtil::CopyCPUToCuda<T, TC, DT>(hvd_cpu_buffer->tensor(), output);
        if (average) {
          TensorUtil::DivideTensorInPlace<TC, DT>(output, horovod_size());
        }
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}
#endif

template <class T, MPIDataType DT>
int DoAllgather(T* tensor, T* output, char* name) {
  ThrowIfError(common::CheckInitialized());

  auto device = TensorUtil::GetDevice<T, DT>(tensor);
  auto ready_event = RecordReadyEvent(device);
  auto hvd_tensor = std::make_shared<TorchTensor<T, DT>>(tensor);
  auto hvd_context = std::make_shared<TorchOpContext<T, DT>>(device, output);

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result =
      EnqueueTensorAllgather(hvd_context, hvd_tensor, ready_event,
                             GetOpName("allgather", name, handle), device,
                             [handle](const Status& status) {
                               handle_manager.MarkDone(handle, status);
                             });
  ThrowIfError(enqueue_result);

  return handle;
}

#if HAVE_CUDA
template <class TC, class T, MPIDataType DT>
int DoAllgatherCudaOnCPU(TC* tensor, TC* output, char* name) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto device = TensorUtil::GetDevice<TC, DT>(tensor);
  auto hvd_cpu_tensor =
      std::make_shared<TorchTemporaryBuffer<T, DT>>(CPU_DEVICE_ID);
  TensorUtil::AsyncCopyCudaToCPU<TC, T, DT>(tensor, hvd_cpu_tensor->tensor());
  auto ready_event = RecordReadyEvent(device);

  auto hvd_cpu_output =
      std::make_shared<TorchTemporaryBuffer<T, DT>>(CPU_DEVICE_ID);
  auto hvd_context = std::make_shared<TorchOpContext<T, DT>>(
      CPU_DEVICE_ID, hvd_cpu_output->tensor());

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorAllgather(
      hvd_context, hvd_cpu_tensor, ready_event,
      GetOpName("allgather", name, handle), CPU_DEVICE_ID,
      [handle, hvd_cpu_output, output](const Status& status) {
        TensorUtil::CopyCPUToCuda<T, TC, DT>(hvd_cpu_output->tensor(), output);
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}
#endif

template <class T, MPIDataType DT>
int DoBroadcast(T* tensor, T* output, int root_rank, char* name) {
  ThrowIfError(common::CheckInitialized());

  auto device = TensorUtil::GetDevice<T, DT>(tensor);
  auto ready_event = RecordReadyEvent(device);
  auto hvd_tensor = std::make_shared<TorchTensor<T, DT>>(tensor);
  auto hvd_context = std::make_shared<TorchOpContext<T, DT>>(device, output);
  std::shared_ptr<Tensor> hvd_output = nullptr;
  if (horovod_rank() == root_rank) {
    if (tensor != output) {
      TensorUtil::Copy<T, DT>(output, tensor);
    }
  } else {
    hvd_output = std::make_shared<TorchTensor<T, DT>>(output);
  }

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result =
      EnqueueTensorBroadcast(hvd_context, hvd_tensor, hvd_output, root_rank,
                             ready_event, GetOpName("broadcast", name, handle),
                             device, [handle](const Status& status) {
                               handle_manager.MarkDone(handle, status);
                             });
  ThrowIfError(enqueue_result);

  return handle;
}

#if HAVE_CUDA
template <class TC, class T, MPIDataType DT>
int DoBroadcastCudaOnCPU(TC* tensor, TC* output, int root_rank, char* name) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto device = TensorUtil::GetDevice<TC, DT>(tensor);
  auto hvd_cpu_buffer =
      std::make_shared<TorchTemporaryBuffer<T, DT>>(CPU_DEVICE_ID);
  TensorUtil::AsyncCopyCudaToCPU<TC, T, DT>(tensor, hvd_cpu_buffer->tensor());
  auto ready_event = RecordReadyEvent(device);

  auto hvd_context = std::make_shared<TorchOpContext<T, DT>>(
      CPU_DEVICE_ID, hvd_cpu_buffer->tensor());

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorBroadcast(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, root_rank, ready_event,
      GetOpName("broadcast", name, handle), CPU_DEVICE_ID,
      [handle, hvd_cpu_buffer, output](const Status& status) {
        TensorUtil::CopyCPUToCuda<T, TC, DT>(hvd_cpu_buffer->tensor(), output);
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}
#endif

#define ALLREDUCE(torch_Tensor, THTensor, HorovodType)                         \
  extern "C" int horovod_torch_allreduce_async_##torch_Tensor(                 \
      THTensor* tensor, THTensor* output, int average, char* name) {           \
    return DoAllreduce<THTensor, HorovodType>(tensor, output, average, name);  \
  }

ALLREDUCE(torch_IntTensor, THIntTensor, MPIDataType::HOROVOD_INT32)
ALLREDUCE(torch_LongTensor, THLongTensor, MPIDataType::HOROVOD_INT64)
ALLREDUCE(torch_FloatTensor, THFloatTensor, MPIDataType::HOROVOD_FLOAT32)
ALLREDUCE(torch_DoubleTensor, THDoubleTensor, MPIDataType::HOROVOD_FLOAT64)

#if HOROVOD_GPU_ALLREDUCE
ALLREDUCE(torch_cuda_IntTensor, THCudaIntTensor, MPIDataType::HOROVOD_INT32)
ALLREDUCE(torch_cuda_LongTensor, THCudaLongTensor, MPIDataType::HOROVOD_INT64)
ALLREDUCE(torch_cuda_FloatTensor, THCudaTensor, MPIDataType::HOROVOD_FLOAT32)
ALLREDUCE(torch_cuda_DoubleTensor, THCudaDoubleTensor,
          MPIDataType::HOROVOD_FLOAT64)
#endif

#define ALLREDUCE_CUDA_ON_CPU(torch_Tensor, THCTensor, THTensor, HorovodType)  \
  extern "C" int horovod_torch_allreduce_async_##torch_Tensor(                 \
      THCTensor* tensor, THCTensor* output, int average, char* name) {         \
    return DoAllreduceCudaOnCPU<THCTensor, THTensor, HorovodType>(             \
        tensor, output, average, name);                                        \
  }

#if !HOROVOD_GPU_ALLREDUCE && HAVE_CUDA
ALLREDUCE_CUDA_ON_CPU(torch_cuda_IntTensor, THCudaIntTensor, THIntTensor,
                      MPIDataType::HOROVOD_INT32)
ALLREDUCE_CUDA_ON_CPU(torch_cuda_LongTensor, THCudaLongTensor, THLongTensor,
                      MPIDataType::HOROVOD_INT64)
ALLREDUCE_CUDA_ON_CPU(torch_cuda_FloatTensor, THCudaTensor, THFloatTensor,
                      MPIDataType::HOROVOD_FLOAT32)
ALLREDUCE_CUDA_ON_CPU(torch_cuda_DoubleTensor, THCudaDoubleTensor,
                      THDoubleTensor, MPIDataType::HOROVOD_FLOAT64)
#endif

#define ALLGATHER(torch_Tensor, THTensor, HorovodType)                         \
  extern "C" int horovod_torch_allgather_async_##torch_Tensor(                 \
      THTensor* tensor, THTensor* output, char* name) {                        \
    return DoAllgather<THTensor, HorovodType>(tensor, output, name);           \
  }

ALLGATHER(torch_ByteTensor, THByteTensor, MPIDataType::HOROVOD_UINT8)
ALLGATHER(torch_CharTensor, THCharTensor, MPIDataType::HOROVOD_INT8)
ALLGATHER(torch_ShortTensor, THShortTensor, MPIDataType::HOROVOD_INT16)
ALLGATHER(torch_IntTensor, THIntTensor, MPIDataType::HOROVOD_INT32)
ALLGATHER(torch_LongTensor, THLongTensor, MPIDataType::HOROVOD_INT64)
ALLGATHER(torch_FloatTensor, THFloatTensor, MPIDataType::HOROVOD_FLOAT32)
ALLGATHER(torch_DoubleTensor, THDoubleTensor, MPIDataType::HOROVOD_FLOAT64)

#if HOROVOD_GPU_ALLGATHER
ALLGATHER(torch_cuda_ByteTensor, THCudaByteTensor, MPIDataType::HOROVOD_UINT8)
ALLGATHER(torch_cuda_CharTensor, THCudaCharTensor, MPIDataType::HOROVOD_INT8)
ALLGATHER(torch_cuda_ShortTensor, THCudaShortTensor, MPIDataType::HOROVOD_INT16)
ALLGATHER(torch_cuda_IntTensor, THCudaIntTensor, MPIDataType::HOROVOD_INT32)
ALLGATHER(torch_cuda_LongTensor, THCudaLongTensor, MPIDataType::HOROVOD_INT64)
ALLGATHER(torch_cuda_FloatTensor, THCudaTensor, MPIDataType::HOROVOD_FLOAT32)
ALLGATHER(torch_cuda_DoubleTensor, THCudaDoubleTensor,
          MPIDataType::HOROVOD_FLOAT64)
#endif

#define ALLGATHER_CUDA_ON_CPU(torch_Tensor, THCTensor, THTensor, HorovodType)  \
  extern "C" int horovod_torch_allgather_async_##torch_Tensor(                 \
      THCTensor* tensor, THCTensor* output, char* name) {                      \
    return DoAllgatherCudaOnCPU<THCTensor, THTensor, HorovodType>(             \
        tensor, output, name);                                                 \
  }

#if !HOROVOD_GPU_ALLGATHER && HAVE_CUDA
ALLGATHER_CUDA_ON_CPU(torch_cuda_ByteTensor, THCudaByteTensor, THByteTensor,
                      MPIDataType::HOROVOD_UINT8)
ALLGATHER_CUDA_ON_CPU(torch_cuda_CharTensor, THCudaCharTensor, THCharTensor,
                      MPIDataType::HOROVOD_INT8)
ALLGATHER_CUDA_ON_CPU(torch_cuda_ShortTensor, THCudaShortTensor, THShortTensor,
                      MPIDataType::HOROVOD_INT16)
ALLGATHER_CUDA_ON_CPU(torch_cuda_IntTensor, THCudaIntTensor, THIntTensor,
                      MPIDataType::HOROVOD_INT32)
ALLGATHER_CUDA_ON_CPU(torch_cuda_LongTensor, THCudaLongTensor, THLongTensor,
                      MPIDataType::HOROVOD_INT64)
ALLGATHER_CUDA_ON_CPU(torch_cuda_FloatTensor, THCudaTensor, THFloatTensor,
                      MPIDataType::HOROVOD_FLOAT32)
ALLGATHER_CUDA_ON_CPU(torch_cuda_DoubleTensor, THCudaDoubleTensor,
                      THDoubleTensor, MPIDataType::HOROVOD_FLOAT64)
#endif

#define BROADCAST(torch_Tensor, THTensor, HorovodType)                         \
  extern "C" int horovod_torch_broadcast_async_##torch_Tensor(                 \
      THTensor* tensor, THTensor* output, int root_rank, char* name) {         \
    return DoBroadcast<THTensor, HorovodType>(tensor, output, root_rank,       \
                                              name);                           \
  }

BROADCAST(torch_ByteTensor, THByteTensor, MPIDataType::HOROVOD_UINT8)
BROADCAST(torch_CharTensor, THCharTensor, MPIDataType::HOROVOD_INT8)
BROADCAST(torch_ShortTensor, THShortTensor, MPIDataType::HOROVOD_INT16)
BROADCAST(torch_IntTensor, THIntTensor, MPIDataType::HOROVOD_INT32)
BROADCAST(torch_LongTensor, THLongTensor, MPIDataType::HOROVOD_INT64)
BROADCAST(torch_FloatTensor, THFloatTensor, MPIDataType::HOROVOD_FLOAT32)
BROADCAST(torch_DoubleTensor, THDoubleTensor, MPIDataType::HOROVOD_FLOAT64)

#if HOROVOD_GPU_BROADCAST
BROADCAST(torch_cuda_ByteTensor, THCudaByteTensor, MPIDataType::HOROVOD_UINT8)
BROADCAST(torch_cuda_CharTensor, THCudaCharTensor, MPIDataType::HOROVOD_INT8)
BROADCAST(torch_cuda_ShortTensor, THCudaShortTensor, MPIDataType::HOROVOD_INT16)
BROADCAST(torch_cuda_IntTensor, THCudaIntTensor, MPIDataType::HOROVOD_INT32)
BROADCAST(torch_cuda_LongTensor, THCudaLongTensor, MPIDataType::HOROVOD_INT64)
BROADCAST(torch_cuda_FloatTensor, THCudaTensor, MPIDataType::HOROVOD_FLOAT32)
BROADCAST(torch_cuda_DoubleTensor, THCudaDoubleTensor,
          MPIDataType::HOROVOD_FLOAT64)
#endif

#define BROADCAST_CUDA_ON_CPU(torch_Tensor, THCTensor, THTensor, HorovodType)  \
  extern "C" int horovod_torch_broadcast_async_##torch_Tensor(                 \
      THCTensor* tensor, THCTensor* output, int root_rank, char* name) {       \
    return DoBroadcastCudaOnCPU<THCTensor, THTensor, HorovodType>(             \
        tensor, output, root_rank, name);                                      \
  }

#if !HOROVOD_GPU_BROADCAST && HAVE_CUDA
BROADCAST_CUDA_ON_CPU(torch_cuda_ByteTensor, THCudaByteTensor, THByteTensor,
                      MPIDataType::HOROVOD_UINT8)
BROADCAST_CUDA_ON_CPU(torch_cuda_CharTensor, THCudaCharTensor, THCharTensor,
                      MPIDataType::HOROVOD_INT8)
BROADCAST_CUDA_ON_CPU(torch_cuda_ShortTensor, THCudaShortTensor, THShortTensor,
                      MPIDataType::HOROVOD_INT16)
BROADCAST_CUDA_ON_CPU(torch_cuda_IntTensor, THCudaIntTensor, THIntTensor,
                      MPIDataType::HOROVOD_INT32)
BROADCAST_CUDA_ON_CPU(torch_cuda_LongTensor, THCudaLongTensor, THLongTensor,
                      MPIDataType::HOROVOD_INT64)
BROADCAST_CUDA_ON_CPU(torch_cuda_FloatTensor, THCudaTensor, THFloatTensor,
                      MPIDataType::HOROVOD_FLOAT32)
BROADCAST_CUDA_ON_CPU(torch_cuda_DoubleTensor, THCudaDoubleTensor,
                      THDoubleTensor, MPIDataType::HOROVOD_FLOAT64)
#endif

extern "C" int horovod_torch_poll(int handle) {
  return handle_manager.PollHandle(handle) ? 1 : 0;
}

extern "C" void horovod_torch_wait_and_clear(int handle) {
  while (!handle_manager.PollHandle(handle)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  auto status = handle_manager.ReleaseHandle(handle);
  ThrowIfError(*status);
}

} // namespace torch
} // namespace horovod