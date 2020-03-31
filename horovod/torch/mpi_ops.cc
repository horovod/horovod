// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
// Modifications copyright Microsoft
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

std::string GetOpName(const std::string& prefix, char* name, int handle) {
  if (name != nullptr) {
    return prefix + "." + std::string(name);
  }
  return prefix + ".noname." + std::to_string(handle);
}

} // namespace

template <DataType DT, DeviceType Dev, class T>
int DoAllreduce(T* tensor, T* output, int divisor, char* name, int reduce_op_int) {
  ThrowIfError(common::CheckInitialized());

  auto handle = handle_manager.AllocateHandle();
  auto device = TensorUtil::GetDevice<DT, Dev>(tensor);
  auto ready_event = RecordReadyEvent(device);
  auto hvd_tensor = std::make_shared<TorchTensor<DT, Dev, T>>(tensor);
  auto hvd_context =
      std::make_shared<TorchOpContext<DT, Dev, T>>(device, output);
  auto hvd_output = std::make_shared<TorchTensor<DT, Dev, T>>(output);

  ReduceOp reduce_op = static_cast<ReduceOp>(reduce_op_int);

  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_tensor, hvd_output, ready_event,
      GetOpName("allreduce", name, handle), device,
      [handle, divisor, output](const Status& status) {
        if (divisor > 1) {
          TensorUtil::DivideTensorInPlace<DT, Dev, T>(output, divisor);
        }
        handle_manager.MarkDone(handle, status);
      }, reduce_op);
  ThrowIfError(enqueue_result);

  return handle;
}

#if HAVE_GPU
template <DataType DT, class TC, class T>
int DoAllreduceCudaOnCPU(TC* tensor, TC* output, int divisor, char* name, int reduce_op_int) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto device = TensorUtil::GetDevice<DT, DeviceType::GPU>(tensor);
  auto hvd_cpu_buffer =
      std::make_shared<TorchTemporaryBuffer<DT, DeviceType::CPU, T>>(
          CPU_DEVICE_ID);
  TensorUtil::AsyncCopyCudaToCPU<DT>(tensor, hvd_cpu_buffer->tensor());
  auto ready_event = RecordReadyEvent(device);

  auto hvd_context = std::make_shared<TorchOpContext<DT, DeviceType::CPU, T>>(
      CPU_DEVICE_ID, hvd_cpu_buffer->tensor());

  ReduceOp reduce_op = static_cast<ReduceOp>(reduce_op_int);
  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, ready_event,
      GetOpName("allreduce", name, handle), CPU_DEVICE_ID,
      [handle, divisor, hvd_cpu_buffer, output](const Status& status) {
        TensorUtil::CopyCPUToCuda<DT>(hvd_cpu_buffer->tensor(), output);
        if (divisor > 1) {
          TensorUtil::DivideTensorInPlace<DT, DeviceType::GPU>(output,
                                                               divisor);
        }
        handle_manager.MarkDone(handle, status);
      }, reduce_op);
  ThrowIfError(enqueue_result);

  return handle;
}
#endif

template <DataType DT, DeviceType Dev, class T>
int DoAllgather(T* tensor, T* output, char* name) {
  ThrowIfError(common::CheckInitialized());

  auto device = TensorUtil::GetDevice<DT, Dev>(tensor);
  auto ready_event = RecordReadyEvent(device);
  auto hvd_tensor = std::make_shared<TorchTensor<DT, Dev, T>>(tensor);
  auto hvd_context =
      std::make_shared<TorchOpContext<DT, Dev, T>>(device, output);

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

#if HAVE_GPU
template <DataType DT, class TC, class T>
int DoAllgatherCudaOnCPU(TC* tensor, TC* output, char* name) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto device = TensorUtil::GetDevice<DT, DeviceType::GPU>(tensor);
  auto hvd_cpu_tensor =
      std::make_shared<TorchTemporaryBuffer<DT, DeviceType::CPU, T>>(
          CPU_DEVICE_ID);
  TensorUtil::AsyncCopyCudaToCPU<DT>(tensor, hvd_cpu_tensor->tensor());
  auto ready_event = RecordReadyEvent(device);

  auto hvd_cpu_output =
      std::make_shared<TorchTemporaryBuffer<DT, DeviceType::CPU, T>>(
          CPU_DEVICE_ID);
  auto hvd_context = std::make_shared<TorchOpContext<DT, DeviceType::CPU, T>>(
      CPU_DEVICE_ID, hvd_cpu_output->tensor());

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorAllgather(
      hvd_context, hvd_cpu_tensor, ready_event,
      GetOpName("allgather", name, handle), CPU_DEVICE_ID,
      [handle, hvd_cpu_output, output](const Status& status) {
        TensorUtil::CopyCPUToCuda<DT>(hvd_cpu_output->tensor(), output);
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}
#endif

template <DataType DT, DeviceType Dev, class T>
int DoBroadcast(T* tensor, T* output, int root_rank, char* name) {
  ThrowIfError(common::CheckInitialized());

  auto device = TensorUtil::GetDevice<DT, Dev>(tensor);
  auto ready_event = RecordReadyEvent(device);
  auto hvd_tensor = std::make_shared<TorchTensor<DT, Dev, T>>(tensor);
  auto hvd_context =
      std::make_shared<TorchOpContext<DT, Dev, T>>(device, output);
  std::shared_ptr<Tensor> hvd_output = nullptr;
  if (horovod_rank() == root_rank) {
    if (tensor != output) {
      TensorUtil::Copy<DT, Dev>(output, tensor);
    }
  } else {
    hvd_output = std::make_shared<TorchTensor<DT, Dev, T>>(output);
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

#if HAVE_GPU
template <DataType DT, class TC, class T>
int DoBroadcastCudaOnCPU(TC* tensor, TC* output, int root_rank, char* name) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto device = TensorUtil::GetDevice<DT, DeviceType::GPU>(tensor);
  auto hvd_cpu_buffer =
      std::make_shared<TorchTemporaryBuffer<DT, DeviceType::CPU, T>>(
          CPU_DEVICE_ID);
  TensorUtil::AsyncCopyCudaToCPU<DT>(tensor, hvd_cpu_buffer->tensor());
  auto ready_event = RecordReadyEvent(device);

  auto hvd_context = std::make_shared<TorchOpContext<DT, DeviceType::CPU, T>>(
      CPU_DEVICE_ID, hvd_cpu_buffer->tensor());

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorBroadcast(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, root_rank, ready_event,
      GetOpName("broadcast", name, handle), CPU_DEVICE_ID,
      [handle, hvd_cpu_buffer, output](const Status& status) {
        TensorUtil::CopyCPUToCuda<DT>(hvd_cpu_buffer->tensor(), output);
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}
#endif

int DoJoin(int device) {
  throw std::runtime_error("Join Op is not supported for PyTorch < 1.0");
}

#define ALLREDUCE(torch_Tensor, HorovodType, DeviceType, THTensor)                    \
  extern "C" int horovod_torch_allreduce_async_##torch_Tensor(                        \
      THTensor* tensor, THTensor* output, int divisor, char* name, int reduce_op) {   \
    return DoAllreduce<HorovodType, DeviceType>(tensor, output, divisor,              \
                                                name, reduce_op);                     \
  }

ALLREDUCE(torch_IntTensor, DataType::HOROVOD_INT32, DeviceType::CPU,
          THIntTensor)
ALLREDUCE(torch_LongTensor, DataType::HOROVOD_INT64, DeviceType::CPU,
          THLongTensor)
ALLREDUCE(torch_FloatTensor, DataType::HOROVOD_FLOAT32, DeviceType::CPU,
          THFloatTensor)
ALLREDUCE(torch_DoubleTensor, DataType::HOROVOD_FLOAT64, DeviceType::CPU,
          THDoubleTensor)

#if HOROVOD_GPU_ALLREDUCE
ALLREDUCE(torch_cuda_IntTensor, DataType::HOROVOD_INT32, DeviceType::GPU,
          THCudaIntTensor)
ALLREDUCE(torch_cuda_LongTensor, DataType::HOROVOD_INT64, DeviceType::GPU,
          THCudaLongTensor)
ALLREDUCE(torch_cuda_FloatTensor, DataType::HOROVOD_FLOAT32, DeviceType::GPU,
          THCudaTensor)
ALLREDUCE(torch_cuda_DoubleTensor, DataType::HOROVOD_FLOAT64,
          DeviceType::GPU, THCudaDoubleTensor)
#endif

#define ALLREDUCE_CUDA_ON_CPU(torch_Tensor, HorovodType, THCTensor, THTensor)         \
  extern "C" int horovod_torch_allreduce_async_##torch_Tensor(                        \
      THCTensor* tensor, THCTensor* output, int divisor, char* name, int reduce_op) { \
    return DoAllreduceCudaOnCPU<HorovodType, THCTensor, THTensor>(                    \
        tensor, output, divisor, name, reduce_op);                                    \
  }

#if !HOROVOD_GPU_ALLREDUCE && HAVE_GPU
ALLREDUCE_CUDA_ON_CPU(torch_cuda_IntTensor, DataType::HOROVOD_INT32,
                      THCudaIntTensor, THIntTensor)
ALLREDUCE_CUDA_ON_CPU(torch_cuda_LongTensor, DataType::HOROVOD_INT64,
                      THCudaLongTensor, THLongTensor)
ALLREDUCE_CUDA_ON_CPU(torch_cuda_FloatTensor, DataType::HOROVOD_FLOAT32,
                      THCudaTensor, THFloatTensor)
ALLREDUCE_CUDA_ON_CPU(torch_cuda_DoubleTensor, DataType::HOROVOD_FLOAT64,
                      THCudaDoubleTensor, THDoubleTensor)
#endif

#define ALLGATHER(torch_Tensor, HorovodType, DeviceType, THTensor)             \
  extern "C" int horovod_torch_allgather_async_##torch_Tensor(                 \
      THTensor* tensor, THTensor* output, char* name) {                        \
    return DoAllgather<HorovodType, DeviceType>(tensor, output, name);         \
  }

ALLGATHER(torch_ByteTensor, DataType::HOROVOD_UINT8, DeviceType::CPU,
          THByteTensor)
ALLGATHER(torch_CharTensor, DataType::HOROVOD_INT8, DeviceType::CPU,
          THCharTensor)
ALLGATHER(torch_ShortTensor, DataType::HOROVOD_INT16, DeviceType::CPU,
          THShortTensor)
ALLGATHER(torch_IntTensor, DataType::HOROVOD_INT32, DeviceType::CPU,
          THIntTensor)
ALLGATHER(torch_LongTensor, DataType::HOROVOD_INT64, DeviceType::CPU,
          THLongTensor)
ALLGATHER(torch_FloatTensor, DataType::HOROVOD_FLOAT32, DeviceType::CPU,
          THFloatTensor)
ALLGATHER(torch_DoubleTensor, DataType::HOROVOD_FLOAT64, DeviceType::CPU,
          THDoubleTensor)

#if HOROVOD_GPU_ALLGATHER
ALLGATHER(torch_cuda_ByteTensor, DataType::HOROVOD_UINT8, DeviceType::GPU,
          THCudaByteTensor)
ALLGATHER(torch_cuda_CharTensor, DataType::HOROVOD_INT8, DeviceType::GPU,
          THCudaCharTensor)
ALLGATHER(torch_cuda_ShortTensor, DataType::HOROVOD_INT16, DeviceType::GPU,
          THCudaShortTensor)
ALLGATHER(torch_cuda_IntTensor, DataType::HOROVOD_INT32, DeviceType::GPU,
          THCudaIntTensor)
ALLGATHER(torch_cuda_LongTensor, DataType::HOROVOD_INT64, DeviceType::GPU,
          THCudaLongTensor)
ALLGATHER(torch_cuda_FloatTensor, DataType::HOROVOD_FLOAT32, DeviceType::GPU,
          THCudaTensor)
ALLGATHER(torch_cuda_DoubleTensor, DataType::HOROVOD_FLOAT64,
          DeviceType::GPU, THCudaDoubleTensor)
#endif

#define ALLGATHER_CUDA_ON_CPU(torch_Tensor, HorovodType, THCTensor, THTensor)  \
  extern "C" int horovod_torch_allgather_async_##torch_Tensor(                 \
      THCTensor* tensor, THCTensor* output, char* name) {                      \
    return DoAllgatherCudaOnCPU<HorovodType, THCTensor, THTensor>(             \
        tensor, output, name);                                                 \
  }

#if !HOROVOD_GPU_ALLGATHER && HAVE_GPU
ALLGATHER_CUDA_ON_CPU(torch_cuda_ByteTensor, DataType::HOROVOD_UINT8,
                      THCudaByteTensor, THByteTensor)
ALLGATHER_CUDA_ON_CPU(torch_cuda_CharTensor, DataType::HOROVOD_INT8,
                      THCudaCharTensor, THCharTensor)
ALLGATHER_CUDA_ON_CPU(torch_cuda_ShortTensor, DataType::HOROVOD_INT16,
                      THCudaShortTensor, THShortTensor)
ALLGATHER_CUDA_ON_CPU(torch_cuda_IntTensor, DataType::HOROVOD_INT32,
                      THCudaIntTensor, THIntTensor)
ALLGATHER_CUDA_ON_CPU(torch_cuda_LongTensor, DataType::HOROVOD_INT64,
                      THCudaLongTensor, THLongTensor)
ALLGATHER_CUDA_ON_CPU(torch_cuda_FloatTensor, DataType::HOROVOD_FLOAT32,
                      THCudaTensor, THFloatTensor)
ALLGATHER_CUDA_ON_CPU(torch_cuda_DoubleTensor, DataType::HOROVOD_FLOAT64,
                      THCudaDoubleTensor, THDoubleTensor)
#endif

#define BROADCAST(torch_Tensor, HorovodType, DeviceType, THTensor)             \
  extern "C" int horovod_torch_broadcast_async_##torch_Tensor(                 \
      THTensor* tensor, THTensor* output, int root_rank, char* name) {         \
    return DoBroadcast<HorovodType, DeviceType>(tensor, output, root_rank,     \
                                                name);                         \
  }

BROADCAST(torch_ByteTensor, DataType::HOROVOD_UINT8, DeviceType::CPU,
          THByteTensor)
BROADCAST(torch_CharTensor, DataType::HOROVOD_INT8, DeviceType::CPU,
          THCharTensor)
BROADCAST(torch_ShortTensor, DataType::HOROVOD_INT16, DeviceType::CPU,
          THShortTensor)
BROADCAST(torch_IntTensor, DataType::HOROVOD_INT32, DeviceType::CPU,
          THIntTensor)
BROADCAST(torch_LongTensor, DataType::HOROVOD_INT64, DeviceType::CPU,
          THLongTensor)
BROADCAST(torch_FloatTensor, DataType::HOROVOD_FLOAT32, DeviceType::CPU,
          THFloatTensor)
BROADCAST(torch_DoubleTensor, DataType::HOROVOD_FLOAT64, DeviceType::CPU,
          THDoubleTensor)

#if HOROVOD_GPU_BROADCAST
BROADCAST(torch_cuda_ByteTensor, DataType::HOROVOD_UINT8, DeviceType::GPU,
          THCudaByteTensor)
BROADCAST(torch_cuda_CharTensor, DataType::HOROVOD_INT8, DeviceType::GPU,
          THCudaCharTensor)
BROADCAST(torch_cuda_ShortTensor, DataType::HOROVOD_INT16, DeviceType::GPU,
          THCudaShortTensor)
BROADCAST(torch_cuda_IntTensor, DataType::HOROVOD_INT32, DeviceType::GPU,
          THCudaIntTensor)
BROADCAST(torch_cuda_LongTensor, DataType::HOROVOD_INT64, DeviceType::GPU,
          THCudaLongTensor)
BROADCAST(torch_cuda_FloatTensor, DataType::HOROVOD_FLOAT32, DeviceType::GPU,
          THCudaTensor)
BROADCAST(torch_cuda_DoubleTensor, DataType::HOROVOD_FLOAT64, DeviceType::GPU,
          THCudaDoubleTensor)
#endif

#define BROADCAST_CUDA_ON_CPU(torch_Tensor, HorovodType, THCTensor, THTensor)  \
  extern "C" int horovod_torch_broadcast_async_##torch_Tensor(                 \
      THCTensor* tensor, THCTensor* output, int root_rank, char* name) {       \
    return DoBroadcastCudaOnCPU<HorovodType, THCTensor, THTensor>(             \
        tensor, output, root_rank, name);                                      \
  }

#if !HOROVOD_GPU_BROADCAST && HAVE_GPU
BROADCAST_CUDA_ON_CPU(torch_cuda_ByteTensor, DataType::HOROVOD_UINT8,
                      THCudaByteTensor, THByteTensor)
BROADCAST_CUDA_ON_CPU(torch_cuda_CharTensor, DataType::HOROVOD_INT8,
                      THCudaCharTensor, THCharTensor)
BROADCAST_CUDA_ON_CPU(torch_cuda_ShortTensor, DataType::HOROVOD_INT16,
                      THCudaShortTensor, THShortTensor)
BROADCAST_CUDA_ON_CPU(torch_cuda_IntTensor, DataType::HOROVOD_INT32,
                      THCudaIntTensor, THIntTensor)
BROADCAST_CUDA_ON_CPU(torch_cuda_LongTensor, DataType::HOROVOD_INT64,
                      THCudaLongTensor, THLongTensor)
BROADCAST_CUDA_ON_CPU(torch_cuda_FloatTensor, DataType::HOROVOD_FLOAT32,
                      THCudaTensor, THFloatTensor)
BROADCAST_CUDA_ON_CPU(torch_cuda_DoubleTensor, DataType::HOROVOD_FLOAT64,
                      THCudaDoubleTensor, THDoubleTensor)
#endif

extern "C" int horovod_torch_join(int device) {
  return DoJoin(device);
}

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