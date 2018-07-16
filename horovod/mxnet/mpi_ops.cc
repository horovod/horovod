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
namespace mxnet {

static HandleManager handle_manager;

namespace {

std::string GetOpName(std::string prefix, char* name, int handle) {
  if (name != nullptr) {
    return prefix + "." + std::string(name);
  }
  return prefix + ".noname." + std::to_string(handle);
}

} // namespace

int DoAllreduce(NDArray* tensor, NDArray* output, int average, char* name) {
  ThrowIfError(common::CheckInitialized());

  auto handle = handle_manager.AllocateHandle();
  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<MXTensor<NDArray>>(tensor);
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(device, output);
  auto hvd_output = std::make_shared<MXTensor<NDArray>>(output);

  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_tensor, hvd_output, nullptr,
      GetOpName("allreduce", name, handle), device,
      [handle, average, output](const Status& status) {
        if (average) {
          TensorUtil::DivideTensorInPlace(output, horovod_size());
        }
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}

#if HAVE_CUDA
int DoAllreduceCudaOnCPU(NDArray* tensor, NDArray* output, int average, 
                         char* name) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_cpu_buffer =
      std::make_shared<MXTemporaryBuffer<NDArray>>(CPU_DEVICE_ID);
  TensorUtil::AsyncCopyCudaToCPU(tensor, hvd_cpu_buffer->tensor());
  auto ready_event = std::make_shared<MXReadyEvent<NDArray>>(device);

  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(
      CPU_DEVICE_ID, hvd_cpu_buffer->tensor());

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, ready_event,
      GetOpName("allreduce", name, handle), CPU_DEVICE_ID,
      [handle, average, hvd_cpu_buffer, output](const Status& status) {
        TensorUtil::CopyCPUToCuda(hvd_cpu_buffer->tensor(), output);
        if (average) {
          TensorUtil::DivideTensorInPlace(output, horovod_size());
        }
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}
#endif

int DoAllgather(NDArray* tensor, NDArray* output, char* name) {
  ThrowIfError(common::CheckInitialized());

  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<MXTensor<NDArray>>(tensor);
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(device, output);

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorAllgather(
      hvd_context, hvd_tensor, nullptr, GetOpName("allgather", name, handle),
      device, [handle](const Status& status) {
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}

#if HAVE_CUDA
int DoAllgatherCudaOnCPU(NDArray* tensor, NDArray* output, char* name) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_cpu_tensor =
      std::make_shared<MXTemporaryBuffer<NDArray>>(CPU_DEVICE_ID);
  TensorUtil::AsyncCopyCudaToCPU(tensor, hvd_cpu_tensor->tensor());
  auto ready_event = std::make_shared<MXReadyEvent<NDArray>>(device);

  auto hvd_cpu_output =
      std::make_shared<MXTemporaryBuffer<NDArray>>(CPU_DEVICE_ID);
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(
      CPU_DEVICE_ID, hvd_cpu_output->tensor());

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorAllgather(
      hvd_context, hvd_cpu_tensor, ready_event,
      GetOpName("allgather", name, handle), CPU_DEVICE_ID,
      [handle, hvd_cpu_output, output](const Status& status) {
        TensorUtil::CopyCPUToCuda(hvd_cpu_output->tensor(), output);
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}
#endif

int DoBroadcast(NDArray* tensor, NDArray* output, int root_rank, char* name) {
  ThrowIfError(common::CheckInitialized());

  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<MXTensor<NDArray>>(tensor);
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(device, output);
  std::shared_ptr<Tensor> hvd_output = nullptr;
  if (horovod_rank() == root_rank) {
    if (tensor != output) {
      TensorUtil::Copy(output, tensor);
    }
  } else {
    hvd_output = std::make_shared<MXTensor<NDArray>>(output);
  }

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result =
      EnqueueTensorBroadcast(hvd_context, hvd_tensor, hvd_output, root_rank,
                             nullptr, GetOpName("broadcast", name, handle),
                             device, [handle](const Status& status) {
                               handle_manager.MarkDone(handle, status);
                             });
  ThrowIfError(enqueue_result);

  return handle;
}

#if HAVE_CUDA
int DoBroadcastCudaOnCPU(NDArray* tensor, NDArray* output, int root_rank, 
                         char* name) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_cpu_buffer =
      std::make_shared<MXTemporaryBuffer<NDArray>>(CPU_DEVICE_ID);
  TensorUtil::AsyncCopyCudaToCPU(tensor, hvd_cpu_buffer->tensor());
  auto ready_event = std::make_shared<MXReadyEvent<NDArray>>(device);

  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(
      CPU_DEVICE_ID, hvd_cpu_buffer->tensor());

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorBroadcast(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, root_rank, ready_event,
      GetOpName("broadcast", name, handle), CPU_DEVICE_ID,
      [handle, hvd_cpu_buffer, output](const Status& status) {
        TensorUtil::CopyCPUToCuda(hvd_cpu_buffer->tensor(), output);
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}
#endif

// Do AllReduce on GPU only if src and dst are on GPU
// Otherwise do AllReduce on CPU
extern "C" int horovod_mxnet_allreduce_async(
    NDArray* tensor, NDArray* output, int average, char* name) {
  if (tensor->ctx().dev_mask() == gpu::kDevMask &&
      output->ctx().dev_mask() == gpu::kDevMask)
    return DoAllreduce(tensor, output, average, name);
  else
    return DoAllreduceCudaOnCPU(tensor, output, average, name);
}

extern "C" int horovod_mxnet_allgather_async(
    NDArray* tensor, NDArray* output, char* name) {
  if (tensor->ctx().dev_mask() == gpu::kDevMask &&
      output->ctx().dev_mask() == gpu::kDevMask)
    return DoAllgather(tensor, output, name);
  else
    return DoAllgatherCudaOnCPU(tensor, output, name);
}

extern "C" int horovod_mxnet_broadcast_async(
    NDArray* tensor, NDArray* output, int root_rank, char* name) {
  if (tensor->ctx().dev_mask() == gpu::kDevMask &&
      output->ctx().dev_mask() == gpu::kDevMask)
    return DoBroadcast(tensor, output, root_rank, name);
  else
    return DoBroadcastCudaOnCPU(tensor, output, root_rank, name);
}

extern "C" int horovod_mxnet_poll(int handle) {
  return handle_manager.PollHandle(handle) ? 1 : 0;
}

extern "C" void horovod_mxnet_wait_and_clear(int handle) {
  while (!handle_manager.PollHandle(handle)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  auto status = handle_manager.ReleaseHandle(handle);
  ThrowIfError(*status);
}

} // namespace mxnet
} // namespace horovod
