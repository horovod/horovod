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
#include <torch/extension.h>

#include "../common/operations.h"
#include "adapter_v2.h"
#include "cuda_util.h"
#include "handle_manager.h"
#include "ready_event.h"

namespace horovod {
namespace torch {

static HandleManager handle_manager;

namespace {

std::string GetOpName(const std::string& prefix, const std::string& name,
                      int handle) {
  if (!name.empty()) {
    return prefix + "." + std::string(name);
  }
  return prefix + ".noname." + std::to_string(handle);
}

int GetDeviceID(const at::Tensor& tensor) {
  if (tensor.device().is_cuda()) {
    return tensor.device().index();
  }
  return CPU_DEVICE_ID;
}

} // namespace

int DoAllreduce(at::Tensor tensor, at::Tensor output, int average,
                const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  auto handle = handle_manager.AllocateHandle();
  auto device = GetDeviceID(tensor);
  auto ready_event = RecordReadyEvent(device);
  auto hvd_tensor = std::make_shared<TorchTensor>(tensor);
  auto hvd_context = std::make_shared<TorchOpContext>(device, output);
  auto hvd_output = std::make_shared<TorchTensor>(output);

  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_tensor, hvd_output, ready_event,
      GetOpName("allreduce", name, handle), device,
      [handle, average, output](const Status& status) mutable {
        // Will execute in the `device` context.
        if (average) {
          output.div_(horovod_size());
        }
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}

int DoAllreduceCudaOnCPU(at::Tensor tensor, at::Tensor output, int average,
                         const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto device = GetDeviceID(tensor);
  auto cpu_buffer =
      tensor.to(at::Device(at::DeviceType::CPU), /*non_blocking=*/true);
  auto hvd_cpu_buffer = std::make_shared<TorchTensor>(cpu_buffer);
  auto ready_event = RecordReadyEvent(device);

  auto hvd_context =
      std::make_shared<TorchOpContext>(CPU_DEVICE_ID, cpu_buffer);

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, ready_event,
      GetOpName("allreduce", name, handle), CPU_DEVICE_ID,
      [handle, average, cpu_buffer, output,
       device](const Status& status) mutable {
        // Since the operation was on CPU, need to perform copy with the GPU
        // device guard.
        with_device device_guard(device);
        output.copy_(cpu_buffer);
        if (average) {
          output.div_(horovod_size());
        }
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}

int DoAllgather(at::Tensor tensor, at::Tensor output, const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  auto device = GetDeviceID(tensor);
  auto ready_event = RecordReadyEvent(device);
  auto hvd_tensor = std::make_shared<TorchTensor>(tensor);
  auto hvd_context = std::make_shared<TorchOpContext>(device, output);

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

int DoAllgatherCudaOnCPU(at::Tensor tensor, at::Tensor output,
                         const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto device = GetDeviceID(tensor);
  auto cpu_tensor =
      tensor.to(at::Device(at::DeviceType::CPU), /*non_blocking=*/true);
  auto hvd_cpu_tensor = std::make_shared<TorchTensor>(cpu_tensor);
  auto ready_event = RecordReadyEvent(device);

  auto cpu_output =
      at::tensor(cpu_tensor.type()).to(at::Device(at::DeviceType::CPU));
  auto hvd_cpu_output = std::make_shared<TorchTensor>(cpu_output);
  auto hvd_context =
      std::make_shared<TorchOpContext>(CPU_DEVICE_ID, cpu_output);

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorAllgather(
      hvd_context, hvd_cpu_tensor, ready_event,
      GetOpName("allgather", name, handle), CPU_DEVICE_ID,
      [handle, cpu_output, output, device](const Status& status) mutable {
        // Since the operation was on CPU, need to perform copy with the GPU
        // device guard.
        with_device device_guard(device);
        // output needs to be resized before copying in the CPU tensor.
        output.resize_(cpu_output.sizes());
        output.copy_(cpu_output);
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}

int DoBroadcast(at::Tensor tensor, at::Tensor output, int root_rank,
                const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  auto device = GetDeviceID(tensor);
  auto ready_event = RecordReadyEvent(device);
  auto hvd_tensor = std::make_shared<TorchTensor>(tensor);
  auto hvd_context = std::make_shared<TorchOpContext>(device, output);
  std::shared_ptr<Tensor> hvd_output = nullptr;
  if (horovod_rank() == root_rank) {
    if (tensor.data_ptr() != output.data_ptr()) {
      with_device device_guard(device);
      output.copy_(tensor);
    }
  } else {
    hvd_output = std::make_shared<TorchTensor>(output);
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

int DoBroadcastCudaOnCPU(at::Tensor tensor, at::Tensor output, int root_rank,
                         const std::string& name) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto device = GetDeviceID(tensor);
  auto cpu_buffer =
      tensor.to(at::Device(at::DeviceType::CPU), /*non_blocking=*/true);
  auto hvd_cpu_buffer = std::make_shared<TorchTensor>(cpu_buffer);
  auto ready_event = RecordReadyEvent(device);

  auto hvd_context =
      std::make_shared<TorchOpContext>(CPU_DEVICE_ID, cpu_buffer);

  auto handle = handle_manager.AllocateHandle();
  auto enqueue_result = EnqueueTensorBroadcast(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, root_rank, ready_event,
      GetOpName("broadcast", name, handle), CPU_DEVICE_ID,
      [handle, cpu_buffer, output, device](const Status& status) mutable {
        // Since the operation was on CPU, need to perform copy with the GPU
        // device guard.
        with_device device_guard(device);
        output.copy_(cpu_buffer);
        handle_manager.MarkDone(handle, status);
      });
  ThrowIfError(enqueue_result);

  return handle;
}

int PollHandle(int handle) { return handle_manager.PollHandle(handle) ? 1 : 0; }

void WaitAndClear(int handle) {
  while (!handle_manager.PollHandle(handle)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  auto status = handle_manager.ReleaseHandle(handle);
  ThrowIfError(*status);
}

PYBIND11_MODULE(mpi_lib_v2, m) {
  // allreduce
  m.def("horovod_torch_allreduce_async_torch_IntTensor", &DoAllreduce);
  m.def("horovod_torch_allreduce_async_torch_LongTensor", &DoAllreduce);
  m.def("horovod_torch_allreduce_async_torch_FloatTensor", &DoAllreduce);
  m.def("horovod_torch_allreduce_async_torch_DoubleTensor", &DoAllreduce);
#if HOROVOD_GPU_ALLREDUCE
  m.def("horovod_torch_allreduce_async_torch_cuda_IntTensor", &DoAllreduce);
  m.def("horovod_torch_allreduce_async_torch_cuda_LongTensor", &DoAllreduce);
  m.def("horovod_torch_allreduce_async_torch_cuda_FloatTensor", &DoAllreduce);
  m.def("horovod_torch_allreduce_async_torch_cuda_DoubleTensor", &DoAllreduce);
#else
  m.def("horovod_torch_allreduce_async_torch_cuda_IntTensor",
        &DoAllreduceCudaOnCPU);
  m.def("horovod_torch_allreduce_async_torch_cuda_LongTensor",
        &DoAllreduceCudaOnCPU);
  m.def("horovod_torch_allreduce_async_torch_cuda_FloatTensor",
        &DoAllreduceCudaOnCPU);
  m.def("horovod_torch_allreduce_async_torch_cuda_DoubleTensor",
        &DoAllreduceCudaOnCPU);
#endif

  // allgather
  m.def("horovod_torch_allgather_async_torch_ByteTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_CharTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_ShortTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_IntTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_LongTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_FloatTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_DoubleTensor", &DoAllgather);
#if HOROVOD_GPU_ALLGATHER
  m.def("horovod_torch_allgather_async_torch_cuda_ByteTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_cuda_CharTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_cuda_ShortTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_cuda_IntTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_cuda_LongTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_cuda_FloatTensor", &DoAllgather);
  m.def("horovod_torch_allgather_async_torch_cuda_DoubleTensor", &DoAllgather);
#else
  m.def("horovod_torch_allgather_async_torch_cuda_ByteTensor",
        &DoAllgatherCudaOnCPU);
  m.def("horovod_torch_allgather_async_torch_cuda_CharTensor",
        &DoAllgatherCudaOnCPU);
  m.def("horovod_torch_allgather_async_torch_cuda_ShortTensor",
        &DoAllgatherCudaOnCPU);
  m.def("horovod_torch_allgather_async_torch_cuda_IntTensor",
        &DoAllgatherCudaOnCPU);
  m.def("horovod_torch_allgather_async_torch_cuda_LongTensor",
        &DoAllgatherCudaOnCPU);
  m.def("horovod_torch_allgather_async_torch_cuda_FloatTensor",
        &DoAllgatherCudaOnCPU);
  m.def("horovod_torch_allgather_async_torch_cuda_DoubleTensor",
        &DoAllgatherCudaOnCPU);
#endif

  // broadcast
  m.def("horovod_torch_broadcast_async_torch_ByteTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_CharTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_ShortTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_IntTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_LongTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_FloatTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_DoubleTensor", &DoBroadcast);
#if HOROVOD_GPU_BROADCAST
  m.def("horovod_torch_broadcast_async_torch_cuda_ByteTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_cuda_CharTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_cuda_ShortTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_cuda_IntTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_cuda_LongTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_cuda_FloatTensor", &DoBroadcast);
  m.def("horovod_torch_broadcast_async_torch_cuda_DoubleTensor", &DoBroadcast);
#else
  m.def("horovod_torch_broadcast_async_torch_cuda_ByteTensor",
        &DoBroadcastCudaOnCPU);
  m.def("horovod_torch_broadcast_async_torch_cuda_CharTensor",
        &DoBroadcastCudaOnCPU);
  m.def("horovod_torch_broadcast_async_torch_cuda_ShortTensor",
        &DoBroadcastCudaOnCPU);
  m.def("horovod_torch_broadcast_async_torch_cuda_IntTensor",
        &DoBroadcastCudaOnCPU);
  m.def("horovod_torch_broadcast_async_torch_cuda_LongTensor",
        &DoBroadcastCudaOnCPU);
  m.def("horovod_torch_broadcast_async_torch_cuda_FloatTensor",
        &DoBroadcastCudaOnCPU);
  m.def("horovod_torch_broadcast_async_torch_cuda_DoubleTensor",
        &DoBroadcastCudaOnCPU);
#endif

  // basics
  m.def("horovod_torch_poll", &PollHandle);
  m.def("horovod_torch_wait_and_clear", &WaitAndClear);
}

} // namespace torch
} // namespace horovod