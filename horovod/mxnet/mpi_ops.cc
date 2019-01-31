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

#include <atomic>

#include "../common/operations.h"
#include "adapter.h"
#include "cuda_util.h"
#include "mpi_ops.h"
#include "ready_event.h"
#include "tensor_util.h"

namespace horovod {
namespace mxnet {

namespace {

std::atomic_int op_count;

std::string GetOpName(std::string prefix, char* name) {
  if (name != nullptr) {
    return prefix + "." + std::string(name);
  }

  op_count.fetch_add(1);
  return prefix + ".noname." + std::to_string(op_count);
}
} // namespace

inline void InvokeCompleteCallback(Callback on_complete, const Status& status) {
  if (status.ok()) {
    on_complete();
  } else {
    auto error = dmlc::Error(status.reason());
    on_complete(&error);
  }
}

void DoAllreduce(NDArray* tensor, NDArray* output, const std::string& name,
                 Callback on_complete) {
  ThrowIfError(common::CheckInitialized());

  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<MXTensor<NDArray>>(tensor);
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(device, output);
  auto hvd_output = std::make_shared<MXTensor<NDArray>>(output);

  auto enqueue_result =
      EnqueueTensorAllreduce(hvd_context, hvd_tensor, hvd_output, nullptr,
                             name, device,
                             [on_complete](const Status& status) {
                               InvokeCompleteCallback(on_complete, status);
                             });
  ThrowIfError(enqueue_result);
}

#if HAVE_CUDA
void DoAllreduceCudaOnCPU(NDArray* tensor, NDArray* output, std::string& name,
                          Callback on_complete) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto hvd_cpu_buffer = std::make_shared<MXTemporaryBuffer<NDArray>>(
      CPU_DEVICE_ID, tensor->dtype());
  TensorUtil::AsyncCopyCudaToCPU(tensor, hvd_cpu_buffer->tensor());
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(
      CPU_DEVICE_ID, hvd_cpu_buffer->tensor());
  auto ready_event = std::make_shared<MXReadyEvent<NDArray>>(tensor);

  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, ready_event,
      name, CPU_DEVICE_ID,
      [hvd_cpu_buffer, output, on_complete](const Status& status) {
        TensorUtil::CopyCPUToCuda(hvd_cpu_buffer->tensor(), output);
        InvokeCompleteCallback(on_complete, status);
      });
  ThrowIfError(enqueue_result);
}
#endif

void DoAllgather(NDArray* tensor, NDArray* output, std::string& name,
                 Callback on_complete) {
  ThrowIfError(common::CheckInitialized());

  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<MXTensor<NDArray>>(tensor);
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(device, output);

  auto enqueue_result =
      EnqueueTensorAllgather(hvd_context, hvd_tensor, nullptr,
                             name, device,
                             [on_complete](const Status& status) {
                               InvokeCompleteCallback(on_complete, status);
                             });
  ThrowIfError(enqueue_result);
}

#if HAVE_CUDA
void DoAllgatherCudaOnCPU(NDArray* tensor, NDArray* output, std::string& name,
                          Callback on_complete) {
  ThrowIfError(common::CheckInitialized());

  // Make async copy of input tensor to CPU tensor and record completion event.
  auto hvd_cpu_tensor = std::make_shared<MXTemporaryBuffer<NDArray>>(
      CPU_DEVICE_ID, tensor->dtype());
  TensorUtil::AsyncCopyCudaToCPU(tensor, hvd_cpu_tensor->tensor());
  auto ready_event = std::make_shared<MXReadyEvent<NDArray>>(tensor);

  auto hvd_cpu_output = std::make_shared<MXTemporaryBuffer<NDArray>>(
      CPU_DEVICE_ID, output->dtype());
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(
      CPU_DEVICE_ID, hvd_cpu_output->tensor());

  auto enqueue_result = EnqueueTensorAllgather(
      hvd_context, hvd_cpu_tensor, ready_event,
      name, CPU_DEVICE_ID,
      [hvd_cpu_output, output, on_complete](const Status& status) {
        TensorUtil::CopyCPUToCuda(hvd_cpu_output->tensor(), output);
        InvokeCompleteCallback(on_complete, status);
      });
  ThrowIfError(enqueue_result);
}
#endif

void DoBroadcast(NDArray* tensor, NDArray* output, int root_rank,
                 std::string& name, Callback on_complete) {
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

  auto enqueue_result = EnqueueTensorBroadcast(
      hvd_context, hvd_tensor, hvd_output, root_rank, nullptr,
      name, device,
      [on_complete](const Status& status) {
        InvokeCompleteCallback(on_complete, status);
      });
  ThrowIfError(enqueue_result);
}

#if HAVE_CUDA
void DoBroadcastCudaOnCPU(
    std::shared_ptr<MXTemporaryBuffer<NDArray>>& hvd_cpu_buffer, int root_rank,
    std::string& name, Callback on_complete) {
  // Make async copy of input tensor to CPU tensor and record completion event.
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(
      CPU_DEVICE_ID, hvd_cpu_buffer->tensor());
  auto ready_event =
      std::make_shared<MXReadyEvent<NDArray>>(hvd_cpu_buffer->tensor());

  auto enqueue_result = EnqueueTensorBroadcast(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, root_rank, ready_event,
      name, CPU_DEVICE_ID,
      [on_complete](const Status& status) {
        InvokeCompleteCallback(on_complete, status);
      });
  ThrowIfError(enqueue_result);
}
#endif

extern "C" int horovod_mxnet_allreduce_async(NDArray* input, NDArray* output,
                                             char* name, bool average) {
  MX_API_BEGIN();

  std::string op_name = GetOpName("allreduce", name);
  auto allreduce_async_fn = [input, output,
                             op_name](RunContext rctx,
                                      Callback on_complete) mutable {
    DoAllreduce(input, output, op_name, on_complete);
  };

#if HAVE_CUDA
  auto allreduce_async_cpu_fn =
      [input, output, op_name](RunContext rctx,
                               Callback on_complete) mutable {
        DoAllreduceCudaOnCPU(input, output, op_name, on_complete);
      };
#endif

#if HAVE_CUDA && !HOROVOD_GPU_ALLREDUCE
  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(allreduce_async_cpu_fn, input->ctx(),
                             {input->var()}, {output->var()},
                             FnProperty::kNormal, 0, "HorovodAllreduce");
  // In-place
  } else {
    Engine::Get()->PushAsync(allreduce_async_cpu_fn, input->ctx(), {},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodAllreduce");
  }
#else
  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(allreduce_async_fn, input->ctx(), {input->var()},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodAllreduce");
  // In-place
  } else {
    Engine::Get()->PushAsync(allreduce_async_fn, input->ctx(), {},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodAllreduce");
  }
#endif

  if (average) {
    *output /= horovod_size();
  }

  MX_API_END();
}

extern "C" int horovod_mxnet_allgather_async(NDArray* input, NDArray* output,
                                             char* name) {
  MX_API_BEGIN();

  std::string op_name = GetOpName("allgather", name);
  auto allgather_async_fn = [input, output,
                             op_name](RunContext rctx,
                                      Callback on_complete) mutable {
    DoAllgather(input, output, op_name, on_complete);
  };

#if HAVE_CUDA
  auto allgather_async_cpu_fn =
      [input, output, op_name](RunContext rctx,
                               Callback on_complete) mutable {
        DoAllgatherCudaOnCPU(input, output, op_name, on_complete);
      };
#endif

#if HAVE_CUDA && HOROVOD_GPU_ALLGATHER != 'M'
  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(allgather_async_cpu_fn, input->ctx(),
                             {input->var()}, {output->var()},
                             FnProperty::kNormal, 0, "HorovodAllgather");
  // In-place
  } else {
    Engine::Get()->PushAsync(allgather_async_cpu_fn, input->ctx(), {},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodAllgather");
  }
#else
  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(allgather_async_fn, input->ctx(),
                             {input->var()}, {output->var()},
                             FnProperty::kNormal, 0, "HorovodAllgather");
  // In-place
  } else {
    Engine::Get()->PushAsync(allgather_async_fn, input->ctx(), {},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodAllgather");
  }
#endif

  MX_API_END();
}

extern "C" int horovod_mxnet_broadcast_async(NDArray* input, NDArray* output,
                                             int root_rank, char* name) {
  MX_API_BEGIN();

  std::string op_name = GetOpName("broadcast", name);
  auto broadcast_async_fn = [input, output, op_name,
                             root_rank](RunContext rctx,
                                        Callback on_complete) mutable {
    DoBroadcast(input, output, root_rank, op_name, on_complete);
  };

#if HAVE_CUDA && HOROVOD_GPU_BROADCAST != 'M'
  ThrowIfError(common::CheckInitialized());
  // Make async copy of input tensor to CPU tensor and record completion event.
  auto hvd_cpu_buffer = std::make_shared<MXTemporaryBuffer<NDArray>>(
      CPU_DEVICE_ID, input->dtype());
  TensorUtil::AsyncCopyCudaToCPU(input, hvd_cpu_buffer->tensor());
  auto broadcast_async_cpu_fn =
        [hvd_cpu_buffer, op_name, root_rank]
        (RunContext rctx, Callback on_complete) mutable {
          DoBroadcastCudaOnCPU(hvd_cpu_buffer, root_rank, op_name, on_complete);
        };

  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(broadcast_async_cpu_fn, input->ctx(),
                             {input->var()}, {output->var()},
                             FnProperty::kNormal, 0, "HorovodBroadcast");
  // In-place
  } else {
    Engine::Get()->PushAsync(broadcast_async_cpu_fn, input->ctx(), {},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodBroadcast");
  }

  TensorUtil::CopyCPUToCuda(hvd_cpu_buffer->tensor(), output);
#else
  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(broadcast_async_fn, input->ctx(), {input->var()},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodBroadcast");
  // In-place
  } else {
    Engine::Get()->PushAsync(broadcast_async_fn, input->ctx(), {},
                             {output->var()}, FnProperty::kNormal, 0,
                             "HorovodBroadcast");
  }
#endif

  MX_API_END();
}

} // namespace mxnet
} // namespace horovod
