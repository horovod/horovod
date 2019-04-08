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
#include "tensor_util.h"

namespace horovod {
namespace mxnet {

namespace {

std::atomic_int op_count;

std::string GetOpName(const std::string& prefix, const char* name) {
  if (name != nullptr) {
    return prefix + "." + std::string(name);
  }

  op_count.fetch_add(1);
  return prefix + ".noname." + std::to_string(op_count);
}
} // namespace

static const auto MX_EXEC_CTX = Context::CPU();

inline void InvokeCompleteCallback(CallbackOnComplete on_complete, const Status& status) {
  if (status.ok()) {
    on_complete();
  } else {
    auto error = dmlc::Error(status.reason());
    on_complete(&error);
  }
}

void DoAllreduce(NDArray* tensor, NDArray* output, const std::string& name,
                 CallbackOnComplete on_complete) {
  ThrowIfError(common::CheckInitialized());

  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<MXTensor<NDArray>>(tensor);
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(device, output);
  auto hvd_output = std::make_shared<MXTensor<NDArray>>(output);

  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_tensor, hvd_output, nullptr, name, device,
      [on_complete](const Status& status) {
        InvokeCompleteCallback(on_complete, status);
      });
  ThrowIfError(enqueue_result);
}

void DoAllgather(NDArray* tensor, NDArray* output, std::string& name,
                 CallbackOnComplete on_complete) {
  ThrowIfError(common::CheckInitialized());

  auto device = TensorUtil::GetDevice(tensor);
  auto hvd_tensor = std::make_shared<MXTensor<NDArray>>(tensor);
  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(device, output);

  auto enqueue_result = EnqueueTensorAllgather(
      hvd_context, hvd_tensor, nullptr, name, device,
      [on_complete](const Status& status) {
        InvokeCompleteCallback(on_complete, status);
      });
  ThrowIfError(enqueue_result);
}

void DoBroadcast(NDArray* tensor, NDArray* output, std::string& name,
                 int root_rank, CallbackOnComplete on_complete) {
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
      hvd_context, hvd_tensor, hvd_output, root_rank, nullptr, name, device,
      [on_complete](const Status& status) {
        InvokeCompleteCallback(on_complete, status);
      });
  ThrowIfError(enqueue_result);
}

inline void PushHorovodOperation(OperationType op_type, NDArray* input,
                                 NDArray* output, const char* name,
                                 int priority, int root_rank = -1) {
  std::string op_type_name;
  std::string op_name;
  ExecFn exec_fn;
  switch (op_type) {
    case OperationType::ALLREDUCE:
      op_type_name = "horovod_allreduce";
      op_name = GetOpName(op_type_name, name);
      exec_fn = [input, output, op_name]
                (RunContext rctx, CallbackOnComplete on_complete) mutable {
        DoAllreduce(input, output, op_name, on_complete);
      };
      break;
    case OperationType::ALLGATHER:
      op_type_name = "horovod_allgather";
      op_name = GetOpName(op_type_name, name);
      exec_fn = [input, output, op_name]
                (RunContext rctx, CallbackOnComplete on_complete) mutable {
        DoAllgather(input, output, op_name, on_complete);
      };
      break;
    case OperationType::BROADCAST:
      op_type_name = "horovod_broadcast";
      op_name = GetOpName(op_type_name, name);
      exec_fn = [input, output, op_name, root_rank]
                (RunContext rctx, CallbackOnComplete on_complete) mutable {
        DoBroadcast(input, output, op_name, root_rank, on_complete);
      };
      break;
    default:
      LOG(FATAL) << "Unsupported Horovod operation type";
  }

  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsync(exec_fn, MX_EXEC_CTX,
                             {input->var()}, {output->var()},
                             FnProperty::kCPUPrioritized, priority,
                             op_type_name.c_str());
  // In-place
  } else {
    Engine::Get()->PushAsync(exec_fn, MX_EXEC_CTX,
                             {}, {output->var()},
                             FnProperty::kCPUPrioritized, priority,
                             op_type_name.c_str());
  }
}

#if HAVE_CUDA
void DoAllreduceCudaOnCPU(MXTempBufferShared& hvd_cpu_buffer, std::string& name,
                          CallbackOnComplete on_complete) {
  ThrowIfError(common::CheckInitialized());

  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(
      CPU_DEVICE_ID, hvd_cpu_buffer->tensor());

  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, nullptr, name, CPU_DEVICE_ID,
      [on_complete](const Status& status) {
        InvokeCompleteCallback(on_complete, status);
      });
  ThrowIfError(enqueue_result);
}

void DoAllgatherCudaOnCPU(MXTempBufferShared& hvd_cpu_buffer, std::string& name,
                          CallbackOnComplete on_complete) {
  ThrowIfError(common::CheckInitialized());

  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(
      CPU_DEVICE_ID, hvd_cpu_buffer->tensor());

  auto enqueue_result = EnqueueTensorAllgather(
      hvd_context, hvd_cpu_buffer, nullptr, name, CPU_DEVICE_ID,
      [on_complete](const Status& status) {
        InvokeCompleteCallback(on_complete, status);
      });
  ThrowIfError(enqueue_result);
}

void DoBroadcastCudaOnCPU(MXTempBufferShared& hvd_cpu_buffer, std::string& name,
                          int root_rank, CallbackOnComplete on_complete) {
  ThrowIfError(common::CheckInitialized());

  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(
      CPU_DEVICE_ID, hvd_cpu_buffer->tensor());

  auto enqueue_result = EnqueueTensorBroadcast(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, root_rank, nullptr,
      name, CPU_DEVICE_ID,
      [on_complete](const Status& status) {
        InvokeCompleteCallback(on_complete, status);
      });
  ThrowIfError(enqueue_result);
}

inline void PushHorovodOperationCudaOnCPU(OperationType op_type, NDArray* input,
                                          NDArray* output, const char* name,
                                          int priority, int root_rank = -1) {
  auto hvd_cpu_buffer = std::make_shared<MXTemporaryBuffer<NDArray>>(
      CPU_DEVICE_ID, input->dtype());
  auto cpu_tensor = hvd_cpu_buffer->tensor();
  std::string op_type_name;
  std::string op_name;
  ExecFn exec_fn;
  switch (op_type) {
    case OperationType::ALLREDUCE:
      op_type_name = "horovod_allreduce";
      op_name = GetOpName(op_type_name, name);
      exec_fn = [hvd_cpu_buffer, op_name]
                (RunContext rctx, CallbackOnComplete on_complete) mutable {
        DoAllreduceCudaOnCPU(hvd_cpu_buffer, op_name, on_complete);
      };
      break;
    case OperationType::ALLGATHER:
      op_type_name = "horovod_allgather";
      op_name = GetOpName(op_type_name, name);
      exec_fn = [hvd_cpu_buffer, op_name]
                (RunContext rctx, CallbackOnComplete on_complete) mutable {
        DoAllgatherCudaOnCPU(hvd_cpu_buffer, op_name, on_complete);
      };
      break;
    case OperationType::BROADCAST:
      op_type_name = "horovod_broadcast";
      op_name = GetOpName(op_type_name, name);
      exec_fn = [hvd_cpu_buffer, op_name, root_rank]
                (RunContext rctx, CallbackOnComplete on_complete) mutable {
        DoBroadcastCudaOnCPU(hvd_cpu_buffer, op_name, root_rank, on_complete);
      };
      break;
    default:
      LOG(FATAL) << "Unsupported Horovod operation type.";
  }

  // Make async copy of input tensor to CPU tensor.
  TensorUtil::AsyncCopyCudaToCPU(input, cpu_tensor);

  // In-place
  Engine::Get()->PushAsync(exec_fn, MX_EXEC_CTX,
                           {}, {cpu_tensor->var()},
                           FnProperty::kCPUPrioritized, priority,
                           op_type_name.c_str());

  // Make async copy of CPU tensor to output tensor.
  TensorUtil::AsyncCopyCPUToCuda(cpu_tensor, output);
}
#endif

extern "C" int horovod_mxnet_allreduce_async(NDArray* input, NDArray* output,
                                             const char* name, bool average,
                                             int priority) {
  MX_API_BEGIN();

#if HAVE_CUDA && !HOROVOD_GPU_ALLREDUCE
  if (input->ctx().dev_mask() == cpu::kDevMask &&
      output->ctx().dev_mask() == cpu::kDevMask) {
    PushHorovodOperation(OperationType::ALLREDUCE, input, output,
                         name, priority);
  } else {
    PushHorovodOperationCudaOnCPU(OperationType::ALLREDUCE, input, output,
                                  name, priority);
  }
#else
  PushHorovodOperation(OperationType::ALLREDUCE, input, output,
                       name, priority);
#endif

  if (average) {
    *output /= horovod_size();
  }

  MX_API_END();
}

extern "C" int horovod_mxnet_allgather_async(NDArray* input, NDArray* output,
                                             const char* name, int priority) {
  MX_API_BEGIN();

#if HAVE_CUDA && !HOROVOD_GPU_ALLGATHER
  if (input->ctx().dev_mask() == cpu::kDevMask &&
      output->ctx().dev_mask() == cpu::kDevMask) {
    PushHorovodOperation(OperationType::ALLGATHER, input, output,
                         name, priority);
  } else {
    PushHorovodOperationCudaOnCPU(OperationType::ALLGATHER, input, output,
                                  name, priority);
  }
#else
  PushHorovodOperation(OperationType::ALLGATHER, input, output,
                       name, priority);
#endif

  MX_API_END();
}

extern "C" int horovod_mxnet_broadcast_async(NDArray* input, NDArray* output,
                                             const char* name, int root_rank,
                                             int priority) {
  MX_API_BEGIN();

#if HAVE_CUDA && !HOROVOD_GPU_BROADCAST
  if (input->ctx().dev_mask() == cpu::kDevMask &&
      output->ctx().dev_mask() == cpu::kDevMask) {
    PushHorovodOperation(OperationType::BROADCAST, input, output,
                         name, priority, root_rank);

  } else {
    PushHorovodOperationCudaOnCPU(OperationType::BROADCAST, input, output,
                                  name, priority, root_rank);
  }
#else
  PushHorovodOperation(OperationType::BROADCAST, input, output,
                       name, priority, root_rank);
#endif

  MX_API_END();
}

} // namespace mxnet
} // namespace horovod
