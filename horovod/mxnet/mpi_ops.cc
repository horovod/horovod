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
#include "cuda_util.h"
#include "mpi_ops.h"

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

void DoAllreduce(RunContext rctx, CallbackOnComplete on_complete, void* param) {
  ThrowIfError(common::CheckInitialized());

  auto ops_param = static_cast<MpiOpsParam*>(param);
  auto tensor = ops_param->input;
  auto output = ops_param->output;
  auto name = ops_param->op_name;

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

void DoAllgather(RunContext rctx, CallbackOnComplete on_complete, void* param) {
  ThrowIfError(common::CheckInitialized());

  auto ops_param = static_cast<MpiOpsParam*>(param);
  auto tensor = ops_param->input;
  auto output = ops_param->output;
  auto name = ops_param->op_name;

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

void DoBroadcast(RunContext rctx, CallbackOnComplete on_complete, void* param) {
  ThrowIfError(common::CheckInitialized());

  auto ops_param = static_cast<MpiOpsParam*>(param);
  auto tensor = ops_param->input;
  auto output = ops_param->output;
  auto name = ops_param->op_name;
  auto root_rank = ops_param->root_rank;

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
  ExecFnPtr exec_fn_ptr;
  switch (op_type) {
    case OperationType::ALLREDUCE:
      op_type_name = "horovod_allreduce";
      exec_fn_ptr = DoAllreduce;
      break;
    case OperationType::ALLGATHER:
      op_type_name = "horovod_allgather";
      exec_fn_ptr = DoAllgather;
      break;
    case OperationType::BROADCAST:
      op_type_name = "horovod_broadcast";
      exec_fn_ptr = DoBroadcast;
      break;
    default:
      LOG(FATAL) << "Unsupported Horovod operation type";
  }

  auto op_name = GetOpName(op_type_name, name);
  auto ops_param = CreateMpiOpsParam(input, output, op_name, false, root_rank);

  // Not in-place
  if (input->var() != output->var()) {
    Engine::Get()->PushAsyncPtr(exec_fn_ptr, ops_param, DeleteMpiOpsParam,
                                MX_EXEC_CTX, {input->var()}, {output->var()},
                                FnProperty::kCPUPrioritized, priority,
                                op_type_name.c_str());
  // In-place
  } else {
    Engine::Get()->PushAsyncPtr(exec_fn_ptr, ops_param, DeleteMpiOpsParam,
                                MX_EXEC_CTX, {}, {output->var()},
                                FnProperty::kCPUPrioritized, priority,
                                op_type_name.c_str());
  }
}

#if HAVE_CUDA
void DoAllreduceCudaOnCPU(RunContext rctx, CallbackOnComplete on_complete, void* param) {
  ThrowIfError(common::CheckInitialized());

  auto ops_param = static_cast<MpiOpsParam*>(param);
  auto hvd_cpu_buffer = std::make_shared<MXTemporaryBuffer<NDArray>>(
      ops_param->cpu_tensor);
  auto name = ops_param->op_name;

  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(
      CPU_DEVICE_ID, hvd_cpu_buffer->tensor());

  auto enqueue_result = EnqueueTensorAllreduce(
      hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, nullptr, name, CPU_DEVICE_ID,
      [on_complete](const Status& status) {
        InvokeCompleteCallback(on_complete, status);
      });
  ThrowIfError(enqueue_result);
}

void DoAllgatherCudaOnCPU(RunContext rctx, CallbackOnComplete on_complete, void* param) {
  ThrowIfError(common::CheckInitialized());

  auto ops_param = static_cast<MpiOpsParam*>(param);
  auto hvd_cpu_buffer = std::make_shared<MXTemporaryBuffer<NDArray>>(
      ops_param->cpu_tensor);
  auto name = ops_param->op_name;

  auto hvd_context = std::make_shared<MXOpContext<NDArray>>(
      CPU_DEVICE_ID, hvd_cpu_buffer->tensor());

  auto enqueue_result = EnqueueTensorAllgather(
      hvd_context, hvd_cpu_buffer, nullptr, name, CPU_DEVICE_ID,
      [on_complete](const Status& status) {
        InvokeCompleteCallback(on_complete, status);
      });
  ThrowIfError(enqueue_result);
}

void DoBroadcastCudaOnCPU(RunContext rctx, CallbackOnComplete on_complete, void* param) {
  ThrowIfError(common::CheckInitialized());

  auto ops_param = static_cast<MpiOpsParam*>(param);
  auto hvd_cpu_buffer = std::make_shared<MXTemporaryBuffer<NDArray>>(
      ops_param->cpu_tensor);
  auto name = ops_param->op_name;
  auto root_rank = ops_param->root_rank;

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
  std::string op_type_name;
  ExecFnPtr exec_fn_ptr;
  switch (op_type) {
    case OperationType::ALLREDUCE:
      op_type_name = "horovod_allreduce";
      exec_fn_ptr = DoAllreduceCudaOnCPU;
      break;
    case OperationType::ALLGATHER:
      op_type_name = "horovod_allgather";
      exec_fn_ptr = DoAllgatherCudaOnCPU;
      break;
    case OperationType::BROADCAST:
      op_type_name = "horovod_broadcast";
      exec_fn_ptr = DoBroadcastCudaOnCPU;
      break;
    default:
      LOG(FATAL) << "Unsupported Horovod operation type.";
  }

  auto op_name = GetOpName(op_type_name, name);
  auto ops_param = CreateMpiOpsParam(input, output, op_name, true, root_rank);
  auto cpu_tensor = ops_param->cpu_tensor;

  // Make async copy of input tensor to CPU tensor.
  TensorUtil::AsyncCopyCudaToCPU(input, cpu_tensor);

  // In-place
  Engine::Get()->PushAsyncPtr(exec_fn_ptr, ops_param, DeleteMpiOpsParam,
                              MX_EXEC_CTX, {}, {cpu_tensor->var()},
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
