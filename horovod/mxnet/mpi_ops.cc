// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// Modifications copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

std::string GetOpName(const char* prefix, const char* name) {
  if (name != nullptr) {
    return std::string(prefix) + "." + std::string(name);
  }

  op_count.fetch_add(1);
  return std::string(prefix) + ".noname." + std::to_string(op_count);
}
} // namespace

static const auto MX_EXEC_CTX = Context();
static const auto MX_FUNC_PROP = FnProperty::kCPUPrioritized;
static const char* ALLREDUCE_OP_TYPE_NAME = "horovod_allreduce";
static const char* ALLGATHER_OP_TYPE_NAME = "horovod_allgather";
static const char* BROADCAST_OP_TYPE_NAME = "horovod_broadcast";
static const char* ALLTOALL_OP_TYPE_NAME = "horovod_alltoall";

inline void InvokeCompleteCallback(CallbackOnComplete on_complete, const Status& status) {
  if (status.ok()) {
    on_complete();
  } else {
    auto error = dmlc::Error(status.reason());
    on_complete(&error);
  }
}

inline const char* GetOpTypeName(OperationType op_type) {
  switch (op_type) {
    case OperationType::ALLREDUCE:
      return ALLREDUCE_OP_TYPE_NAME;
    case OperationType::ALLGATHER:
      return ALLGATHER_OP_TYPE_NAME;
    case OperationType::BROADCAST:
      return BROADCAST_OP_TYPE_NAME;
    case OperationType::ALLTOALL:
      return ALLTOALL_OP_TYPE_NAME;
    default:
      throw std::logic_error("Unsupported Horovod operation type.");
  }
}

bool IsTensorOnCPU(NDArray* tensor) {
  return tensor->ctx().dev_mask() == cpu::kDevMask;
}

void DoHorovodOperation(void*, void* on_complete_ptr, void* param) {
  ThrowIfError(common::CheckInitialized());

  auto on_complete = *static_cast<CallbackOnComplete*>(on_complete_ptr);
  auto ops_param = static_cast<MpiOpsParam*>(param);
  auto input_tensor = ops_param->input_tensor.get();
  auto output_tensor = ops_param->output_tensor.get();
  auto output = ops_param->output;
  auto name = ops_param->op_name;
  auto average = ops_param->average;
  auto prescale_factor = ops_param->prescale_factor;
  auto postscale_factor = ops_param->postscale_factor;
  auto device = TensorUtil::GetDevice(input_tensor);

  auto hvd_tensor = std::make_shared<MXTensor>(input_tensor);
  auto hvd_context = std::make_shared<MXOpContext>(device, output);  
  std::shared_ptr<Tensor> hvd_output = nullptr;  

  Status enqueue_result;
  switch (ops_param->op_type) {
    case OperationType::ALLREDUCE:
      hvd_output = std::make_shared<MXTensor>(output_tensor);
      enqueue_result = EnqueueTensorAllreduce(
          hvd_context, hvd_tensor, hvd_output, nullptr, name, device,
          [on_complete](const Status& status) {
            InvokeCompleteCallback(on_complete, status);
      }, (average) ? ReduceOp::AVERAGE : ReduceOp::SUM, prescale_factor, postscale_factor);
      break;
    case OperationType::ALLGATHER:
      enqueue_result = EnqueueTensorAllgather(
          hvd_context, hvd_tensor, nullptr, name, device,
          [on_complete](const Status& status) {
            InvokeCompleteCallback(on_complete, status);
      });
      break;
    case OperationType::BROADCAST:
      if (horovod_rank() != ops_param->root_rank) {
        hvd_output = std::make_shared<MXTensor>(output_tensor);
      }

      enqueue_result = EnqueueTensorBroadcast(
          hvd_context, hvd_tensor, hvd_output, ops_param->root_rank,
          nullptr, name, device,
          [on_complete](const Status& status) {
            InvokeCompleteCallback(on_complete, status);
      });
      break;
    case OperationType::ALLTOALL:
    {
      auto hvd_splits = std::make_shared<MXTensor>(ops_param->splits_tensor.get());
      enqueue_result = EnqueueTensorAlltoall(
          hvd_context, hvd_tensor, hvd_splits, nullptr, name, device,
          [on_complete](const Status& status) {
            InvokeCompleteCallback(on_complete, status);
      });
      break;
    }
    default:
      throw std::logic_error("Unsupported Horovod operation type.");
  }

  ThrowIfError(enqueue_result);
}

inline void PushHorovodOperation(OperationType op_type, NDArray* input,
                                 NDArray* output, const char* name,
                                 int priority, int root_rank = -1,
                                 bool average = true,
                                 NDArray* splits = nullptr,
                                 double prescale_factor = 1.0,
                                 double postscale_factor = 1.0) {
  auto op_type_name = GetOpTypeName(op_type);
  auto op_name = GetOpName(op_type_name, name);

  // We need to create a shared_ptr to NDArray object with
  // shallow copy to prevent from NDArray object being freed
  // before MXNet engine process it
  auto input_copy = std::make_shared<NDArray>(*input);
  auto output_copy = std::make_shared<NDArray>(*output);
  std::shared_ptr<NDArray> splits_tensor;
  if (splits) {
#if HAVE_CUDA
    // We expect splits to be a tensor on CPU. Create CPU copy if required.
    if (!IsTensorOnCPU(splits)) {
      splits_tensor = std::make_shared<NDArray>(Context::Create(Context::kCPU, 0),
      splits->dtype());
      TensorUtil::AsyncCopyCudaToCPU(splits, splits_tensor.get());
    } else {
      splits_tensor = std::make_shared<NDArray>(*splits);
    }
#else
    splits_tensor = std::make_shared<NDArray>(*splits);
#endif
  }
  auto ops_param = CreateMpiOpsParam(input_copy, output_copy, output,
    nullptr /* cpu_input_tensor */, nullptr /* cpu_output_tensor */,
    op_type, op_name, root_rank, average, splits_tensor, prescale_factor, postscale_factor);

  // Not in-place
  auto input_var = input->var();
  auto output_var = output->var();
  if (input_var != output_var) {
    std::vector<void*> input_vars {input_var};
    if (splits) {
      // Add splits tensor to input list to enforce dependency on possible async D2H copy
      input_vars.push_back(splits_tensor->var());
    }
    MXEnginePushAsync(DoHorovodOperation, ops_param, DeleteMpiOpsParam,
                      &MX_EXEC_CTX, input_vars.data(), input_vars.size(), &output_var, 1,
                      &MX_FUNC_PROP, priority, op_type_name);
  // In-place
  } else {
    std::vector<void*> input_vars;
    if (splits) {
      input_vars.push_back(splits_tensor->var());
    }
    MXEnginePushAsync(DoHorovodOperation, ops_param, DeleteMpiOpsParam,
                      &MX_EXEC_CTX, input_vars.data(), input_vars.size(), &output_var, 1,
                      &MX_FUNC_PROP, priority, op_type_name);
  }
}

#if HAVE_CUDA
void DoHorovodOperationCudaOnCPU(void*, void* on_complete_ptr, void* param) {
  ThrowIfError(common::CheckInitialized());

  auto on_complete = *static_cast<CallbackOnComplete*>(on_complete_ptr);
  auto ops_param = static_cast<MpiOpsParam*>(param);
  auto name = ops_param->op_name;
  auto hvd_cpu_buffer = std::make_shared<MXTensor>(ops_param->cpu_input_tensor.get());
  auto hvd_context = std::make_shared<MXOpContext>(
    CPU_DEVICE_ID, ops_param->cpu_output_tensor.get());
  auto average = ops_param->average;
  auto prescale_factor = ops_param->prescale_factor;
  auto postscale_factor = ops_param->postscale_factor;

  Status enqueue_result;
  switch (ops_param->op_type) {
    case OperationType::ALLREDUCE:
      enqueue_result = EnqueueTensorAllreduce(
          hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, nullptr, name, CPU_DEVICE_ID,
          [on_complete](const Status& status) {
            InvokeCompleteCallback(on_complete, status);
      }, (average) ? ReduceOp::AVERAGE : ReduceOp::SUM, prescale_factor, postscale_factor);
      break;
    case OperationType::ALLGATHER:
      enqueue_result = EnqueueTensorAllgather(
          hvd_context, hvd_cpu_buffer, nullptr, name, CPU_DEVICE_ID,
          [on_complete](const Status& status) {
            InvokeCompleteCallback(on_complete, status);
      });
      break;
    case OperationType::BROADCAST:
      enqueue_result = EnqueueTensorBroadcast(
          hvd_context, hvd_cpu_buffer, hvd_cpu_buffer, ops_param->root_rank,
          nullptr, name, CPU_DEVICE_ID,
          [on_complete](const Status& status) {
            InvokeCompleteCallback(on_complete, status);
      });
      break;
    case OperationType::ALLTOALL:
    {
      auto hvd_splits = std::make_shared<MXTensor>(ops_param->splits_tensor.get());
      enqueue_result = EnqueueTensorAlltoall(
          hvd_context, hvd_cpu_buffer, hvd_splits, nullptr, name, CPU_DEVICE_ID,
          [on_complete](const Status& status) {
            InvokeCompleteCallback(on_complete, status);
      });
      break;
    }
    default:
      throw std::logic_error("Unsupported Horovod operation type.");
  }

  ThrowIfError(enqueue_result);
}

inline void PushHorovodOperationCudaOnCPU(OperationType op_type, NDArray* input,
                                          NDArray* output, const char* name,
                                          int priority, int root_rank = -1,
                                          bool average = true,
                                          NDArray* splits = nullptr,
                                          double prescale_factor = 1.0,
                                          double postscale_factor = 1.0) {
  auto op_type_name = GetOpTypeName(op_type);
  auto op_name = GetOpName(op_type_name, name);

  auto cpu_input_tensor = std::make_shared<NDArray>(Context::Create(Context::kCPU, 0),
    input->dtype());
  auto cpu_output_tensor = std::make_shared<NDArray>(Context::Create(Context::kCPU, 0),
    input->dtype());

  // Make async copy of input tensor to CPU tensor.
  TensorUtil::AsyncCopyCudaToCPU(input, cpu_input_tensor.get());

  std::shared_ptr<NDArray> splits_tensor;
  if (splits) {
    // We expect splits to be a tensor on CPU. Create CPU copy if required.
    if (!IsTensorOnCPU(splits)) {
      splits_tensor = std::make_shared<NDArray>(Context::Create(Context::kCPU, 0),
      splits->dtype());
      TensorUtil::AsyncCopyCudaToCPU(splits, splits_tensor.get());
    } else {
      splits_tensor = std::make_shared<NDArray>(*splits);
    }
  }

  auto ops_param = CreateMpiOpsParam(nullptr, nullptr, output, cpu_input_tensor,
                                     cpu_output_tensor, op_type, op_name, root_rank,
                                     average, splits_tensor, prescale_factor, postscale_factor);

  auto input_var = input->var();
  auto output_var = output->var();
  auto cpu_input_var = cpu_input_tensor->var();
  auto cpu_output_var = cpu_output_tensor->var();
  if (op_type == OperationType::ALLGATHER ||
      op_type == OperationType::ALLTOALL) {
    // Use out-of-place path for operations that have unknown output size (allgather, alltoall)
    std::vector<void*> input_vars {cpu_input_var};
    if (splits) {
      // Add splits tensor to input list to enforce dependency on possible async D2H copy
      input_vars.push_back(splits_tensor->var());
    }

    MXEnginePushAsync(DoHorovodOperationCudaOnCPU, ops_param, DeleteMpiOpsParam,
                      &MX_EXEC_CTX, input_vars.data(), input_vars.size(), &cpu_output_var, 1,
                      &MX_FUNC_PROP, priority, op_type_name);

    // Since cpu_output_tensor is resized in out-of-place path, need
    // to wait for operation to complete before copying to GPU output.
    cpu_output_tensor->WaitToRead();

    // Make async copy of CPU output tensor to output tensor.
    TensorUtil::AsyncCopyCPUToCuda(cpu_output_tensor.get(), output);
  } else {
    // Use in-place otherwise
    MXEnginePushAsync(DoHorovodOperationCudaOnCPU, ops_param, DeleteMpiOpsParam,
                      &MX_EXEC_CTX, nullptr, 0, &cpu_input_var, 1,
                      &MX_FUNC_PROP, priority, op_type_name);

    // Make async copy of CPU input tensor to output tensor.
    TensorUtil::AsyncCopyCPUToCuda(cpu_input_tensor.get(), output);
  }
}
#endif

extern "C" int horovod_mxnet_allreduce_async(NDArray* input, NDArray* output,
                                             const char* name, bool average,
                                             int priority,
                                             double prescale_factor,
                                             double postscale_factor) {
  MX_API_BEGIN();

#if HAVE_ROCM
  // Averaging left at framework level for ROCm until ScaleBuffer implementation
  // added.
  bool average_in_framework = average;
  average = false;
#endif

#if HAVE_CUDA && !HOROVOD_GPU_ALLREDUCE
  if (IsTensorOnCPU(input) && IsTensorOnCPU(output)) {
    PushHorovodOperation(OperationType::ALLREDUCE, input, output,
                         name, priority, -1, average, nullptr, prescale_factor, postscale_factor);
  } else {
    PushHorovodOperationCudaOnCPU(OperationType::ALLREDUCE, input, output,
                                  name, priority, -1, average, nullptr, prescale_factor, postscale_factor);
  }
#else
  PushHorovodOperation(OperationType::ALLREDUCE, input, output,
                       name, priority, -1, average, nullptr, prescale_factor, postscale_factor);
#endif

#if HAVE_ROCM
  if (average_in_framework) {
    *output /= horovod_size();
  }
#endif

  MX_API_END();
}

extern "C" int horovod_mxnet_allgather_async(NDArray* input,
                                             NDArray* output,
                                             const char* name, int priority) {
  MX_API_BEGIN();

#if HAVE_CUDA && !HOROVOD_GPU_ALLGATHER
  if (IsTensorOnCPU(input) && IsTensorOnCPU(output)) {
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

extern "C" int horovod_mxnet_broadcast_async(NDArray* input,
                                             NDArray* output,
                                             const char* name, int root_rank,
                                             int priority) {
  MX_API_BEGIN();

#if HAVE_CUDA && !HOROVOD_GPU_BROADCAST
  if (IsTensorOnCPU(input) && IsTensorOnCPU(output)) {
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

extern "C" int horovod_mxnet_alltoall_async(NDArray* input,
                                            NDArray* output,
                                            const char* name,
                                            NDArray* splits,
                                            int priority) {
  MX_API_BEGIN();

#if HAVE_CUDA && !HOROVOD_GPU_ALLTOALL
  if (IsTensorOnCPU(input) && IsTensorOnCPU(output)) {
    PushHorovodOperation(OperationType::ALLTOALL, input, output,
                         name, priority, -1, false, splits);

  } else {
    PushHorovodOperationCudaOnCPU(OperationType::ALLTOALL, input, output,
                                  name, priority, -1, false, splits);
  }
#else
  PushHorovodOperation(OperationType::ALLTOALL, input, output,
                       name, priority, -1, false, splits);
#endif

  MX_API_END();
}

} // namespace mxnet
} // namespace horovod
