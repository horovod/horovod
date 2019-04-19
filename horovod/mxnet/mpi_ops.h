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

#ifndef HOROVOD_MXNET_MPI_OPS_H
#define HOROVOD_MXNET_MPI_OPS_H

#include <mxnet/base.h>
#include <mxnet/c_api.h>
#include <mxnet/c_api_error.h>
#include <mxnet/engine.h>
#include <mxnet/ndarray.h>

#include "adapter.h"
#include "tensor_util.h"

namespace horovod {
namespace mxnet {

using namespace horovod::common;

typedef ::mxnet::NDArray NDArray;
typedef ::mxnet::Engine::CallbackOnComplete CallbackOnComplete;
typedef Request::RequestType OperationType;

struct MpiOpsParam {
  NDArray* input;
  NDArray* output;
  NDArray* cpu_tensor;
  OperationType op_type;
  std::string op_name;
  int root_rank;

  MpiOpsParam(NDArray* input, NDArray* output, NDArray* cpu_tensor,
              const OperationType& op_type, const std::string& op_name,
              int root_rank)
      : input(input),
        output(output),
        cpu_tensor(cpu_tensor),
        op_type(op_type),
        op_name(op_name),
        root_rank(root_rank) {
  }
};

inline MpiOpsParam* CreateMpiOpsParam(NDArray* input, NDArray* output,
                                      const OperationType& op_type,
                                      const std::string& op_name,
                                      int root_rank, bool cuda_on_cpu) {
  if (cuda_on_cpu) {
    auto cpu_tensor = TensorUtil::New(CPU_DEVICE_ID, input->dtype());
    return new MpiOpsParam(nullptr, nullptr, cpu_tensor, op_type, op_name, root_rank);
  }

  return new MpiOpsParam(input, output, nullptr, op_type, op_name, root_rank);
}

void DeleteMpiOpsParam(void* param) {
  auto ops_param = static_cast<MpiOpsParam*>(param);
  delete ops_param;
}

extern "C" int horovod_mxnet_allreduce_async(NDArray* input, NDArray* output,
                                             const char* name, bool average,
                                             int priority);
extern "C" int horovod_mxnet_allgather_async(NDArray* input, NDArray* output,
                                             const char* name, int priority);
extern "C" int horovod_mxnet_broadcast_async(NDArray* input, NDArray* output,
                                             const char* name, int root_rank,
                                             int priority);

} // namespace mxnet
} // namespace horovod

#endif // HOROVOD_MXNET_MPI_OPS_H
