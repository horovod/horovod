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
typedef std::shared_ptr<MXTensor> MXTensorSharedPtr;
typedef std::shared_ptr<NDArray> NDArraySharedPtr;

struct MpiOpsParam {
  NDArraySharedPtr input_tensor;
  NDArraySharedPtr output_tensor;
  NDArray* output;
  NDArraySharedPtr cpu_input_tensor;
  NDArraySharedPtr cpu_output_tensor;
  OperationType op_type;
  std::string op_name;
  int root_rank;
  NDArraySharedPtr splits_tensor;

  MpiOpsParam(NDArraySharedPtr input_tensor,
              NDArraySharedPtr output_tensor,
              NDArray* output,
              NDArraySharedPtr cpu_input_tensor,
              NDArraySharedPtr cpu_output_tensor,
              const OperationType& op_type, const std::string& op_name,
              int root_rank,
              NDArraySharedPtr splits_tensor)
      : input_tensor(input_tensor),
        output_tensor(output_tensor),
        output(output),
        cpu_input_tensor(cpu_input_tensor),
        cpu_output_tensor(cpu_output_tensor),
        op_type(op_type),
        op_name(op_name),
        root_rank(root_rank),
        splits_tensor(splits_tensor) {
  }
};

inline MpiOpsParam* CreateMpiOpsParam(NDArraySharedPtr input_tensor,
                                      NDArraySharedPtr output_tensor,
                                      NDArray* output,
                                      NDArraySharedPtr cpu_input_tensor,
                                      NDArraySharedPtr cpu_output_tensor,
                                      const OperationType& op_type,
                                      const std::string& op_name,
                                      int root_rank,
                                      NDArraySharedPtr splits_tensor) {
  return new MpiOpsParam(input_tensor, output_tensor, output, cpu_input_tensor,
    cpu_output_tensor, op_type, op_name, root_rank, splits_tensor);
}

void DeleteMpiOpsParam(void* param) {
  auto ops_param = static_cast<MpiOpsParam*>(param);
  delete ops_param;
}

extern "C" int horovod_mxnet_allreduce_async(NDArray* input,
                                             NDArray* output,
                                             const char* name, bool average,
                                             int priority);
extern "C" int horovod_mxnet_allgather_async(NDArray* input,
                                             NDArray* output,
                                             const char* name, int priority);
extern "C" int horovod_mxnet_broadcast_async(NDArray* input,
                                             NDArray* output,
                                             const char* name, int root_rank,
                                             int priority);
extern "C" int horovod_mxnet_alltoall_async(NDArray* input,
                                            NDArray* output,
                                            const char* name,
                                            NDArray* splits,
                                            int priority);

} // namespace mxnet
} // namespace horovod

#endif // HOROVOD_MXNET_MPI_OPS_H
