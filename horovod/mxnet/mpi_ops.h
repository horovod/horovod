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

namespace horovod {
namespace mxnet {

using namespace horovod::common;

typedef ::mxnet::Engine Engine;
typedef ::mxnet::NDArray NDArray;
typedef ::mxnet::Engine::CallbackOnComplete Callback;

extern "C" int horovod_mxnet_allreduce_async(NDArray* tensor, NDArray* output,
                                             char* name, bool average);
extern "C" int horovod_mxnet_allgather_async(NDArray* tensor, NDArray* output,
                                             char* name);
extern "C" int horovod_mxnet_broadcast_async(NDArray* tensor, NDArray* output,
                                             int root_rank, char* name);

} // namespace mxnet
} // namespace horovod

#endif // HOROVOD_MXNET_MPI_OPS_H
