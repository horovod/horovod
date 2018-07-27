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

#ifndef HOROVOD_TORCH_MPI_OPS_H
#define HOROVOD_TORCH_MPI_OPS_H

#include <TH/TH.h>

#if HAVE_CUDA
#include <THC/THC.h>
#endif

namespace horovod {
namespace torch {

using namespace horovod::common;

#define ALLREDUCE_H(torch_Tensor, THTensor)                                    \
  extern "C" int horovod_torch_allreduce_async_##torch_Tensor(                 \
      THTensor* tensor, THTensor* output, char* name);

ALLREDUCE_H(torch_IntTensor, THIntTensor)
ALLREDUCE_H(torch_LongTensor, THLongTensor)
ALLREDUCE_H(torch_FloatTensor, THFloatTensor)
ALLREDUCE_H(torch_DoubleTensor, THDoubleTensor)

#if HAVE_CUDA
ALLREDUCE_H(torch_cuda_IntTensor, THCudaIntTensor)
ALLREDUCE_H(torch_cuda_LongTensor, THCudaLongTensor)
ALLREDUCE_H(torch_cuda_FloatTensor, THCudaTensor)
ALLREDUCE_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)
#endif

#define ALLGATHER_H(torch_Tensor, THTensor)                                    \
  extern "C" int horovod_torch_allgather_async_##torch_Tensor(                 \
      THTensor* tensor, THTensor* output, char* name);

ALLGATHER_H(torch_ByteTensor, THByteTensor)
ALLGATHER_H(torch_CharTensor, THCharTensor)
ALLGATHER_H(torch_ShortTensor, THShortTensor)
ALLGATHER_H(torch_IntTensor, THIntTensor)
ALLGATHER_H(torch_LongTensor, THLongTensor)
ALLGATHER_H(torch_FloatTensor, THFloatTensor)
ALLGATHER_H(torch_DoubleTensor, THDoubleTensor)

#if HAVE_CUDA
ALLGATHER_H(torch_cuda_ByteTensor, THCudaByteTensor)
ALLGATHER_H(torch_cuda_CharTensor, THCudaCharTensor)
ALLGATHER_H(torch_cuda_ShortTensor, THCudaShortTensor)
ALLGATHER_H(torch_cuda_IntTensor, THCudaIntTensor)
ALLGATHER_H(torch_cuda_LongTensor, THCudaLongTensor)
ALLGATHER_H(torch_cuda_FloatTensor, THCudaTensor)
ALLGATHER_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)
#endif

#define BROADCAST_H(torch_Tensor, THTensor)                                    \
  extern "C" int horovod_torch_broadcast_async_##torch_Tensor(                 \
      THTensor* tensor, THTensor* output, int root_rank, char* name);

BROADCAST_H(torch_ByteTensor, THByteTensor)
BROADCAST_H(torch_CharTensor, THCharTensor)
BROADCAST_H(torch_ShortTensor, THShortTensor)
BROADCAST_H(torch_IntTensor, THIntTensor)
BROADCAST_H(torch_LongTensor, THLongTensor)
BROADCAST_H(torch_FloatTensor, THFloatTensor)
BROADCAST_H(torch_DoubleTensor, THDoubleTensor)

#if HAVE_CUDA
BROADCAST_H(torch_cuda_ByteTensor, THCudaByteTensor)
BROADCAST_H(torch_cuda_CharTensor, THCudaCharTensor)
BROADCAST_H(torch_cuda_ShortTensor, THCudaShortTensor)
BROADCAST_H(torch_cuda_IntTensor, THCudaIntTensor)
BROADCAST_H(torch_cuda_LongTensor, THCudaLongTensor)
BROADCAST_H(torch_cuda_FloatTensor, THCudaTensor)
BROADCAST_H(torch_cuda_DoubleTensor, THCudaDoubleTensor)
#endif

extern "C" int horovod_torch_poll(int handle);
extern "C" void horovod_torch_wait_and_clear(int handle);

} // namespace torch
} // namespace horovod

#endif // HOROVOD_TORCH_MPI_OPS_H
