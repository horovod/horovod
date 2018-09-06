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

int horovod_torch_allreduce_async_torch_cuda_IntTensor(THCudaIntTensor* tensor,
                                                       THCudaIntTensor* output,
                                                       int average, char* name);
int horovod_torch_allreduce_async_torch_cuda_LongTensor(
    THCudaLongTensor* tensor, THCudaLongTensor* output, int average,
    char* name);
int horovod_torch_allreduce_async_torch_cuda_FloatTensor(THCudaTensor* tensor,
                                                         THCudaTensor* output,
                                                         int average,
                                                         char* name);
int horovod_torch_allreduce_async_torch_cuda_DoubleTensor(
    THCudaDoubleTensor* tensor, THCudaDoubleTensor* output, int average,
    char* name);

int horovod_torch_allgather_async_torch_cuda_ByteTensor(
    THCudaByteTensor* tensor, THCudaByteTensor* output, char* name);
int horovod_torch_allgather_async_torch_cuda_CharTensor(
    THCudaCharTensor* tensor, THCudaCharTensor* output, char* name);
int horovod_torch_allgather_async_torch_cuda_ShortTensor(
    THCudaShortTensor* tensor, THCudaShortTensor* output, char* name);
int horovod_torch_allgather_async_torch_cuda_IntTensor(THCudaIntTensor* tensor,
                                                       THCudaIntTensor* output,
                                                       char* name);
int horovod_torch_allgather_async_torch_cuda_LongTensor(
    THCudaLongTensor* tensor, THCudaLongTensor* output, char* name);
int horovod_torch_allgather_async_torch_cuda_FloatTensor(THCudaTensor* tensor,
                                                         THCudaTensor* output,
                                                         char* name);
int horovod_torch_allgather_async_torch_cuda_DoubleTensor(
    THCudaDoubleTensor* tensor, THCudaDoubleTensor* output, char* name);

int horovod_torch_broadcast_async_torch_cuda_ByteTensor(
    THCudaByteTensor* tensor, THCudaByteTensor* output, int root_rank,
    char* name);
int horovod_torch_broadcast_async_torch_cuda_CharTensor(
    THCudaCharTensor* tensor, THCudaCharTensor* output, int root_rank,
    char* name);
int horovod_torch_broadcast_async_torch_cuda_ShortTensor(
    THCudaShortTensor* tensor, THCudaShortTensor* output, int root_rank,
    char* name);
int horovod_torch_broadcast_async_torch_cuda_IntTensor(THCudaIntTensor* tensor,
                                                       THCudaIntTensor* output,
                                                       int root_rank,
                                                       char* name);
int horovod_torch_broadcast_async_torch_cuda_LongTensor(
    THCudaLongTensor* tensor, THCudaLongTensor* output, int root_rank,
    char* name);
int horovod_torch_broadcast_async_torch_cuda_FloatTensor(THCudaTensor* tensor,
                                                         THCudaTensor* output,
                                                         int root_rank,
                                                         char* name);
int horovod_torch_broadcast_async_torch_cuda_DoubleTensor(
    THCudaDoubleTensor* tensor, THCudaDoubleTensor* output, int root_rank,
    char* name);
