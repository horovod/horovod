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

int horovod_torch_allreduce_async_torch_IntTensor(THIntTensor* tensor,
                                                  THIntTensor* output,
                                                  int average, char* name);
int horovod_torch_allreduce_async_torch_LongTensor(THLongTensor* tensor,
                                                   THLongTensor* output,
                                                   int average, char* name);
int horovod_torch_allreduce_async_torch_FloatTensor(THFloatTensor* tensor,
                                                    THFloatTensor* output,
                                                    int average, char* name);
int horovod_torch_allreduce_async_torch_DoubleTensor(THDoubleTensor* tensor,
                                                     THDoubleTensor* output,
                                                     int average, char* name);

int horovod_torch_allgather_async_torch_ByteTensor(THByteTensor* tensor,
                                                   THByteTensor* output,
                                                   char* name);
int horovod_torch_allgather_async_torch_CharTensor(THCharTensor* tensor,
                                                   THCharTensor* output,
                                                   char* name);
int horovod_torch_allgather_async_torch_ShortTensor(THShortTensor* tensor,
                                                    THShortTensor* output,
                                                    char* name);
int horovod_torch_allgather_async_torch_IntTensor(THIntTensor* tensor,
                                                  THIntTensor* output,
                                                  char* name);
int horovod_torch_allgather_async_torch_LongTensor(THLongTensor* tensor,
                                                   THLongTensor* output,
                                                   char* name);
int horovod_torch_allgather_async_torch_FloatTensor(THFloatTensor* tensor,
                                                    THFloatTensor* output,
                                                    char* name);
int horovod_torch_allgather_async_torch_DoubleTensor(THDoubleTensor* tensor,
                                                     THDoubleTensor* output,
                                                     char* name);

int horovod_torch_broadcast_async_torch_ByteTensor(THByteTensor* tensor,
                                                   THByteTensor* output,
                                                   int root_rank, char* name);
int horovod_torch_broadcast_async_torch_CharTensor(THCharTensor* tensor,
                                                   THCharTensor* output,
                                                   int root_rank, char* name);
int horovod_torch_broadcast_async_torch_ShortTensor(THShortTensor* tensor,
                                                    THShortTensor* output,
                                                    int root_rank, char* name);
int horovod_torch_broadcast_async_torch_IntTensor(THIntTensor* tensor,
                                                  THIntTensor* output,
                                                  int root_rank, char* name);
int horovod_torch_broadcast_async_torch_LongTensor(THLongTensor* tensor,
                                                   THLongTensor* output,
                                                   int root_rank, char* name);
int horovod_torch_broadcast_async_torch_FloatTensor(THFloatTensor* tensor,
                                                    THFloatTensor* output,
                                                    int root_rank, char* name);
int horovod_torch_broadcast_async_torch_DoubleTensor(THDoubleTensor* tensor,
                                                     THDoubleTensor* output,
                                                     int root_rank, char* name);

int horovod_torch_poll(int handle);
void horovod_torch_wait_and_clear(int handle);
