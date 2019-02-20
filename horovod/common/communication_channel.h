// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
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

#ifndef HOROVOD_COMMUNICATION_CONTEXT_H
#define HOROVOD_COMMUNICATION_CONTEXT_H

#include "message.h"

namespace horovod {
namespace common {

class Channel {
public:
  enum Communicator {
    GLOBAL = 0,
    LOCAL = 1,
    CROSS = 2
  };

  inline std::string CommunicatorName(Communicator comm) {
    switch (comm) {
      case GLOBAL:
        return "global";
      case LOCAL:
        return "local";
      case CROSS:
        return "cross";
      default:
        return "<unknown>";
    }
  }

  virtual void Allreduce(const void* buffer_data, int64_t num_elements,
                         TensorTableEntry& first_entry, const void* sendbuff=nullptr,
                         Communicator comm=Communicator::GLOBAL) = 0;

  virtual void Allgatherv(const void *sendbuf, int sendcount, DataType sendtype,
                          void *recvbuf, const int recvcounts[],
                          const int displs[], DataType recvtype,
                          Communicator comm=Communicator::GLOBAL) = 0;

  virtual void Broadcast(const void* buffer_data, int64_t num_elements,
                         DataType dtype, int root_rank,
                         Communicator comm=Communicator::GLOBAL) = 0;

  virtual void Barrier(Communicator comm=Communicator::GLOBAL) = 0;

  virtual void AllocateSharedBuffer(int64_t window_size, int element_size, void* baseptr,
                                    Communicator comm=Communicator::GLOBAL) = 0;

  virtual void FreeSharedBuffer() = 0;

  virtual void QuerySharedBuffer(int rank, void* baseptr) = 0;

  virtual void GetTypeSize(DataType dtype, int* out) = 0;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_COMMUNICATION_CONTEXT_H
