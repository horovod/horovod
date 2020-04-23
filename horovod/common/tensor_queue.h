// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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

#ifndef HOROVOD_TENSOR_QUEUE_H
#define HOROVOD_TENSOR_QUEUE_H

#include <iostream>
#include <mutex>
#include <queue>

#include "common.h"

namespace horovod {
namespace common {

class TensorQueue {
public:
  TensorQueue() = default;
  TensorQueue(const TensorQueue&) = delete;
  Status AddToTensorQueue(TensorTableEntry& e, Request& message);

  void FinalizeTensorQueue(std::vector<StatusCallback>& callbacks_buffer);

  int64_t GetTensorDataForAutotuner(const ResponseList& response_list,
                                    std::vector<std::string>& tensor_names);

  void GetTensorEntriesFromResponse(const Response& response,
                                    std::vector<TensorTableEntry>& entries,
                                    bool joined = false);

  const TensorTableEntry& GetTensorEntry(const std::string& tensor_name) const;

  void PopMessagesFromQueue(std::deque<Request>& message_queue_buffer);

  void PushMessageToQueue(Request& message);

  void PushMessagesToQueue(std::deque<Request>& messages);

  void RemoveJoinTensor();

protected:
  // Tensors waiting to be allreduced or allgathered.
  std::unordered_map<std::string, TensorTableEntry> tensor_table_;

  // Queue of MPI requests waiting to be sent to the coordinator node.
  std::queue<Request> message_queue_;

  // A mutex that needs to be used whenever operations on message queue are
  // done.
  mutable std::mutex mutex_;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_TENSOR_QUEUE_H
