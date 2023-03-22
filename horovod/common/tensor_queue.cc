// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
// Modifications copyright Microsoft
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

#include "tensor_queue.h"

#include <assert.h>

#include "logging.h"

namespace horovod {
namespace common {

const Status DUPLICATE_NAME_ERROR = Status::InvalidArgument(
    "Requested to allreduce, allgather, or broadcast a tensor with the same "
    "name as another tensor that is currently being processed.  If you want "
    "to request another tensor, use a different tensor name.");

// Add a TensorTableEntry as well as its message to the queue.
Status TensorQueue::AddToTensorQueue(TensorTableEntry& e, Request& message) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (tensor_table_.find(e.tensor_name) != tensor_table_.end()) {
    return DUPLICATE_NAME_ERROR;
  }
  tensor_table_.emplace(e.tensor_name, std::move(e));
  message_queue_.push(std::move(message));
  return Status::OK();
}

Status TensorQueue::AddToTensorQueueMulti(std::vector<TensorTableEntry>& entries,
                                          std::vector<Request>& messages) {
  std::lock_guard<std::mutex> guard(mutex_);

  for (std::size_t i = 0; i < entries.size(); ++i) {
    if (tensor_table_.find(entries[i].tensor_name) != tensor_table_.end()) {
      return DUPLICATE_NAME_ERROR;
    }
    tensor_table_.emplace(entries[i].tensor_name, std::move(entries[i]));
    message_queue_.push(std::move(messages[i]));
  }
  return Status::OK();
}

// Execute callback for each tensor and clear tensor queue
void TensorQueue::FinalizeTensorQueue(const Status& status) {
  std::lock_guard<std::mutex> guard(mutex_);
  for (auto& e : tensor_table_) {
    e.second.FinishWithCallback(status);
  }
  tensor_table_.clear();
  while (!message_queue_.empty()) {
    message_queue_.pop();
  }
}

// Helper function to get list of allreduced tensor names and total size for
// use with the autotuner.
int64_t
TensorQueue::GetTensorDataForAutotuner(const ResponseList& response_list,
                                       std::vector<std::string>& tensor_names) {
  int64_t total_tensor_size = 0;
  for (auto& response : response_list.responses()) {
    if (response.response_type() == Response::ResponseType::ALLREDUCE) {
      for (auto& tensor_name : response.tensor_names()) {
        tensor_names.push_back(tensor_name);
        LOG(TRACE) << "Looking for tensor with name " << tensor_name;
        auto& entry = tensor_table_.at(tensor_name);
        LOG(TRACE) << "Found tensor with name " << tensor_name;
        total_tensor_size += entry.tensor->size();
      }
    }
  }
  return total_tensor_size;
}

// Parse tensor names from response and generate a vector of corresponding
// tensor entries.
void TensorQueue::GetTensorEntriesFromResponse(
    const Response& response, std::vector<TensorTableEntry>& entries,
    bool joined) {
  // Reserve to save re-allocation costs, as we know the size before.
  entries.reserve(response.tensor_names().size());
  {
    // Lock on the tensor table.
    std::lock_guard<std::mutex> guard(mutex_);
    if (response.response_type() == Response::JOIN) {
      assert(response.tensor_names().size() == 1);
      assert(response.tensor_names()[0] == JOIN_TENSOR_NAME);
      auto iter = tensor_table_.find(JOIN_TENSOR_NAME);
      assert(iter != tensor_table_.end());

      entries.push_back(std::move(iter->second));

      // The tensor table will be cleared of the join tensor later in
      // RemoveJoinTensor().
      return;
    }
    int64_t i = 0;
    for (auto& name : response.tensor_names()) {
      assert(response.response_type() == Response::ALLREDUCE ||
             response.response_type() == Response::ALLGATHER ||
             response.response_type() == Response::BROADCAST ||
             response.response_type() == Response::ALLTOALL ||
             response.response_type() == Response::REDUCESCATTER ||
             response.response_type() == Response::ADASUM ||
             response.response_type() == Response::BARRIER ||
             response.response_type() == Response::ERROR);

      if (!joined) {
        // We should never fail at finding this key in the tensor table.
        auto iter = tensor_table_.find(name);
        assert(iter != tensor_table_.end());
        entries.push_back(std::move(iter->second));

        // Clear the tensor table of this tensor.
        tensor_table_.erase(iter);

      } else if (response.response_type() != Response::ERROR) {

        // Find Join tensor to use its context.
        auto join_iter = tensor_table_.find(JOIN_TENSOR_NAME);
        assert(join_iter != tensor_table_.end());

        TensorTableEntry entry;
        join_iter->second.context->AllocateZeros(response.tensor_sizes()[i],
                                                 response.tensor_type(),
                                                 &(entry.tensor));

        entry.output = entry.tensor;
        entry.device = join_iter->second.device;
        entry.context = join_iter->second.context;
        entry.tensor_name = name;
        entries.push_back(std::move(entry));
      }
      i++;
    }
  }
}

// Get tensor entry given a tensor name
const TensorTableEntry&
TensorQueue::GetTensorEntry(const std::string& tensor_name) const{
  // Lock on the tensor table.
  std::lock_guard<std::mutex> guard(mutex_);
  auto& iter = tensor_table_.at(tensor_name);

  return iter;
}

bool TensorQueue::IsTensorPresentInTable(const std::string& tensor_name) const {
  // Lock on the tensor table.
  std::lock_guard<std::mutex> guard(mutex_);
  return tensor_table_.find(tensor_name) != tensor_table_.end();
}

// Pop out all the messages from the queue
void TensorQueue::PopMessagesFromQueue(
    std::deque<Request>& message_queue_buffer) {
  std::lock_guard<std::mutex> guard(mutex_);
  while (!message_queue_.empty()) {
    Request& message = message_queue_.front();
    message_queue_buffer.push_back(std::move(message));
    message_queue_.pop();
  }
}

// Push a message to message queue
void TensorQueue::PushMessageToQueue(Request& message) {
  std::lock_guard<std::mutex> guard(mutex_);
  message_queue_.push(std::move(message));
}

// Push messages to message queue
void TensorQueue::PushMessagesToQueue(
    std::deque<Request>& messages) {
  std::lock_guard<std::mutex> guard(mutex_);
  while (!messages.empty()) {
    Request& message = messages.front();
    message_queue_.push(std::move(message));
    messages.pop_front();
  }
}

// Remove JoinOp tensor from the table and execute the callback
void TensorQueue::RemoveJoinTensor() {
  // Lock on the tensor table.
  std::lock_guard<std::mutex> guard(mutex_);
  auto iter = tensor_table_.find(JOIN_TENSOR_NAME);
  assert(iter != tensor_table_.end());
  auto& e = iter->second;
  e.FinishWithCallback(Status::OK());
  tensor_table_.erase(iter);
}

} // namespace common
} // namespace horovod
