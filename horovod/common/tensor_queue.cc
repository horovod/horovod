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

#include "tensor_queue.h"

#include <assert.h>

#include "logging.h"

namespace horovod {
namespace common {

// Add a TensorTableEntry as well as its message to the queue.
Status TensorQueue::AddToTensorQueue(TensorTableEntry& e, Request& message) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (tensor_table_.find(e.tensor_name) != tensor_table_.end()) {
    return DUPLICATE_NAME_ERROR;
  }
  tensor_table_.emplace(e.tensor_name, std::move(e));
  message_queue_.push(message);
  return Status::OK();
}

// Put callbacks for each tensor in the callback buffer and clear tensor queue
void TensorQueue::FinalizeTensorQueue(
    std::vector<StatusCallback>& callbacks_buffer) {
  std::lock_guard<std::mutex> guard(mutex_);
  for (auto& e : tensor_table_) {
    callbacks_buffer.emplace_back(e.second.callback);
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
    Response& response, std::vector<TensorTableEntry>& entries,
    int rank, bool joined, int join_device) {
  // Reserve to save re-allocation costs, as we know the size before.
  entries.reserve(response.tensor_names().size());
  {
    // Lock on the tensor table.
    std::lock_guard<std::mutex> guard(mutex_);
    for (auto& name : response.tensor_names()) {
      assert(response.response_type() == Response::ALLREDUCE ||
             response.response_type() == Response::ALLGATHER ||
             response.response_type() == Response::BROADCAST ||
             response.response_type() == Response::ERROR);

      if (!joined) {
        // We should never fail at finding this key in the tensor table.
        auto iter = tensor_table_.find(name);
        assert(iter != tensor_table_.end());

        entries.push_back(std::move(iter->second));

        // Clear the tensor table of this tensor.
        tensor_table_.erase(iter);
      } else {
        TensorTableEntry entry;
        switch (response.tensor_type()) {
        case HOROVOD_UINT8:
          entry.tensor = std::make_shared<JoinTensor<HOROVOD_UINT8, uint8_t>>(
              join_device, response.tensor_sizes()[0]);
          break;
        case HOROVOD_INT8:
          entry.tensor = std::make_shared<JoinTensor<HOROVOD_INT8, int8_t>>(
              join_device, response.tensor_sizes()[0]);
          break;
        case HOROVOD_UINT16:
          entry.tensor = std::make_shared<JoinTensor<HOROVOD_UINT16, uint16_t>>(
              join_device, response.tensor_sizes()[0]);
          break;
        case HOROVOD_INT16:
          entry.tensor = std::make_shared<JoinTensor<HOROVOD_INT16, int16_t>>(
              join_device, response.tensor_sizes()[0]);
          break;
        case HOROVOD_INT32:
          entry.tensor = std::make_shared<JoinTensor<HOROVOD_INT32, int32_t>>(
              join_device, response.tensor_sizes()[0]);
          break;
        case HOROVOD_INT64:
          entry.tensor = std::make_shared<JoinTensor<HOROVOD_INT64, int64_t>>(
              join_device, response.tensor_sizes()[0]);
          break;
        case HOROVOD_FLOAT16:
          entry.tensor = std::make_shared<JoinTensor<HOROVOD_FLOAT16, float>>(
              join_device, response.tensor_sizes()[0]);
          break;
        case HOROVOD_FLOAT32:
          entry.tensor = std::make_shared<JoinTensor<HOROVOD_FLOAT32, float>>(
              join_device, response.tensor_sizes()[0]);
          break;
        case HOROVOD_FLOAT64:
          entry.tensor = std::make_shared<JoinTensor<HOROVOD_FLOAT64, double>>(
              join_device, response.tensor_sizes()[0]);
          break;
        case HOROVOD_BOOL:
          entry.tensor = std::make_shared<JoinTensor<HOROVOD_BOOL, bool>>(
              join_device, response.tensor_sizes()[0]);
          break;
        default:
          throw std::logic_error("Unknown tensor data type");
        }

        entry.output = entry.tensor;
        entry.device = join_device;
        entry.tensor_name = name;
        entries.push_back(std::move(entry));
      }
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

// Pop out all the messages from the queue
void TensorQueue::PopMessagesFromQueue(
    std::deque<Request>& message_queue_buffer) {
  std::lock_guard<std::mutex> guard(mutex_);
  while (!message_queue_.empty()) {
    Request message = message_queue_.front();
    message_queue_.pop();
    message_queue_buffer.push_back(std::move(message));
  }
}

// Push a message to message queue
void TensorQueue::PushMessageToQueue(Request& message) {
  std::lock_guard<std::mutex> guard(mutex_);
  message_queue_.push(std::move(message));
}

// Remove JoinOp tensor from the table and execute the callback
void TensorQueue::RemoveJoinTensor() {
  // Lock on the tensor table.
  std::lock_guard<std::mutex> guard(mutex_);
  auto iter = tensor_table_.find(JOIN_TENSOR_NAME);
  assert(iter != tensor_table_.end());
  auto& e = iter->second;
  Status status;
  e.callback(status);
  tensor_table_.erase(iter);
}

template <DataType DT, class T>
JoinTensor<DT, T>::JoinTensor(int device, int64_t num_elements) {
  num_elements_ = num_elements;
  device_ = device;
  if (device_ == CPU_DEVICE_ID) {
    buffer_data_ = new T[num_elements_];
    for (int i = 0; i < num_elements_; i++) {
      buffer_data_[i] = 0;
    }
  } else {
#if HAVE_CUDA
    cudaSetDevice(device_);
    cudaMalloc(&buffer_data_, size());
    auto tmp = new T[num_elements_];
    for (int i = 0; i < num_elements_; i++) {
      tmp[i] = 0;
    }
    cudaMemcpy(buffer_data_, tmp, size(), cudaMemcpyHostToDevice);
    delete[] tmp;
#else
    throw std::logic_error("Internal error. Requested Join "
                           "with GPU device but not compiled with CUDA.");
#endif
  }
}

template <DataType DT, class T> JoinTensor<DT, T>::~JoinTensor() {
  if (device_ == CPU_DEVICE_ID) {
    delete[] buffer_data_;
  } else {
#if HAVE_CUDA
    cudaFree(buffer_data_);
#else
    throw std::logic_error("Internal error. Requested Join "
                           "with GPU device but not compiled with CUDA.");
#endif
  }
}

template <DataType DT, class T>
const DataType JoinTensor<DT, T>::dtype() const {
  return DT;
}

template <DataType DT, class T>
const TensorShape JoinTensor<DT, T>::shape() const {
  TensorShape shape;
  shape.AddDim(num_elements_);
  return shape;
}

template <DataType DT, class T> const void* JoinTensor<DT, T>::data() const {
  return (void*)buffer_data_;
}

template <DataType DT, class T> int64_t JoinTensor<DT, T>::size() const {
  return num_elements_ * sizeof(T);
}

} // namespace common
} // namespace horovod