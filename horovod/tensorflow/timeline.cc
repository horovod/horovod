// Copyright 2017 Uber Technologies, Inc. All Rights Reserved.
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

#include <sstream>

#include "timeline.h"

namespace horovod {
namespace tensorflow {

void Timeline::Initialize(std::string file_name) {
  file_.open(file_name, std::ios::out | std::ios::trunc);
  if (file_.good()) {
    // Initialize the timeline with '[' character.
    file_ << "[" << std::endl;
    start_time_ = std::chrono::steady_clock::now();
    last_flush_time_ = std::chrono::steady_clock::now();
    initialized_ = true;
  } else {
    std::cerr << "WARNING: Error opening the Horovod Timeline file "
              << file_name << ", will not write a timeline." << std::endl;
  }
}

bool Timeline::Initialized() const { return initialized_; }

// Write event to the Horovod Timeline file.
void Timeline::WriteEvent(const std::string& tensor_name, const char phase,
                          const std::string& op_name, const std::string& args) {
  if (!file_.good()) {
    return;
  }

  // Ensure only single thread writes to file to avoid mangling.
  std::lock_guard<std::mutex> guard(mutex_);

  // Check if file is still good, as it may have errored in competing thread.
  if (!file_.good()) {
    return;
  }

  auto now = std::chrono::steady_clock::now();
  auto ts = now - start_time_;
  auto ts_micros =
      std::chrono::duration_cast<std::chrono::microseconds>(ts).count();

  auto& tensor_idx = tensor_table_[tensor_name];
  if (tensor_idx == 0) {
    tensor_idx = (int)tensor_table_.size();

    // We model tensors as processes. Register metadata for this "pid".
    file_ << "{";
    file_ << "\"name\": \"process_name\"";
    file_ << ", \"ph\": \"M\"";
    file_ << ", \"pid\": " << tensor_idx << "";
    file_ << ", \"args\": {\"name\": \"" << tensor_name << "\"}";
    file_ << "}," << std::endl;
    file_ << "{";
    file_ << "\"name\": \"process_sort_index\"";
    file_ << ", \"ph\": \"M\"";
    file_ << ", \"pid\": " << tensor_idx << "";
    file_ << ", \"args\": {\"sort_index\": " << tensor_idx << "}";
    file_ << "}," << std::endl;
  }

  file_ << "{";
  file_ << "\"ph\": \"" << phase << "\"";
  if (phase != 'E') {
    // Not necessary for ending event.
    file_ << ", \"name\": \"" << op_name << "\"";
  }
  file_ << ", \"ts\": " << ts_micros << "";
  file_ << ", \"pid\": " << tensor_idx << "";
  if (phase == 'X') {
    file_ << ", \"dur\": " << 0 << "";
  }
  if (args != "") {
    file_ << ", \"args\": {" << args << "}";
  }
  file_ << "}," << std::endl;

  if (now - last_flush_time_ >= TIMELINE_FLUSH_TIME) {
    file_.flush();
    last_flush_time_ = now;
  }

  if (!file_.good()) {
    std::cerr << "WARNING: Error writing to the Horovod Timeline after it was "
                 "successfully opened, will stop writing the timeline."
              << std::endl;
  }
}

void Timeline::NegotiateStart(const std::string& tensor_name,
                              const MPIRequest::RequestType request_type) {
  if (!initialized_) {
    return;
  }
  assert(tensor_states_[tensor_name] == TimelineState::UNKNOWN);
  auto event_category =
      "NEGOTIATE_" + MPIRequest::RequestType_Name(request_type);
  WriteEvent(tensor_name, 'B', event_category);
  tensor_states_[tensor_name] = TimelineState::NEGOTIATING;
}

void Timeline::NegotiateRankReady(const std::string& tensor_name,
                                  const int rank) {
  if (!initialized_) {
    return;
  }
  assert(tensor_states_[tensor_name] == TimelineState::NEGOTIATING);
  WriteEvent(tensor_name, 'X', std::to_string(rank));
}

void Timeline::NegotiateEnd(const std::string& tensor_name) {
  if (!initialized_) {
    return;
  }
  assert(tensor_states_[tensor_name] == TimelineState::NEGOTIATING);
  WriteEvent(tensor_name, 'E');
  tensor_states_.erase(tensor_name);
}

void Timeline::Start(const std::string& tensor_name,
                     const MPIResponse::ResponseType response_type) {
  if (!initialized_) {
    return;
  }
  assert(tensor_states_[tensor_name] == TimelineState::UNKNOWN);
  auto event_category = MPIResponse::ResponseType_Name(response_type);
  WriteEvent(tensor_name, 'B', event_category);
  tensor_states_[tensor_name] = TimelineState::TOP_LEVEL;
}

void Timeline::ActivityStart(const std::string& tensor_name, const std::string& activity) {
  if (!initialized_) {
    return;
  }
  assert(tensor_states_[tensor_name] == TimelineState::TOP_LEVEL);
  WriteEvent(tensor_name, 'B', activity);
  tensor_states_[tensor_name] = TimelineState::ACTIVITY;
}

void Timeline::ActivityEnd(const std::string& tensor_name) {
  if (!initialized_) {
    return;
  }
  assert(tensor_states_[tensor_name] == TimelineState::ACTIVITY);
  WriteEvent(tensor_name, 'E');
  tensor_states_[tensor_name] = TimelineState::TOP_LEVEL;
}

namespace {

// Trying to use TensorFlow's default DataType_Name leads to linking issue with
// Protobuf library.
std::string TFDataType_Name(DataType dtype) {
  switch (dtype) {
  case DT_UINT8:
    static const std::string uint8("uint8");
    return uint8;
  case DT_INT8:
    static const std::string int8("int8");
    return int8;
  case DT_UINT16:
    static const std::string uint16("uint16");
    return uint16;
  case DT_INT16:
    static const std::string int16("int16");
    return int16;
  case DT_INT32:
    static const std::string int32("int32");
    return int32;
  case DT_INT64:
    static const std::string int64("int64");
    return int64;
  case DT_FLOAT:
    static const std::string float_("float");
    return float_;
  case DT_DOUBLE:
    static const std::string double_("double");
    return double_;
  default:
    static const std::string unknown("<unknown>");
    return unknown;
  }
}

} // namespace

void Timeline::End(const std::string& tensor_name,
                   const Tensor* output_tensor) {
  if (!initialized_) {
    return;
  }

  // Pop out of current state, if applicable.
  if (tensor_states_[tensor_name] == TimelineState::ACTIVITY) {
    ActivityEnd(tensor_name);
  }

  std::stringstream args;
  if (output_tensor != nullptr) {
    args << "\"dtype\": \"" << TFDataType_Name(output_tensor->dtype()) << "\"";
    args << ", \"shape\": \"" << output_tensor->shape().DebugString() << "\"";
  }
  WriteEvent(tensor_name, 'E', "", args.str());
}

} // namespace tensorflow
} // namespace horovod