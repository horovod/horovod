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

#include "timeline.h"

#include <cassert>
#include <chrono>
#include <sstream>
#include <thread>

#include "logging.h"

namespace horovod {
namespace common {

void TimelineWriter::Initialize(std::string file_name) {
  file_.open(file_name, std::ios::out | std::ios::trunc);
  if (file_.good()) {
    // Initialize the timeline with '[' character.
    file_ << "[\n";
    healthy_ = true;

    // Spawn writer thread.
    std::thread writer_thread(&TimelineWriter::WriterLoop, this);
    writer_thread.detach();
  } else {
    LOG(ERROR) << "Error opening the Horovod Timeline file " << file_name
               << ", will not write a timeline.";
  }
}

void TimelineWriter::EnqueueWriteEvent(const std::string& tensor_name,
                                       char phase, const std::string& op_name,
                                       const std::string& args,
                                       long ts_micros) {
  TimelineRecord r{};
  r.type = TimelineRecordType::EVENT;
  r.tensor_name = tensor_name;
  r.phase = phase;
  r.op_name = op_name;
  r.args = args;
  r.ts_micros = ts_micros;

  while (healthy_ && !record_queue_.push(r))
    ;
}

void TimelineWriter::EnqueueWriteMarker(const std::string& name,
                                        long ts_micros) {
  TimelineRecord r{};
  r.type = TimelineRecordType::MARKER;
  r.marker_name = name;
  r.ts_micros = ts_micros;

  while (healthy_ && !record_queue_.push(r))
    ;
}

void TimelineWriter::DoWriteEvent(const TimelineRecord& r) {
  assert(r.type == TimelineRecordType::EVENT);

  auto& tensor_idx = tensor_table_[r.tensor_name];
  if (tensor_idx == 0) {
    tensor_idx = (int)tensor_table_.size();

    // We model tensors as processes. Register metadata for this "pid".
    file_ << "{";
    file_ << "\"name\": \"process_name\"";
    file_ << ", \"ph\": \"M\"";
    file_ << ", \"pid\": " << tensor_idx << "";
    file_ << ", \"args\": {\"name\": \"" << r.tensor_name << "\"}";
    file_ << "}," << std::endl;
    file_ << "{";
    file_ << "\"name\": \"process_sort_index\"";
    file_ << ", \"ph\": \"M\"";
    file_ << ", \"pid\": " << tensor_idx << "";
    file_ << ", \"args\": {\"sort_index\": " << tensor_idx << "}";
    file_ << "}," << std::endl;
  }

  file_ << "{";
  file_ << "\"ph\": \"" << r.phase << "\"";
  if (r.phase != 'E') {
    // Not necessary for ending event.
    file_ << ", \"name\": \"" << r.op_name << "\"";
  }
  file_ << ", \"ts\": " << r.ts_micros << "";
  file_ << ", \"pid\": " << tensor_idx << "";
  if (r.phase == 'X') {
    file_ << ", \"dur\": " << 0 << "";
  }
  if (r.args != "") {
    file_ << ", \"args\": {" << r.args << "}";
  }
  file_ << "}," << std::endl;
}

void TimelineWriter::DoWriteMarker(const TimelineRecord& r) {
  assert(r.type == TimelineRecordType::MARKER);

  file_ << "{";
  file_ << "\"ph\": \"i\"";
  file_ << ", \"name\": \"" << r.marker_name << "\"";
  file_ << ", \"ts\": " << r.ts_micros << "";
  file_ << ", \"s\": \"g\"";
  file_ << "}," << std::endl;
}

void TimelineWriter::WriterLoop() {
  while (healthy_) {
    while (healthy_ && !record_queue_.empty()) {
      auto& r = record_queue_.front();
      switch (r.type) {
      case TimelineRecordType::EVENT:
        DoWriteEvent(r);
        break;
      case TimelineRecordType::MARKER:
        DoWriteMarker(r);
        break;
      default:
        throw std::logic_error("Unknown event type provided.");
      }
      record_queue_.pop();

      if (!file_.good()) {
        LOG(ERROR) << "Error writing to the Horovod Timeline after it was "
                      "successfully opened, will stop writing the timeline.";
        healthy_ = false;
      }
    }

    // Allow scheduler to schedule other work for this core.
    std::this_thread::yield();
  }
}

void Timeline::Initialize(std::string file_name, unsigned int horovod_size) {
  if (initialized_) {
    return;
  }

  // Start the writer.
  writer_.Initialize(std::move(file_name));

  // Initialize if we were able to open the file successfully.
  initialized_ = writer_.IsHealthy();

  // Pre-initialize the string representation for each rank.
  rank_strings_ = std::vector<std::string>(horovod_size);
  for (unsigned int i = 0; i < horovod_size; i++) {
    rank_strings_[i] = std::to_string(i);
  }
}

long Timeline::TimeSinceStartMicros() const {
  auto now = std::chrono::steady_clock::now();
  auto ts = now - start_time_;
  return std::chrono::duration_cast<std::chrono::microseconds>(ts).count();
}

// Write event to the Horovod Timeline file.
void Timeline::WriteEvent(const std::string& tensor_name, const char phase,
                          const std::string& op_name, const std::string& args) {
  auto ts_micros = TimeSinceStartMicros();
  writer_.EnqueueWriteEvent(tensor_name, phase, op_name, args, ts_micros);
}

void Timeline::WriteMarker(const std::string& name) {
  auto ts_micros = TimeSinceStartMicros();
  writer_.EnqueueWriteMarker(name, ts_micros);
}

void Timeline::NegotiateStart(const std::string& tensor_name,
                              const Request::RequestType request_type) {
  if (!initialized_) {
    return;
  }

  std::lock_guard<std::recursive_mutex> guard(mutex_);
  // Note: Need to enable repeated calls to this routine during negotiate
  // phase. Repeated calls can occur if a cached response initiates the
  // negotiation phase, either due to multiple cycles with cache misses on
  // some worker, or if the response is evicted from the cache before
  // completion and its handling proceeds to the default communication path.
  // First call takes precedence.
  if (tensor_states_[tensor_name] == TimelineState::NEGOTIATING) {
    return;
  }

  assert(tensor_states_[tensor_name] == TimelineState::UNKNOWN);
  auto event_category =
      "NEGOTIATE_" + Request::RequestType_Name(request_type);
  WriteEvent(tensor_name, 'B', event_category);
  tensor_states_[tensor_name] = TimelineState::NEGOTIATING;
}

void Timeline::NegotiateRankReady(const std::string& tensor_name,
                                  const int rank) {
  if (!initialized_) {
    return;
  }

  std::lock_guard<std::recursive_mutex> guard(mutex_);
  assert(tensor_states_[tensor_name] == TimelineState::NEGOTIATING);
  WriteEvent(tensor_name, 'X', rank_strings_[rank]);
}

void Timeline::NegotiateEnd(const std::string& tensor_name) {
  if (!initialized_) {
    return;
  }

  std::lock_guard<std::recursive_mutex> guard(mutex_);
  assert(tensor_states_[tensor_name] == TimelineState::NEGOTIATING);
  WriteEvent(tensor_name, 'E');
  tensor_states_.erase(tensor_name);
}

void Timeline::Start(const std::string& tensor_name,
                     const Response::ResponseType response_type) {
  if (!initialized_) {
    return;
  }

  std::lock_guard<std::recursive_mutex> guard(mutex_);
  assert(tensor_states_[tensor_name] == TimelineState::UNKNOWN);
  auto event_category = Response::ResponseType_Name(response_type);
  WriteEvent(tensor_name, 'B', event_category);
  tensor_states_[tensor_name] = TimelineState::TOP_LEVEL;
}

void Timeline::ActivityStartAll(const std::vector<TensorTableEntry>& entries,
                                const std::string& activity) {
  for (auto& e : entries) {
    ActivityStart(e.tensor_name, activity);
  }
}

void Timeline::ActivityStart(const std::string& tensor_name,
                             const std::string& activity) {
  if (!initialized_) {
    return;
  }

  std::lock_guard<std::recursive_mutex> guard(mutex_);
  assert(tensor_states_[tensor_name] == TimelineState::TOP_LEVEL);
  WriteEvent(tensor_name, 'B', activity);
  tensor_states_[tensor_name] = TimelineState::ACTIVITY;
}

void Timeline::ActivityEndAll(const std::vector<TensorTableEntry>& entries) {
  for (auto& e : entries) {
    ActivityEnd(e.tensor_name);
  }
}

void Timeline::ActivityEnd(const std::string& tensor_name) {
  if (!initialized_) {
    return;
  }

  std::lock_guard<std::recursive_mutex> guard(mutex_);
  assert(tensor_states_[tensor_name] == TimelineState::ACTIVITY);
  WriteEvent(tensor_name, 'E');
  tensor_states_[tensor_name] = TimelineState::TOP_LEVEL;
}

void Timeline::End(const std::string& tensor_name,
                   const std::shared_ptr<Tensor> tensor) {
  if (!initialized_) {
    return;
  }

  std::lock_guard<std::recursive_mutex> guard(mutex_);

  // Pop out of current state, if applicable.
  if (tensor_states_[tensor_name] == TimelineState::ACTIVITY) {
    ActivityEnd(tensor_name);
  }

  std::stringstream args;
  if (tensor != nullptr) {
    args << "\"dtype\": \"" << DataType_Name(tensor->dtype()) << "\"";
    args << ", \"shape\": \"" << tensor->shape().DebugString() << "\"";
  }
  WriteEvent(tensor_name, 'E', "", args.str());
}

void Timeline::MarkCycleStart() {
  if (!initialized_) {
    return;
  }

  std::lock_guard<std::recursive_mutex> guard(mutex_);
  WriteMarker("CYCLE_START");
}

} // namespace common
} // namespace horovod
