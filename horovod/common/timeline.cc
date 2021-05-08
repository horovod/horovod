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

#if HAVE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#include "logging.h"

namespace horovod {
namespace common {

TimelineWriter::TimelineWriter() {
  std::lock_guard<std::recursive_mutex> guard(writer_mutex_);
  cur_filename_ = "";
  new_pending_filename_ = "";
  pending_status_ = false;
}

void TimelineWriter::SetPendingTimelineFile(const std::string& filename) {
  {
    std::lock_guard<std::recursive_mutex> guard(writer_mutex_);
    if (cur_filename_ == filename) {
      LOG(INFO) << "Current filename for timeline is same as new filename. "
                   "Returning.";
      return;
    }
    new_pending_filename_ = filename;
    pending_status_ = true;
  }
  // block until pending_Status is applied
  while (true) {
    {
      std::lock_guard<std::recursive_mutex> guard(writer_mutex_);
      if (!pending_status_) {
        return;
      }
    }
    if (filename.empty()) {
      LOG(DEBUG) << "StopTimeline is called. Blocking thread since "
                    "pending_status is still true.\n";
    } else {
      LOG(DEBUG) << "StartTimeline is called. Blocking thread since "
                    "pending_status is still true.\n";
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

std::string TimelineWriter::PendingTimelineFile() {
  std::lock_guard<std::recursive_mutex> guard(writer_mutex_);
  return new_pending_filename_;
}

void TimelineWriter::SetTimelineFile(const std::string& filename) {
  // No op if there are pending events in record_queue, let all the event from
  // record queue to get dumped to file

  LOG(INFO) << "Setting TimelineFile. Current file:" << cur_filename_
            << " New filename:" << filename;

  // Close if there existing file open and new file is not same as existing file
  if (!cur_filename_.empty() && cur_filename_ != filename) {

    if (!record_queue_.empty()) {
      LOG(DEBUG) << " SetTimelineFile is no-op as there are events in "
                    "record_queue. Will allow those events to be dumped.";
      active_.exchange(0);
      // Give chance to dump existing event
      return;
    }
    if (file_.is_open()) {
      file_.flush();
      file_.close();
      LOG(INFO) << "Closed timeline file:" << cur_filename_;
    }
    tensor_table_.clear();
  }
  // if new filename is empty, we need to stop accepting activities. This would
  // stopping timeline
  if (filename.empty()) {

    healthy_.exchange(1);
    active_.exchange(0);
    cur_filename_ = filename;
    new_pending_filename_ = cur_filename_;
    // empty filename is special which tells that init the timeline but don't
    // activate it.

    pending_status_ = false;
    LOG(INFO) << "Inited TimelineWriter but active_ is false, since filename "
                 "passed is empty string";
    return;
  }
  // all other cases, need to create a new file
  file_.open(filename, std::ios::out | std::ios::trunc);
  if (file_.good()) {
    LOG(INFO) << "Opened new timeline file" << filename
              << " Set active and healthy to true";
    cur_filename_ = filename;
    new_pending_filename_ = cur_filename_;
    is_new_file_ = true;
    healthy_.exchange(1);
    active_.exchange(1);
  } else {
    LOG(ERROR) << "Error opening the Horovod Timeline file " << filename
               << ", will not write a timeline.";
    healthy_.exchange(1);
    active_.exchange(0);
  }
  pending_status_ = false;
}

void TimelineWriter::Initialize(
    const std::string& file_name,
    std::chrono::steady_clock::time_point start_time_) {
  std::lock_guard<std::recursive_mutex> guard(writer_mutex_);
  if (healthy())
    return;

  SetTimelineFile(file_name);
  auto p1 = std::chrono::system_clock::now();
  auto tt = std::chrono::duration_cast<std::chrono::microseconds>(
                start_time_ - std::chrono::steady_clock::now())
                .count();
  start_time_since_epoch_utc_micros_ =
      std::chrono::duration_cast<std::chrono::microseconds>(
          p1.time_since_epoch())
          .count() +
      tt;
  // Spawn writer thread.
  writer_thread_ = std::thread(&TimelineWriter::WriterLoop, this);
}

void TimelineWriter::Shutdown() {
  active_.exchange(0);
  healthy_.exchange(0);
  try {
    if (writer_thread_.joinable()) {
      writer_thread_.join();
    }
  } catch (const std::system_error& e) {
    LOG(INFO) << "Caught system_error while joining writer thread. Code "
              << e.code() << " meaning " << e.what();
  }

  if (!cur_filename_.empty() && file_.is_open()) {
    file_.flush();
    file_.close();
  }
  tensor_table_.clear();
}

short TimelineWriter::active() { return active_.fetch_and(1); }
short TimelineWriter::healthy() { return healthy_.fetch_and(1); }
void TimelineWriter::EnqueueWriteEvent(const std::string& tensor_name,
                                       char phase, const std::string& op_name,
                                       const std::string& args,
                                       long ts_micros) {
  {
    std::lock_guard<std::recursive_mutex> guard(writer_mutex_);
    if (!active() || !healthy())
      return;
  }
  TimelineRecord r{};
  r.type = TimelineRecordType::EVENT;
  r.tensor_name = tensor_name;
  r.phase = phase;
  r.op_name = op_name;
  r.args = args;
  r.ts_micros = ts_micros;
  while (healthy() && active() && !record_queue_.push(r))
    ;
}

void TimelineWriter::EnqueueWriteMarker(const std::string& name,
                                        long ts_micros) {
  {
    std::lock_guard<std::recursive_mutex> guard(writer_mutex_);
    if (!active() || !healthy())
      return;
  }
  TimelineRecord r{};
  r.type = TimelineRecordType::MARKER;
  r.marker_name = name;
  r.ts_micros = ts_micros;

  while (healthy() && active() && !record_queue_.push(r))
    ;
}

void TimelineWriter::WriteAtFileStart() {
  file_ << "[\n";

  file_ << "{";
  file_ << "\"name\": \"process_name\"";
  // Note name of process can be given in args{"name:"}
  file_ << ", \"ph\": \"M\"";
  file_ << ", \"pid\": " << 0 << "";
  file_ << ", \"args\": {\"start_time_since_epoch_in_micros\":"
        << start_time_since_epoch_utc_micros_ << "}";
  file_ << "}," << std::endl;
  file_ << "{";
  file_ << "\"name\": \"process_sort_index\"";
  file_ << ", \"ph\": \"M\"";
  file_ << ", \"pid\": " << 0 << "";
  file_ << ", \"args\": {\"sort_index\": " << 0 << "}";
  file_ << "}," << std::endl;
}
void TimelineWriter::DoWriteEvent(const TimelineRecord& r) {
  assert(r.type == TimelineRecordType::EVENT);
  if (is_new_file_) {
    WriteAtFileStart();
    is_new_file_ = false;
  } else {
    // last event closed the json ']' , need to seek to one position back and
    // write ',' to continue
    long pos = file_.tellp();
    file_.seekp(pos - 1);
    file_ << ",";
  }
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
  if (!r.args.empty()) {
    file_ << ", \"args\": {" << r.args << "}";
  }
  // We make sure that the events are written always produce valid json file
  file_ << "}]";
}

void TimelineWriter::DoWriteMarker(const TimelineRecord& r) {
  assert(r.type == TimelineRecordType::MARKER);
  if (is_new_file_) {
    WriteAtFileStart();
    is_new_file_ = false;
  } else {
    // last event closed the json ']' , need to seek to one position back and
    // write ',' to continue
    long pos = file_.tellp();
    file_.seekp(pos - 1);
    file_ << ",";
  }
  file_ << "{";
  file_ << "\"ph\": \"i\"";
  file_ << ", \"name\": \"" << r.marker_name << "\"";
  file_ << ", \"ts\": " << r.ts_micros << "";
  file_ << ", \"s\": \"g\"";
  // We make sure that the events are written always produce valid json file
  file_ << "}]";
}

void TimelineWriter::WriterLoop() {
  while (healthy()) {
    while (healthy() && !record_queue_.empty()) {
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
    }
    {
      std::lock_guard<std::recursive_mutex> guard(writer_mutex_);
      // check if we need to call SetTimeLineFile
      if (pending_status_)
        SetTimelineFile(PendingTimelineFile());
      if (active() && !file_.good()) {
        LOG(ERROR) << "Error writing to the Horovod Timeline after it was "
                      "successfully opened, will stop writing the timeline."
                   << " eofbit:" << file_.eof() << " failbit:" << file_.fail()
                   << " badbit" << file_.bad() << "\n";
        active_.exchange(0);
      }
    }
    // Allow scheduler to schedule other work for this core.
    std::this_thread::yield();
  }
}

#if HAVE_NVTX
class TimelineNvtxHandle {
public:
  TimelineNvtxHandle()
      : main_domain_(nvtxDomainCreateA("HorovodTimeline")),
        activity_domain_(nvtxDomainCreateA("HorovodTimelineActivities"))
  { }

  ~TimelineNvtxHandle() {
    Disable();
  }

  void StartRange(const std::string& tensor_name,
                  const std::string& event_category,
                  int64_t tensor_size_payload = -1) {
    if (main_domain_ == nullptr) {
      return;
    }
    const auto message = event_category + " " + tensor_name;
    nvtxRangeId_t range_id = NvtxStartDomainRange(main_domain_, message,
                                                  tensor_size_payload);
    main_tensor_nvtx_range_.emplace(tensor_name, range_id);
  }

  void EndRange(const std::string& tensor_name) {
    if (main_domain_ == nullptr) {
      return;
    }
    auto iter = main_tensor_nvtx_range_.find(tensor_name);
    if (iter != main_tensor_nvtx_range_.end()) {
      auto range_id = iter->second;
      nvtxDomainRangeEnd(main_domain_, range_id);
      main_tensor_nvtx_range_.erase(iter);
    }
  }

  void StartActivityRange(const std::string& tensor_name,
                          const std::string& activity) {
    if (activity_domain_ == nullptr) {
      return;
    }
    const auto message = activity + " " + tensor_name;
    nvtxRangeId_t range_id = NvtxStartDomainRange(activity_domain_, message);
    activity_tensor_nvtx_range_.emplace(tensor_name, range_id);
  }

  void EndActivityRange(const std::string& tensor_name) {
    if (activity_domain_ == nullptr) {
      return;
    }
    auto iter = activity_tensor_nvtx_range_.find(tensor_name);
    if (iter != activity_tensor_nvtx_range_.end()) {
      auto range_id = iter->second;
      nvtxDomainRangeEnd(activity_domain_, range_id);
      activity_tensor_nvtx_range_.erase(iter);
    }
  }

  void Disable() {
    if (main_domain_ != nullptr) {
      nvtxDomainDestroy(main_domain_);
      main_domain_ = nullptr;
    }
    if (activity_domain_ != nullptr) {
      nvtxDomainDestroy(activity_domain_);
      activity_domain_ = nullptr;
    }
  }

private:
  static nvtxRangeId_t NvtxStartDomainRange(nvtxDomainHandle_t domain,
                                            const std::string& message,
                                            int64_t tensor_size_payload = -1) {
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = message.c_str();
    if (tensor_size_payload >= 0) {
      eventAttrib.payloadType = NVTX_PAYLOAD_TYPE_INT64;
      eventAttrib.payload.llValue = tensor_size_payload;
    }

    nvtxRangeId_t range_id = nvtxDomainRangeStartEx(domain, &eventAttrib);
    // contents of eventAttrib and eventAtrrib.message.ascii have been copied
    return range_id;
  }

  nvtxDomainHandle_t main_domain_;      // nullptr if disabled
  nvtxDomainHandle_t activity_domain_;  // nullptr if disabled

  std::unordered_map<std::string, nvtxRangeId_t> main_tensor_nvtx_range_;
  std::unordered_map<std::string, nvtxRangeId_t> activity_tensor_nvtx_range_;
};
#else // HAVE_NVTX
class TimelineNvtxHandle {
public:
  void StartRange(const std::string& tensor_name,
                  const std::string& event_category,
                  int64_t tensor_size_payload = -1) {}
  void EndRange(const std::string& tensor_name) {}
  void StartActivityRange(const std::string& tensor_name,
                          const std::string& activity) {}
  void EndActivityRange(const std::string& tensor_name) {}
  void Disable() {}
};
#endif // HAVE_NVTX


Timeline::Timeline() : nvtx_handle_(new TimelineNvtxHandle) {}

void Timeline::Initialize(const std::string& file_name,
                          unsigned int horovod_size) {
  if (Initialized()) {
    return;
  }
  start_time_ = std::chrono::steady_clock::now();

  // Start the writer.
  writer_.Initialize(file_name, start_time_);

  // Initialize if we were able to open the file successfully.
  initialized_.exchange(writer_.healthy());

  // Pre-initialize the string representation for each rank.
  rank_strings_ = std::vector<std::string>(horovod_size);
  for (unsigned int i = 0; i < horovod_size; i++) {
    rank_strings_[i] = std::to_string(i);
  }
}

void Timeline::Shutdown() {
  std::lock_guard<std::recursive_mutex> guard(mutex_);
  initialized_.exchange(0);
  writer_.Shutdown();
  tensor_states_.clear();
}

Timeline::~Timeline() = default;

long Timeline::TimeSinceStartMicros() const {
  auto now = std::chrono::steady_clock::now();
  auto ts = now - start_time_;
  return std::chrono::duration_cast<std::chrono::microseconds>(ts).count();
}

// Write event to the Horovod Timeline file.
void Timeline::WriteEvent(const std::string& tensor_name, const char phase,
                          const std::string& op_name, const std::string& args) {
  if (!Initialized() || !writer_.active()) {
    return;
  }
  auto ts_micros = TimeSinceStartMicros();
  writer_.EnqueueWriteEvent(tensor_name, phase, op_name, args, ts_micros);
}

void Timeline::WriteMarker(const std::string& name) {
  if (!Initialized() || !writer_.active()) {
    return;
  }
  auto ts_micros = TimeSinceStartMicros();
  writer_.EnqueueWriteMarker(name, ts_micros);
}

void Timeline::NegotiateStart(const std::string& tensor_name,
                              const Request::RequestType request_type) {
  if (!Initialized() || !writer_.active()) {
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
  auto event_category = "NEGOTIATE_" + Request::RequestType_Name(request_type);
  nvtx_handle_->StartRange(tensor_name, event_category);
  WriteEvent(tensor_name, 'B', event_category);
  tensor_states_[tensor_name] = TimelineState::NEGOTIATING;
}

void Timeline::NegotiateRankReady(const std::string& tensor_name,
                                  const int rank) {
  if (!Initialized() || !writer_.active()) {
    return;
  }

  std::lock_guard<std::recursive_mutex> guard(mutex_);
  assert(tensor_states_[tensor_name] == TimelineState::NEGOTIATING);
  WriteEvent(tensor_name, 'X', rank_strings_[rank]);
}

void Timeline::NegotiateEnd(const std::string& tensor_name) {
  if (!Initialized() || !writer_.active()) {
    return;
  }

  std::lock_guard<std::recursive_mutex> guard(mutex_);
  assert(tensor_states_[tensor_name] == TimelineState::NEGOTIATING);
  nvtx_handle_->EndRange(tensor_name);
  WriteEvent(tensor_name, 'E');
  tensor_states_.erase(tensor_name);
}

void Timeline::Start(const std::string& tensor_name,
                     const Response::ResponseType response_type,
                     int64_t tensor_size) {
  if (!Initialized() || !writer_.active()) {
    return;
  }

  std::lock_guard<std::recursive_mutex> guard(mutex_);
  assert(tensor_states_[tensor_name] == TimelineState::UNKNOWN);
  auto event_category = Response::ResponseType_Name(response_type);
  WriteEvent(tensor_name, 'B', event_category);
  nvtx_handle_->StartRange(tensor_name, event_category, tensor_size);
  tensor_states_[tensor_name] = TimelineState::TOP_LEVEL;
}

void Timeline::ActivityStartAll(const std::vector<TensorTableEntry>& entries,
                                const std::string& activity) {
  if (!Initialized() || !writer_.active()) {
    return;
  }
  for (auto& e : entries) {
    ActivityStart(e.tensor_name, activity);
  }
}

void Timeline::ActivityStart(const std::string& tensor_name,
                             const std::string& activity) {
  if (!Initialized() || !writer_.active()) {
    return;
  }

  std::lock_guard<std::recursive_mutex> guard(mutex_);
  assert(tensor_states_[tensor_name] == TimelineState::TOP_LEVEL);
  nvtx_handle_->StartActivityRange(tensor_name, activity);
  WriteEvent(tensor_name, 'B', activity);
  tensor_states_[tensor_name] = TimelineState::ACTIVITY;
}

void Timeline::ActivityEndAll(const std::vector<TensorTableEntry>& entries) {
  for (auto& e : entries) {
    ActivityEnd(e.tensor_name);
  }
}

void Timeline::ActivityEnd(const std::string& tensor_name) {
  if (!Initialized() || !writer_.active()) {
    return;
  }

  std::lock_guard<std::recursive_mutex> guard(mutex_);
  assert(tensor_states_[tensor_name] == TimelineState::ACTIVITY);
  nvtx_handle_->EndActivityRange(tensor_name);
  WriteEvent(tensor_name, 'E');
  tensor_states_[tensor_name] = TimelineState::TOP_LEVEL;
}

void Timeline::End(const std::string& tensor_name,
                   const std::shared_ptr<Tensor>& output_tensor) {
  if (!Initialized() || !writer_.active()) {
    return;
  }

  std::lock_guard<std::recursive_mutex> guard(mutex_);

  // Pop out of current state, if applicable.
  if (tensor_states_[tensor_name] == TimelineState::ACTIVITY) {
    ActivityEnd(tensor_name);
  }

  nvtx_handle_->EndRange(tensor_name);

  std::stringstream args;
  if (output_tensor != nullptr) {
    args << "\"dtype\": \"" << DataType_Name(output_tensor->dtype()) << "\"";
    args << ", \"shape\": \"" << output_tensor->shape().DebugString() << "\"";
  }
  WriteEvent(tensor_name, 'E', "", args.str());
  tensor_states_.erase(tensor_name);
}

void Timeline::MarkCycleStart() {
  if (!Initialized() || !writer_.active()) {
    return;
  }

  std::lock_guard<std::recursive_mutex> guard(mutex_);
  WriteMarker("CYCLE_START");
}

void Timeline::SetPendingTimelineFile(const std::string& filename) {
  writer_.SetPendingTimelineFile(filename);
  LOG(INFO) << "Set pending timeline file to " << filename;
}

void Timeline::DisableNvtx() {
  nvtx_handle_->Disable();
}

void TimelineController::SetTimelineEnabled(bool value) {
  std::lock_guard<std::recursive_mutex> guard(timeline_mutex_);
  timeline_enabled_pending_ = value;
  timeline_enabled_ = value;
}

void TimelineController::SetTimelineEnabledPending(bool value) {
  std::lock_guard<std::recursive_mutex> guard(timeline_mutex_);
  timeline_enabled_pending_ = value;
}

void TimelineController::SetMarkCyclesInTimelinePending(bool value) {
  std::lock_guard<std::recursive_mutex> guard(timeline_mutex_);
  mark_cycles_in_timeline_pending_ = value;
}

void TimelineController::SynchronizeTimelineEnabled() {
  std::lock_guard<std::recursive_mutex> guard(timeline_mutex_);
  timeline_enabled_ = timeline_enabled_pending_;
}

bool TimelineController::TimelineEnabled() {
  std::lock_guard<std::recursive_mutex> guard(timeline_mutex_);
  return timeline_enabled_;
}

bool TimelineController::TimelineEnabledPending() {
  std::lock_guard<std::recursive_mutex> guard(timeline_mutex_);
  return timeline_enabled_pending_;
}

bool TimelineController::MarkCyclesInTimelinePending() {
  std::lock_guard<std::recursive_mutex> guard(timeline_mutex_);
  return mark_cycles_in_timeline_pending_;
}

} // namespace common
} // namespace horovod
