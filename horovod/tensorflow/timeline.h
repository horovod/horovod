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

#ifndef HOROVOD_TIMELINE_H
#define HOROVOD_TIMELINE_H

#include <chrono>
#include <fstream>
#include <iostream>
#include <unordered_map>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "mpi_message.h"

using namespace tensorflow;

namespace horovod {
namespace tensorflow {

// How frequently Horovod Timeline should be flushed to disk.
#define TIMELINE_FLUSH_TIME std::chrono::seconds(1)

enum TimelineState {
  UNKNOWN,
  NEGOTIATING,
  TOP_LEVEL,
  ACTIVITY
};

// Writes timeline in Chrome Tracing format. Timeline spec is from:
// https://github.com/catapult-project/catapult/tree/master/tracing
class Timeline {
public:
  void Initialize(std::string file_name);
  bool Initialized() const;
  void NegotiateStart(const std::string& tensor_name,
                      const MPIRequest::RequestType request_type);
  void NegotiateRankReady(const std::string& tensor_name, const int rank);
  void NegotiateEnd(const std::string& tensor_name);
  void Start(const std::string& tensor_name,
             const MPIResponse::ResponseType response_type);
  void ActivityStart(const std::string& tensor_name, const std::string& activity);
  void ActivityEnd(const std::string& tensor_name);
  void End(const std::string& tensor_name, const Tensor* output_tensor);

private:
  void WriteEvent(const std::string& tensor_name, const char phase,
                  const std::string& op_name = "",
                  const std::string& args = "");

  // Boolean flag indicating whether Timeline was initialized (and thus should
  // be recorded).
  bool initialized_ = false;

  // Time point when Horovod was started.
  std::chrono::steady_clock::time_point start_time_;

  // Last time stream was flushed.
  std::chrono::steady_clock::time_point last_flush_time_;

  // Timeline file.
  std::ofstream file_;

  // A mutex that guards timeline file from concurrent access.
  std::mutex mutex_;

  // Mapping of tensor names to indexes. It is used to reduce size of the
  // timeline file.
  std::unordered_map<std::string, int> tensor_table_;

  // Current state of each tensor in the timeline.
  std::unordered_map<std::string, TimelineState> tensor_states_;
};

} // namespace tensorflow
} // namespace horovod

#endif // HOROVOD_TIMELINE_H
