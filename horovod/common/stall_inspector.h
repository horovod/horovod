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

#ifndef HOROVOD_STALL_INSPECTOR_H
#define HOROVOD_STALL_INSPECTOR_H

#include <chrono>
#include <iostream>
#include <unordered_map>

#include "response_cache.h"

namespace horovod {
namespace common {

class ResponseCache;

class StallInspector {
public:
  StallInspector() = delete;
  explicit StallInspector(ResponseCache& response_cache)
      : response_cache_(response_cache) {}
  StallInspector(const StallInspector&) = delete;

  // Report Tensors that were submitted to be reduced, gathered or broadcasted
  // by some ranks but not others in the same process set and are waiting for
  // long time to get processed.
  // global_ranks contains the global process rank of each expected process.
  bool CheckForStalledTensors(const std::vector<int>& global_ranks);

  // Invalidate cached tensors that have been pending for a long time.
  void InvalidateStalledCachedTensors(CacheCoordinator& cache_coordinator);

  // Record initial time cached tensor is encountered in queue.
  void RecordCachedTensorStart(const std::string& tensor_name);

  // Record initial time for an uncached tensor is encountered in queue.
  // rank is relative to a process set.
  void RecordUncachedTensorStart(const std::string& tensor_name, int rank,
                                 int process_set_size);

  // Remove timing entry if cached or marked invalid.
  void RemoveCachedTensor(const std::string& tensor_name);

  // Remove timing entry if uncached or marked invalid.
  void RemoveUncachedTensor(const std::string& tensor_name);

  // return whether we should check for stalled tensors.
  bool ShouldPerformCheck();

  // Update last check time.
  void UpdateCheckTime();

  void SetPerformStallCheck(bool value);
  void SetStallWarningTimeSeconds(int value);
  void SetStallShutdownTimeSeconds(int value);

protected:
  // Time point when coordinator last checked for stalled tensors.
  std::chrono::steady_clock::time_point last_stall_check;

  // Flag indicating whether to perform stall tensor check.
  bool perform_stall_check = true;

  // Stall-check warning time
  int stall_warning_time_seconds = 60;

  // Stall-check shutdown time. If perform_stall_check==true and this value
  // is set to be greater than stall_warning_time_seconds, horovod will shut
  // itself down if any rank is stalled for longer than this time.
  int stall_shutdown_time_seconds = 0;

  // Initial time cached tensors are seen in queue. Used for stall message
  // handling.
  std::unordered_map<std::string, std::chrono::steady_clock::time_point>
      cached_tensor_table;

  // Initial time that tensors are seen in the normal message queue. The value
  // consists of a list of ready ranks and the starting point.
  std::unordered_map<
      std::string,
      std::tuple<std::vector<int>, std::chrono::steady_clock::time_point>>
      uncached_tensor_table;

  // Outside dependencies
  ResponseCache& response_cache_;
};

} // namespace common
} // namespace horovod
#endif // HOROVOD_STALL_INSPECTOR_H
