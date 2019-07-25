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

#ifndef HOROVOD_STALL_INSPECTOR_H_
#define HOROVOD_STALL_INSPECTOR_H_

#include <chrono>
#include <iostream>
#include <unordered_map>

#include "response_cache.h"

namespace horovod {
namespace common {

class ResponseCache;

class StallInspector {
public:
  StallInspector(ResponseCache& response_cache)
      : response_cache_(response_cache) {}

  bool CheckForStalledTensors(int global_size);

  void InvalidateStalledCachedTensors(CacheCoordinator& cache_coordinator);

  void RecordCachedTensorStart(const std::string& tensor_name);

  void RecordUncachedTensorStart(const std::string& tensor_name, int rank,
                                 int global_size);

  void RemoveCachedTensor(const std::string& tensor_name);

  void RemoveUncachedTensor(const std::string& tensor_name);

  bool ShouldPerformCheck();

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
#endif // HOROVOD_STALL_INSPECTOR_H_
