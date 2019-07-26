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

#include "stall_inspector.h"

#include <map>
#include <unordered_set>

#include "logging.h"

namespace horovod {
namespace common {

bool StallInspector::CheckForStalledTensors(int global_size) {
  bool should_shut_down = false;
  auto now = std::chrono::steady_clock::now();
  std::map<int32_t, std::set<std::string>> missing_ranks;
  std::unordered_set<int32_t> shutdown_ranks;
  std::chrono::seconds stall_warning_time(stall_warning_time_seconds);
  std::chrono::seconds stall_shutdown_time(stall_shutdown_time_seconds);

  if (stall_shutdown_time > std::chrono::seconds(0) &&
      stall_shutdown_time < stall_warning_time) {
    LOG(WARNING) << "HOROVOD_STALL_SHUTDOWN_TIME_SECONDS is less than "
                    "HOROVOD_STALL_CHECK_TIME_SECONDS, will not shutdown.";
    stall_shutdown_time = std::chrono::seconds(0);
  }

  for (auto& m : uncached_tensor_table) {
    auto tensor_name = m.first;
    std::vector<int>& ranks = std::get<0>(m.second);
    std::chrono::steady_clock::time_point start_at = std::get<1>(m.second);
    auto lag = now - start_at;

    if (lag > stall_warning_time) {
      std::unordered_set<int32_t> ready_ranks;
      for (auto rank : ranks) {
        ready_ranks.insert(rank);
      }

      for (int32_t rank = 0; rank < global_size; ++rank) {
        if (ready_ranks.find(rank) == ready_ranks.end()) {
          missing_ranks[rank].insert(tensor_name);
          if (stall_shutdown_time > std::chrono::seconds(0) &&
              lag > stall_shutdown_time) {
            shutdown_ranks.insert(rank);
            should_shut_down = true;
          }
        }
      }
    }
  }

  if (!missing_ranks.empty()) {
    std::stringstream message;
    message << "One or more tensors were submitted to be "
               "reduced, gathered or broadcasted by subset of ranks and "
               "are waiting for remainder of ranks for more than "
            << stall_warning_time.count() << " seconds. "
            << "This may indicate that different ranks are trying to "
               "submit different tensors or that only subset of ranks is "
               "submitting tensors, which will cause deadlock. "
            << std::endl
            << "Stalled ranks:";
    for (auto& kv : missing_ranks) {
      message << std::endl << kv.first;
      if (shutdown_ranks.find(kv.first) != shutdown_ranks.end()) {
        message << "!";
      }

      message << ": [";
      auto it = kv.second.begin();
      message << *it;
      int count = 0;
      while (++it != kv.second.end()) {
        message << ", " << *it;
        if (++count == 5) {
          message << " ...";
          break;
        }
      }

      message << "]";
    }

    if (should_shut_down) {
      message
          << std::endl
          << "One or more rank (marked by \"!\") is stalled for longer than "
          << stall_shutdown_time.count() << " seconds. Will shutdown.";
      LOG(ERROR) << message.str();
    } else {
      LOG(WARNING) << message.str();
    }
  }

  return should_shut_down;
}

void StallInspector::InvalidateStalledCachedTensors(
    CacheCoordinator& cache_coordinator) {
  auto now = std::chrono::steady_clock::now();
  std::chrono::seconds stall_warning_time(stall_warning_time_seconds);

  for (auto& entry : cached_tensor_table) {
    // If pending time for cached tensor exceeds stall_warning_time, mark entry
    // for global removal from cache to trigger stall messaging.
    if (now - entry.second > stall_warning_time) {
      uint32_t cache_bit = response_cache_.peek_cache_bit(entry.first);
      cache_coordinator.record_invalid_bit(cache_bit);
      cache_coordinator.set_uncached_in_queue(true);
    }
  }
}

void StallInspector::RecordUncachedTensorStart(const std::string& tensor_name,
                                               int rank, int global_size) {
  auto table_iter = uncached_tensor_table.find(tensor_name);
  if (table_iter == uncached_tensor_table.end()) {
    std::vector<int> ranks = {rank};
    ranks.reserve(static_cast<unsigned long>(global_size));
    auto now = std::chrono::steady_clock::now();
    uncached_tensor_table.emplace(tensor_name,
                                  std::make_tuple(std::move(ranks), now));
  } else {
    std::vector<int>& ranks = std::get<0>(table_iter->second);
    ranks.push_back(rank);
  }
}

void StallInspector::RecordCachedTensorStart(const std::string& tensor_name) {
  if (perform_stall_check &&
      cached_tensor_table.find(tensor_name) == cached_tensor_table.end()) {
    cached_tensor_table[tensor_name] = std::chrono::steady_clock::now();
  }
}

void StallInspector::RemoveCachedTensor(const std::string& tensor_name) {
  if (perform_stall_check) {
    cached_tensor_table.erase(tensor_name);
  }
}

void StallInspector::RemoveUncachedTensor(const std::string& tensor_name) {
  uncached_tensor_table.erase(tensor_name);
}

bool StallInspector::ShouldPerformCheck() {
  return perform_stall_check &&
         std::chrono::steady_clock::now() - last_stall_check >
             std::chrono::seconds(stall_warning_time_seconds);
}

void StallInspector::UpdateCheckTime() {
  last_stall_check = std::chrono::steady_clock::now();
}

void StallInspector::SetPerformStallCheck(bool value) {
  perform_stall_check = value;
}

void StallInspector::SetStallWarningTimeSeconds(int value) {
  stall_warning_time_seconds = value;
}

void StallInspector::SetStallShutdownTimeSeconds(int value) {
  stall_shutdown_time_seconds = value;
}

} // namespace common
} // namespace horovod
