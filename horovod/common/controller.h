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

#ifndef HOROVOD_CONTROL_MANAGER_H
#define HOROVOD_CONTROL_MANAGER_H

#include <iostream>
#include <queue>
#include <vector>

#include "half.h"
#include "message_table.h"
#include "parameter_manager.h"
#include "response_cache.h"

#if HAVE_GLOO
#include "gloo_context.h"
#endif

namespace horovod {
namespace common {

class Controller {
public:
  // Functions must be overridden by concrete controller
  virtual void Initialize() = 0;

  virtual int GetTypeSize(DataType dtype) = 0;

  virtual void CrossRankBitwiseAnd(std::vector<long long>& bitvector,
                                   int count) = 0;

  virtual void CrossRankBitwiseOr(std::vector<long long>& bitvector,
                                  int count) = 0;

  virtual void SynchronizeParameters(ParameterManager& para_manager) = 0;

  // General controller functions

  ResponseList PrepareForOps(bool timeline_enabled, Timeline& timeline,
                             ResponseCache& response_cache,
                             ParameterManager& param_manager,
                             std::queue<Request>& state_message_queue,
                             std::mutex& mutex, std::atomic_bool& shut_down,
                             std::shared_ptr<MessageTable> message_table,
                             TensorTable& tensor_table);
  int64_t TensorFusionThresholdBytes(ParameterManager& parameter_manager);

  void SetRank(const int* ranks, int nrank) {
    for (auto i = 0; i < nrank; ++i) {
      ranks_.push_back(ranks[i]);
    }
  };

  std::vector<int>& GetRanks() { return ranks_; };
  int GetRank() { return rank_; };
  int GetLocalRank() { return local_rank_; };
  int GetCrossRank() { return cross_rank_; };
  int GetSize() { return size_; };
  int GetLocalSize() { return local_size_; };
  int GetCrossSize() { return cross_size_; };
  int GetLocalSizeAtNode(int i) { return local_sizes_[i]; };
  const std::vector<int>& GetLocalCommRanks() { return local_comm_ranks_; };
  bool IsCoordinator() const { return is_coordinator_; };
  bool IsHomogeneous() const { return is_homogeneous_; };

  // Struct for stall checking related variables.
  struct StallInspector {
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
        cache_tensor_start;

  };

  struct StallInspector& GetStallInspector() { return stall_inspector_; };

protected:
  virtual bool RecvReadyTensors(std::vector<std::string>& ready_to_reduce,
                                std::shared_ptr<MessageTable> message_table,
                                Timeline& timeline) = 0;

  virtual void SendFinalTensors(ResponseList& response_list) = 0;

  virtual void SendReadyTensors(RequestList& message_list) = 0;

  virtual void RecvFinalTensors(ResponseList& response_list) = 0;

  Response ConstructResponse(std::shared_ptr<MessageTable> message_table,
                             std::string& name);

  bool CheckForStalledTensors(std::shared_ptr<MessageTable> message_table);

  void CoordinateCacheAndState(CacheCoordinator& cache_coordinator,
                               bool timeline_enabled, Timeline& timeline,
                               ResponseCache& response_cache);

  void InvalidateStalledCachedTensors(CacheCoordinator& cache_coordinator,
                                      ResponseCache& response_cache);

  ResponseList FuseResponses(std::deque<Response>& responses, std::mutex& mutex,
                             TensorTable& tensor_table,
                             ParameterManager& param_manager);

  int64_t
  TotalByteSizeOfAllgatherOutput(const std::vector<int64_t>& tensor_sizes,
                                 const TensorTableEntry& entry);

  int rank_ = 0;
  int local_rank_ = 0;
  int cross_rank_ = 0;
  int size_ = 1;
  int local_size_ = 1;
  int cross_size_ = 1;
  bool is_coordinator_ = false;
  bool is_homogeneous_ = false;

  // ranks of the mpi world
  std::vector<int> ranks_;

  // COMM_WORLD ranks of processes running on this node.
  std::vector<int> local_comm_ranks_;

  // Numbers of ranks running per node
  std::vector<int> local_sizes_;

  uint32_t cache_capacity_ = 1024;

  struct StallInspector stall_inspector_;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_CONTROL_MANAGER_H
