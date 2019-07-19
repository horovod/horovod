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
#include "stall_inspector.h"

#if HAVE_GLOO
#include "gloo_context.h"
#endif

namespace horovod {
namespace common {

// Forward declaration
class HorovodGlobalState;

class Controller : public std::enable_shared_from_this<Controller> {
public:
  Controller(HorovodGlobalState& global_state);

  // Functions must be overridden by concrete controller
  virtual void Initialize() = 0;

  virtual int GetTypeSize(DataType dtype) = 0;

  virtual void CrossRankBitwiseAnd(std::vector<long long>& bitvector,
                                   int count) = 0;

  virtual void CrossRankBitwiseOr(std::vector<long long>& bitvector,
                                  int count) = 0;

  virtual void SynchronizeParameters() = 0;

  // Concrete controller functions
  ResponseList ComputeResponseList();

  int64_t TensorFusionThresholdBytes();

  void SetRank(const int* ranks, int nrank) {
    for (auto i = 0; i < nrank; ++i) {
      ranks_.push_back(ranks[i]);
    }
  };

  int GetLocalSizeAtCrossRank(int i);

  std::vector<int>& GetRanks() { return ranks_; };
  int GetRank() { return rank_; };
  int GetLocalRank() { return local_rank_; };
  int GetCrossRank() { return cross_rank_; };
  int GetSize() { return size_; };
  int GetLocalSize() { return local_size_; };
  int GetCrossSize() { return cross_size_; };
  const std::vector<int>& GetLocalCommRanks() { return local_comm_ranks_; };
  bool IsCoordinator() const { return is_coordinator_; };
  bool IsHomogeneous() const { return is_homogeneous_; };

  StallInspector& GetStallInspector() { return stall_inspector_; };

protected:
  virtual bool RecvReadyTensors(std::vector<std::string>& ready_to_reduce) = 0;

  virtual void SendFinalTensors(ResponseList& response_list) = 0;

  virtual void SendReadyTensors(RequestList& message_list) = 0;

  virtual void RecvFinalTensors(ResponseList& response_list) = 0;

  Response ConstructResponse(std::string& name);

  void CoordinateCacheAndState(CacheCoordinator& cache_coordinator);

  ResponseList FuseResponses(std::deque<Response>& responses);

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

  // ranks of the horovod world
  std::vector<int> ranks_;

  // COMM_WORLD ranks of processes running on this node.
  std::vector<int> local_comm_ranks_;

  // Numbers of ranks running per node
  std::vector<int> local_sizes_for_cross_rank_;

  uint32_t cache_capacity_ = 1024;

  // Following variables are used for computing ready tensor list
  StallInspector stall_inspector_;

  bool& timeline_enabled_;

  Timeline& timeline_;

  ResponseCache& response_cache_;

  ParameterManager& parameter_manager_;

  std::queue<Request>& message_queue_;

  std::mutex& mutex_;

  std::atomic_bool& shut_down_;

  std::shared_ptr<MessageTable> message_table_;

  TensorTable& tensor_table_;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_CONTROL_MANAGER_H
