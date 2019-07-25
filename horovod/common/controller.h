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
#include "parameter_manager.h"
#include "response_cache.h"
#include "stall_inspector.h"
#include "tensor_queue.h"
#include "timeline.h"

#if HAVE_GLOO
#include "gloo_context.h"
#endif

namespace horovod {
namespace common {

// Forward declaration
class HorovodGlobalState;

using MessageTable = std::unordered_map<std::string, std::vector<Request>>;

class Controller : public std::enable_shared_from_this<Controller> {
public:
  Controller(ResponseCache& response_cache, TensorQueue& tensor_queue,
             bool& timeline_enabled, Timeline& timeline,
             ParameterManager& parameter_manager);

  // Functions must be overridden by concrete controller
  virtual void Initialize() = 0;

  virtual int GetTypeSize(DataType dtype) = 0;

  virtual void CrossRankBitwiseAnd(std::vector<long long>& bitvector,
                                   int count) = 0;

  virtual void CrossRankBitwiseOr(std::vector<long long>& bitvector,
                                  int count) = 0;

  virtual void SynchronizeParameters() = 0;

  // Concrete controller functions
  ResponseList ComputeResponseList(std::atomic_bool& shut_down);

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
  virtual void RecvReadyTensors(std::vector<std::string>& ready_to_reduce,
                                std::vector<RequestList>& ready_list) = 0;

  virtual void SendFinalTensors(ResponseList& response_list) = 0;

  virtual void SendReadyTensors(RequestList& message_list) = 0;

  virtual void RecvFinalTensors(ResponseList& response_list) = 0;

  Response ConstructResponse(std::string& name);

  void CoordinateCacheAndState(CacheCoordinator& cache_coordinator);

  ResponseList FuseResponses(std::deque<Response>& responses);

  int64_t
  TotalByteSizeOfAllgatherOutput(const std::vector<int64_t>& tensor_sizes,
                                 const TensorTableEntry& entry);

  // Store the Request for a name, and return whether the total count of
  // Requests for that tensor is now equal to the HOROVOD size (and thus we are
  // ready to reduce the tensor).
  bool IncrementTensorCount(const Request& msg);

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

  StallInspector stall_inspector_;

  // Only exists on the coordinator node (rank zero). Maintains a count of
  // how many nodes are ready to allreduce every tensor (keyed by tensor
  // name) and time point when tensor started allreduce op.
  MessageTable message_table_;

  // Outside dependencies
  TensorQueue& tensor_queue_;

  bool& timeline_enabled_;

  Timeline& timeline_;

  ResponseCache& response_cache_;

  ParameterManager& parameter_manager_;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_CONTROL_MANAGER_H
