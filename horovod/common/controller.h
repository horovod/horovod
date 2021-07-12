// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "global_state.h"
#include "group_table.h"
#include "parameter_manager.h"
#include "response_cache.h"
#include "stall_inspector.h"
#include "tensor_queue.h"
#include "timeline.h"

namespace horovod {
namespace common {

using MessageTable = std::unordered_map<std::string, std::vector<Request>>;

class Controller : public std::enable_shared_from_this<Controller> {
public:
  Controller(ResponseCache& response_cache, TensorQueue& tensor_queue,
             Timeline& timeline, ParameterManager& parameter_manager,
             GroupTable& group_table, TimelineController& timeline_controller);

  Controller(const Controller&) = delete;

  void Initialize();

  virtual int GetTypeSize(DataType dtype) = 0;

  virtual void CrossRankBitwiseAnd(std::vector<long long>& bitvector,
                                   int count) = 0;

  virtual void CrossRankBitwiseOr(std::vector<long long>& bitvector,
                                  int count) = 0;

  virtual void Bcast(void* buffer, size_t size, int root_rank,
                     Communicator communicator) = 0;

  virtual void AlltoallGetRecvSplits(const std::vector<int32_t>& splits,
                                     std::vector<int32_t>& recvsplits) = 0;

  virtual void Barrier(Communicator communicator) = 0;

  virtual void Allgather2Ints(std::array<int, 2> values,
                              std::vector<int>& recv_values) = 0;

  //
  // Concrete controller functions
  //

  void SynchronizeParameters();

  // This function performs all the preparation work for workers to agree
  // on what tensors to be all-reduced or all-gathered. The output is a
  // response list that includes all tensors that are ready.
  //
  // The coordinator follows a master-worker paradigm. Rank zero acts
  // as the master (the "coordinator"), whereas all other ranks are simply
  // workers. Each worker maintains a cache of tensors that are previously
  // broadcasted as ready by other ranks. If the cache covers all incoming
  // messages, there's no need for workers to do additional communications.
  // Otherwise, workers will communicate with each other to agree on what
  // tensors to be processed. The communication performs as following:
  //
  //      a) The workers send a Request to the coordinator, indicating what
  //      they would like to do (which tensor they would like to gather and
  //      reduce, as well as their shape and type). They repeat this for every
  //      tensor that they would like to operate on.
  //
  //      b) The workers send an empty "DONE" message to the coordinator to
  //      indicate that there are no more tensors they wish to operate on.
  //
  //      c) The coordinator receives the Requests from the workers, as well
  //      as from its own TensorFlow ops, and stores them in a request table.
  //      The coordinator continues to receive Request messages until it has
  //      received GLOBAL_SIZE number of empty "DONE" messages.
  //
  //      d) The coordinator finds all tensors that are ready to be reduced,
  //      gathered, or all operations that result in an error. For each of
  //      those, it sends a Response to all the workers. When no more
  //      Responses are available, it sends a "DONE" response to the workers.
  //      If the process is being shutdown, it instead sends a "SHUTDOWN"
  //      response.
  //
  //      e) The workers listen for Response messages, processing each one by
  //      doing the required reduce or gather, until they receive a "DONE"
  //      response from the coordinator. At that point, the tick ends.
  //      If instead of "DONE" they receive "SHUTDOWN", they mark it in the
  //      response list.
  ResponseList ComputeResponseList(bool this_process_requested_shutdown,
                                   HorovodGlobalState& state,
                                   ProcessSet& process_set);

  // Get current tensors fusion threshold.
  int64_t TensorFusionThresholdBytes();

  int GetLocalSizeAtCrossRank(int i);

  int GetRank() const { return rank_; };
  int GetLocalRank() const { return local_rank_; };
  int GetCrossRank() const { return cross_rank_; };
  int GetSize() const { return size_; };
  int GetLocalSize() const { return local_size_; };
  int GetCrossSize() const { return cross_size_; };
  const std::vector<int>& GetGlobalRanks() const { return global_ranks_; }
  const std::unordered_map<int, int>& GetGlobalRankToControllerRank() const {
    return global_rank_to_controller_rank_;
  }
  const std::vector<int>& GetLocalCommRanks() const {
    return local_comm_ranks_;
  };
  bool IsCoordinator() const { return is_coordinator_; };
  bool IsHomogeneous() const { return is_homogeneous_; };
  bool IsInitialized() const { return is_initialized_; }
  StallInspector& GetStallInspector() { return stall_inspector_; };

protected:
  //
  // Functions must be overridden by concrete controller
  //

  virtual void DoInitialization() = 0;

  // For rank 0 to receive other ranks' ready tensors.
  virtual void RecvReadyTensors(std::vector<std::string>& ready_to_reduce,
                                std::vector<RequestList>& ready_list) = 0;

  // For other ranks to send their ready tensors to rank 0
  virtual void SendReadyTensors(RequestList& message_list) = 0;

  // For rank 0 to send final tensors ready to be allreduced/allgathered to other ranks.
  virtual void SendFinalTensors(ResponseList& response_list) = 0;

  // For other ranks to receive final ready tensors.
  virtual void RecvFinalTensors(ResponseList& response_list) = 0;

  // Once a tensor is ready to be reduced, the coordinator sends a Response
  // instructing all ranks to start the reduction to all ranks. The Response
  // also contains error messages in case the submitted Requests were not
  // valid (for example, contained mismatched shapes or types).
  // Constructing the Response, thus, requires a whole lot of error checking.
  Response ConstructResponse(const std::string& name, int joined_size = 0);

  // Routine to sync cache hit and invalid bit sets across workers.
  // Also determines global shutdown state and whether uncached requests
  // exist on any worker.
  void CoordinateCacheAndState(CacheCoordinator& cache_coordinator);

  void FuseResponses(std::deque<Response>& responses,
                     HorovodGlobalState& state,
                     ResponseList& response_list);

  // Return the total byte size of the final allgathered output tensor
  int64_t
  TotalByteSizeOfAllgatherOutput(const std::vector<int64_t>& tensor_sizes,
                                 const TensorTableEntry& entry);

  // Store the Request for a name, and return whether the total count of
  // Requests for that tensor is now equal to the HOROVOD size (and thus we are
  // ready to reduce the tensor).
  bool IncrementTensorCount(const Request& msg, int joined_size = 0);

  bool is_initialized_ = false;

  int rank_ = 0;
  int local_rank_ = 0;
  int cross_rank_ = 0;
  int size_ = 1;
  int local_size_ = 1;
  int cross_size_ = 1;
  bool is_coordinator_ = false;
  bool is_homogeneous_ = false;

  // Global rank of each process in the set associated to this controller.
  std::vector<int> global_ranks_;

  // Map (global rank) -> (process set controller rank) for each process in this
  // set.
  std::unordered_map<int,int> global_rank_to_controller_rank_;

  // Controller process set ranks of processes running on this node.
  std::vector<int> local_comm_ranks_;

  // Numbers of ranks running per node
  std::vector<int> local_sizes_for_cross_rank_;

  uint32_t cache_capacity_ = 1024;

  StallInspector stall_inspector_;

  // Only exists on the coordinator node (rank zero). Maintains a vector of
  // requests to allreduce every tensor (keyed by tensor name).
  MessageTable message_table_;

  // Outside dependencies
  TensorQueue& tensor_queue_;

  Timeline& timeline_;

  TimelineController& timeline_controller_;

  ResponseCache& response_cache_;

  ParameterManager& parameter_manager_;

  GroupTable& group_table_;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_CONTROL_MANAGER_H
