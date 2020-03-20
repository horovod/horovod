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

#include "gloo_controller.h"

#include <cstring>

#include "gloo/allgather.h"
#include "gloo/allgatherv.h"
#include "gloo/allreduce.h"
#include "gloo/barrier.h"
#include "gloo/broadcast.h"
#include "gloo/gather.h"

#include "gloo_context.h"
#include "../logging.h"
#include "../ops/gloo_operations.h"

namespace horovod {
namespace common {

void GlooController::DoInitialization() {
  rank_ = gloo_context_.ctx->rank;
  size_ = gloo_context_.ctx->size;
  is_coordinator_ = rank_ == 0;
  if (is_coordinator_) {
    LOG(DEBUG) << "Started Horovod with " << size_ << " processes";
  }

  // Determine local rank by if local context is presented.
  if (gloo_context_.local_ctx != nullptr) {
    local_rank_ = gloo_context_.local_ctx->rank;
    local_size_ = gloo_context_.local_ctx->size;
    local_comm_ranks_ = std::vector<int>((size_t)local_size_);
    local_comm_ranks_[local_rank_] = rank_;
    {
      gloo::AllgatherOptions opts(gloo_context_.local_ctx);
      opts.setInput(&rank_, 1);
      opts.setOutput(local_comm_ranks_.data(), local_size_);
      gloo::allgather(opts);
    }

    // Determine if cluster is homogeneous, i.e., if every node has the same
    // local_size
    auto local_sizes = std::vector<int>(size_);
    {
      gloo::AllgatherOptions opts(gloo_context_.ctx);
      opts.setInput(&local_size_, 1);
      opts.setOutput(local_sizes.data(), size_);
      gloo::allgather(opts);
    }
    is_homogeneous_ = true;
    for (int i = 0; i < size_; ++i) {
      if (local_sizes[i] != local_size_) {
        is_homogeneous_ = false;
        break;
      }
    }

    // Construct a shorter local sizes vector with length cross size.
    // e.g. For local_sizes = {4, 4, 4, 4, 3, 3, 3},
    //      we want to construct a local_sizes_for_cross_rank_ = {4, 3}
    local_sizes_for_cross_rank_ = std::vector<int>(cross_size_);
    int displacement = 0;
    // For each cross rank iter, set corresponding local size and move
    // displacement advance by the local size
    for (int cross_rank = 0; cross_rank < cross_size_; ++cross_rank) {
      local_sizes_for_cross_rank_[cross_rank] = local_sizes[displacement];
      displacement += local_sizes[displacement];
    }
  }

  // Get cross-node rank and size in case of hierarchical allreduce.
  if (gloo_context_.cross_ctx != nullptr) {
    cross_rank_ = gloo_context_.cross_ctx->rank;
    cross_size_ = gloo_context_.cross_ctx->size;
  }

  LOG(DEBUG) << "Gloo controller initialized.";
}

int GlooController::GetTypeSize(DataType dtype) {
  switch (dtype) {
  case HOROVOD_FLOAT16:
    return sizeof(gloo::float16);
  default:
    return DataType_Size(dtype);
  }
}

void GlooController::CrossRankBitwiseAnd(std::vector<long long>& bitvector,
                                         int count) {
  gloo::AllreduceOptions opts(gloo_context_.ctx);
  opts.setOutput(bitvector.data(), count);
  void (*func)(void*, const void*, const void*, size_t) = &BitAnd<long long>;
  opts.setReduceFunction(gloo::AllreduceOptions::Func(func));
  gloo::allreduce(opts);
}

void GlooController::CrossRankBitwiseOr(std::vector<long long>& bitvector,
                                        int count) {
  gloo::AllreduceOptions opts(gloo_context_.ctx);
  opts.setOutput(bitvector.data(), count);
  void (*func)(void*, const void*, const void*, size_t) = &BitOr<long long>;
  opts.setReduceFunction(gloo::AllreduceOptions::Func(func));
  gloo::allreduce(opts);
}

void GlooController::RecvReadyTensors(std::vector<std::string>& ready_to_reduce,
                                      std::vector<RequestList>& ready_list) {
  // Rank zero has put all its own tensors in the tensor count table.
  // Now, it should count all the tensors that are coming from other
  // ranks at this tick.

  // 1. Get message lengths from every rank.
  auto recvcounts = new int[size_];

  // do allgather
  {
    // gloo doesn't have inplace option, put a zero as input for root rank
    int send_data = 0;
    gloo::AllgatherOptions opts(gloo_context_.ctx);
    opts.setInput(&send_data, 1);
    opts.setOutput(recvcounts, size_);
    gloo::allgather(opts);
  }

  // 2. Compute displacements.
  auto displcmnts = new int[size_];
  size_t total_size = 0;
  for (int i = 0; i < size_; ++i) {
    if (i == 0) {
      displcmnts[i] = 0;
    } else {
      displcmnts[i] = recvcounts[i - 1] + displcmnts[i - 1];
    }
    total_size += recvcounts[i];
  }

  // 3. Collect messages from every rank.
  auto buffer = new uint8_t[total_size];

  // do allgatherv
  {
    auto input = new uint8_t[0];
    gloo::AllgathervOptions opts(gloo_context_.ctx);
    opts.setInput(input, 0);
    std::vector<size_t> count_vec(recvcounts, recvcounts + size_);
    opts.setOutput(buffer, count_vec);
    gloo::allgatherv(opts);
  }

  // 4. Process messages.
  // create a dummy list for rank 0
  ready_list.emplace_back();
  for (int i = 1; i < size_; ++i) {
    auto rank_buffer_ptr = buffer + displcmnts[i];
    RequestList received_message_list;
    RequestList::ParseFromBytes(received_message_list, rank_buffer_ptr);
    ready_list.push_back(std::move(received_message_list));
  }

  // 5. Free buffers.
  delete[] recvcounts;
  delete[] displcmnts;
  delete[] buffer;
}

void GlooController::SendFinalTensors(ResponseList& response_list) {
  // Notify all nodes which tensors we'd like to reduce at this step.
  std::string encoded_response;
  ResponseList::SerializeToString(response_list, encoded_response);

  // Boardcast the response length
  int encoded_response_length = (int)encoded_response.length() + 1;
  {
    gloo::BroadcastOptions opts(gloo_context_.ctx);
    opts.setOutput(&encoded_response_length, 1);
    opts.setRoot(RANK_ZERO);
    gloo::broadcast(opts);
  }

  // Boardcast the response
  {
    gloo::BroadcastOptions opts(gloo_context_.ctx);
    opts.setOutput((uint8_t*)(encoded_response.c_str()),
                   encoded_response_length);
    opts.setRoot(RANK_ZERO);
    gloo::broadcast(opts);
  }
}

void GlooController::SendReadyTensors(RequestList& message_list) {
  std::string encoded_message;
  RequestList::SerializeToString(message_list, encoded_message);

  // Gloo doesn't have the gatherv options, using allgatherv instead.

  // send message length to root
  auto recvcounts = new int[size_];
  int encoded_message_length = (int)encoded_message.length() + 1;
  {
    gloo::AllgatherOptions opts(gloo_context_.ctx);
    opts.setInput(&encoded_message_length, 1);
    opts.setOutput(recvcounts, size_);
    gloo::allgather(opts);
  }

  auto displcmnts = new int[size_];
  size_t total_size = 0;
  for (int i = 0; i < size_; ++i) {
    if (i == 0) {
      displcmnts[i] = 0;
    } else {
      displcmnts[i] = recvcounts[i - 1] + displcmnts[i - 1];
    }
    total_size += recvcounts[i];
  }

  // 3. Collect messages from every rank.
  auto buffer = new uint8_t[total_size];
  // send message body to root
  {
    gloo::AllgathervOptions opts(gloo_context_.ctx);
    opts.setInput((uint8_t*)encoded_message.c_str(), encoded_message_length);
    std::vector<size_t> count_vec(recvcounts, recvcounts + size_);
    opts.setOutput((uint8_t*)buffer, count_vec);
    gloo::allgatherv(opts);
  }

  delete[] recvcounts;
  delete[] displcmnts;
  delete[] buffer;
}

void GlooController::RecvFinalTensors(ResponseList& response_list) {
  int msg_length;
  // root broadcast final message length to others
  {
    gloo::BroadcastOptions opts(gloo_context_.ctx);
    opts.setOutput(&msg_length, 1);
    opts.setRoot(RANK_ZERO);
    gloo::broadcast(opts);
  }
  // root broadcast final message to others
  auto buffer = new uint8_t[msg_length];
  memset(buffer, 0, msg_length);
  {
    gloo::BroadcastOptions opts(gloo_context_.ctx);
    opts.setOutput((uint8_t*)buffer, msg_length);
    opts.setRoot(RANK_ZERO);
    gloo::broadcast(opts);
  }

  ResponseList::ParseFromBytes(response_list, buffer);
  delete[] buffer;
}

void GlooController::Bcast(void* buffer, size_t size, int root_rank,
                           Communicator communicator) {
  gloo::BroadcastOptions opts(gloo_context_.GetGlooContext(communicator));
  opts.setOutput((uint8_t*)buffer, size);
  opts.setRoot(root_rank);
  gloo::broadcast(opts);
}

void GlooController::Barrier(Communicator communicator) {
  gloo::BarrierOptions opts(gloo_context_.GetGlooContext(communicator));
  gloo::barrier(opts);
}

} // namespace common
} // namespace horovod
