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

#include "mpi_controller.h"

#include <numeric>

#include "../common.h"
#include "../logging.h"

namespace horovod {
namespace common {

// MPIController
void MPIController::DoInitialization() {
  assert(mpi_ctx_.global_comm != MPI_COMM_NULL);
  assert(mpi_ctx_.mpi_comm != MPI_COMM_NULL);
  assert(mpi_ctx_.local_comm != MPI_COMM_NULL);
  assert(mpi_ctx_.cross_comm != MPI_COMM_NULL);

  // Check if multi-thread is supported.
  int provided;
  MPI_Query_thread(&provided);
  mpi_threads_supported_ = (provided == MPI_THREAD_MULTIPLE);

  // Get MPI rank to determine if we are rank zero.
  MPI_Comm_rank(mpi_ctx_.mpi_comm, &rank_);
  is_coordinator_ = rank_ == 0;

  // Get MPI size to determine how many tensors to wait for before reducing.
  MPI_Comm_size(mpi_ctx_.mpi_comm, &size_);

  if (is_coordinator_) {
    LOG(DEBUG) << "Started Horovod process set with " << size_ << " processes";
  }

  // Build mappings (process-set specific rank) <-> (global rank)
  {
    MPI_Group global_group;
    MPI_Comm_group(mpi_ctx_.global_comm, &global_group);
    MPI_Group process_set_group;
    MPI_Comm_group(mpi_ctx_.mpi_comm, &process_set_group);

    global_ranks_ = std::vector<int>(size_);
    auto process_set_ranks = std::vector<int>(size_);
    std::iota(process_set_ranks.begin(), process_set_ranks.end(), 0);
    MPI_Group_translate_ranks(process_set_group, size_,
                              process_set_ranks.data(), global_group,
                              global_ranks_.data());

    global_rank_to_controller_rank_ = std::unordered_map<int, int>(size_);
    for (int rank = 0; rank < size_; ++rank) {
      global_rank_to_controller_rank_[global_ranks_[rank]] = rank;
    }
  }

  // Determine local rank by querying the local communicator.
  MPI_Comm_rank(mpi_ctx_.local_comm, &local_rank_);
  MPI_Comm_size(mpi_ctx_.local_comm, &local_size_);
  local_comm_ranks_ = std::vector<int>((size_t)local_size_);
  local_comm_ranks_[local_rank_] = rank_;
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, local_comm_ranks_.data(), 1,
                MPI_INT, mpi_ctx_.local_comm);

  // Determine if cluster is homogeneous, i.e., if every node has the same
  // local_size
  auto local_sizes = std::vector<int>(size_);
  MPI_Allgather(&local_size_, 1, MPI_INT, local_sizes.data(), 1, MPI_INT,
                mpi_ctx_.mpi_comm);

  is_homogeneous_ = true;
  for (int i = 0; i < size_; ++i) {
    if (local_sizes[i] != local_size_) {
      is_homogeneous_ = false;
      break;
    }
  }

  // Get cross-node rank and size in case of hierarchical allreduce.
  MPI_Comm_rank(mpi_ctx_.cross_comm, &cross_rank_);
  MPI_Comm_size(mpi_ctx_.cross_comm, &cross_size_);

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

  LOG(DEBUG) << "MPI controller initialized.";
}

int MPIController::GetTypeSize(DataType dtype) {
  return mpi_ctx_.GetMPITypeSize(dtype);
}

void MPIController::CrossRankBitwiseAnd(std::vector<long long>& bitvector,
                                        int count) {
  int ret_code = MPI_Allreduce(MPI_IN_PLACE, bitvector.data(), count,
                               MPI_LONG_LONG_INT, MPI_BAND, mpi_ctx_.mpi_comm);
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_AllReduce failed, see MPI output for details.");
  }
}

void MPIController::CrossRankBitwiseOr(std::vector<long long>& bitvector,
                                       int count) {
  int ret_code = MPI_Allreduce(MPI_IN_PLACE, bitvector.data(), count,
                               MPI_LONG_LONG_INT, MPI_BOR, mpi_ctx_.mpi_comm);
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_AllReduce failed, see MPI output for details.");
  }
}

void MPIController::RecvReadyTensors(std::vector<std::string>& ready_to_reduce,
                                     std::vector<RequestList>& ready_list) {
  // Rank zero has put all its own tensors in the tensor count table.
  // Now, it should count all the tensors that are coming from other
  // ranks at this tick.

  // 1. Get message lengths from every rank.
  auto recvcounts = new int[size_];
  recvcounts[0] = 0;
  MPI_Gather(MPI_IN_PLACE, 1, MPI_INT, recvcounts, 1, MPI_INT, RANK_ZERO,
             mpi_ctx_.mpi_comm);

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
  MPI_Gatherv(nullptr, 0, MPI_BYTE, buffer, recvcounts, displcmnts, MPI_BYTE,
              RANK_ZERO, mpi_ctx_.mpi_comm);

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

void MPIController::SendFinalTensors(ResponseList& response_list) {
  // Notify all nodes which tensors we'd like to reduce at this step.
  std::string encoded_response;
  ResponseList::SerializeToString(response_list, encoded_response);
  int encoded_response_length = (int)encoded_response.length() + 1;
  MPI_Bcast(&encoded_response_length, 1, MPI_INT, RANK_ZERO, mpi_ctx_.mpi_comm);

  MPI_Bcast((void*)encoded_response.c_str(), encoded_response_length, MPI_BYTE,
            RANK_ZERO, mpi_ctx_.mpi_comm);
}

void MPIController::SendReadyTensors(RequestList& message_list) {
  std::string encoded_message;
  RequestList::SerializeToString(message_list, encoded_message);
  int encoded_message_length = (int)encoded_message.length() + 1;
  int ret_code = MPI_Gather(&encoded_message_length, 1, MPI_INT, nullptr, 1,
                            MPI_INT, RANK_ZERO, mpi_ctx_.mpi_comm);
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Gather failed, see MPI output for details.");
  }

  ret_code = MPI_Gatherv((void*)encoded_message.c_str(), encoded_message_length,
                         MPI_BYTE, nullptr, nullptr, nullptr, MPI_BYTE,
                         RANK_ZERO, mpi_ctx_.mpi_comm);
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Gather failed, see MPI output for details.");
  }
}

void MPIController::RecvFinalTensors(ResponseList& response_list) {
  int msg_length;
  int ret_code =
      MPI_Bcast(&msg_length, 1, MPI_INT, RANK_ZERO, mpi_ctx_.mpi_comm);
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_Broadcast failed, see MPI output for details.");
  }

  auto buffer = new uint8_t[msg_length];
  ret_code =
      MPI_Bcast(buffer, msg_length, MPI_BYTE, RANK_ZERO, mpi_ctx_.mpi_comm);
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_Broadcast failed, see MPI output for details.");
  }
  ResponseList::ParseFromBytes(response_list, buffer);
  delete[] buffer;
}

void MPIController::Bcast(void* buffer, size_t size, int root_rank,
                          Communicator communicator) {
  MPI_Comm comm = mpi_ctx_.GetMPICommunicator(communicator);
  int ret_code = MPI_Bcast(buffer, size, MPI_BYTE, root_rank, comm);
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_Broadcast failed, see MPI output for details.");
  }
}

void MPIController::AlltoallGetRecvSplits(const std::vector<int32_t>& splits,
                                          std::vector<int32_t>& recvsplits) {
  recvsplits.resize(size_);
  MPI_Comm comm = mpi_ctx_.GetMPICommunicator(Communicator::GLOBAL);
  int ret_code = MPI_Alltoall(splits.data(), 1, MPI_INT,
                              recvsplits.data(), 1, MPI_INT,
                              comm);
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_Alltoall failed, see MPI output for details.");
  }
};

void MPIController::Barrier(Communicator communicator) {
  MPI_Comm comm = mpi_ctx_.GetMPICommunicator(communicator);
  int ret_code = MPI_Barrier(comm);
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Barrier failed, see MPI output for details.");
  }
}

void MPIController::Allgather2Ints(std::array<int, 2> values,
                                   std::vector<int>& recv_values) {
  recv_values.resize(size_ * 2);
  MPI_Comm comm = mpi_ctx_.GetMPICommunicator(Communicator::GLOBAL);
  int ret_code = MPI_Allgather(values.data(), 2, MPI_INT,
                               recv_values.data(), 2, MPI_INT,
                               comm);
  if (ret_code != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_Allgather failed, see MPI output for details.");
  }
}


} // namespace common
} // namespace horovod
