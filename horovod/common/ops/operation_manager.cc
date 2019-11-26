// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
// Modifications copyright Microsoft
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

#include "operation_manager.h"

namespace horovod {
namespace common {

OperationManager::OperationManager(ParameterManager* param_manager,
                                   std::vector<std::shared_ptr<AllreduceOp>> allreduce_ops,
                                   std::vector<std::shared_ptr<AllgatherOp>> allgather_ops,
                                   std::vector<std::shared_ptr<BroadcastOp>> broadcast_ops,
                                   std::shared_ptr<JoinOp> join_op,
                                   std::vector<std::shared_ptr<AllreduceOp>> adasum_ops,
                                   std::shared_ptr<ErrorOp> error_op)
    : param_manager_(param_manager),
      allreduce_ops_(std::move(allreduce_ops)),
      allgather_ops_(std::move(allgather_ops)),
      broadcast_ops_(std::move(broadcast_ops)),
      join_op_(std::move(join_op)),
      adasum_ops_(std::move(adasum_ops)),
      error_op_(std::move(error_op)) {}

Status OperationManager::ExecuteAllreduce(std::vector<TensorTableEntry>& entries,
                                          const Response& response) const {
  for (auto& op : allreduce_ops_) {
    if (op->Enabled(*param_manager_, entries, response)) {
      return op->Execute(entries, response);
    }
  }
  throw std::logic_error("No Allreduce operation enabled");
}

Status OperationManager::ExecuteAllgather(std::vector<TensorTableEntry>& entries,
                                          const Response& response) const {
  for (auto& op : allgather_ops_) {
    if (op->Enabled(*param_manager_, entries, response)) {
      return op->Execute(entries, response);
    }
  }
  throw std::logic_error("No Allgather operation enabled");
}

Status OperationManager::ExecuteBroadcast(std::vector<TensorTableEntry>& entries,
                                          const Response& response) const {
  for (auto& op : broadcast_ops_) {
    if (op->Enabled(*param_manager_, entries, response)) {
      return op->Execute(entries, response);
    }
  }
  throw std::logic_error("No Broadcast operation enabled");
}

Status OperationManager::ExecuteJoin(std::vector<TensorTableEntry>& entries,
                                          const Response& response) const {
  return join_op_->Execute(entries, response);
}

Status OperationManager::ExecuteAdasum(std::vector<TensorTableEntry>& entries,
                                          const Response& response) const {
  for (auto& op : adasum_ops_) {
    if (op->Enabled(*param_manager_, entries, response)) {
      return op->Execute(entries, response);
    }
  }
  throw std::logic_error("No Adasum operation enabled");
}

Status OperationManager::ExecuteError(std::vector<TensorTableEntry>& entries,
                                      const Response& response) const {
  return error_op_->Execute(entries, response);
}

Status OperationManager::ExecuteOperation(std::vector<TensorTableEntry>& entries,
                                          const Response& response) const {
  if (response.response_type() == Response::ALLREDUCE) {
    return ExecuteAllreduce(entries, response);
  } else if (response.response_type() == Response::ALLGATHER) {
    return ExecuteAllgather(entries, response);
  } else if (response.response_type() == Response::BROADCAST) {
    return ExecuteBroadcast(entries, response);
  } else if (response.response_type() == Response::JOIN) {
    return ExecuteJoin(entries, response);
  } else if (response.response_type() == Response::ADASUM) {
    return ExecuteAdasum(entries, response);
  } else if (response.response_type() == Response::ERROR) {
    return ExecuteError(entries, response);
  } else {
    throw std::logic_error("No operation found for response type provided");
  }
}

} // namespace common
} // namespace horovod
