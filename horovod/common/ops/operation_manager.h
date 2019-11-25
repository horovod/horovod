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

#ifndef HOROVOD_OPERATION_MANAGER_H
#define HOROVOD_OPERATION_MANAGER_H

#include "collective_operations.h"
#include "../parameter_manager.h"

namespace horovod {
namespace common {

class OperationManager {
public:
  OperationManager(ParameterManager* param_manager,
                   std::vector<std::shared_ptr<AllreduceOp>> allreduce_ops,
                   std::vector<std::shared_ptr<AllgatherOp>> allgather_ops,
                   std::vector<std::shared_ptr<BroadcastOp>> broadcast_ops,
                   std::shared_ptr<JoinOp> join_op,
                   std::vector<std::shared_ptr<AllreduceOp>> adasum_ops,
                   std::shared_ptr<ErrorOp> error_op);

  virtual ~OperationManager() = default;

  Status ExecuteAllreduce(std::vector<TensorTableEntry>& entries, const Response& response) const;

  Status ExecuteAllgather(std::vector<TensorTableEntry>& entries, const Response& response) const;

  Status ExecuteBroadcast(std::vector<TensorTableEntry>& entries, const Response& response) const;

  Status ExecuteError(std::vector<TensorTableEntry>& entries, const Response& response) const;

  Status ExecuteJoin(std::vector<TensorTableEntry>& entries, const Response& response) const;

  Status ExecuteAdasum(std::vector<TensorTableEntry>& entries, const Response& response) const;

  Status ExecuteOperation(std::vector<TensorTableEntry>& entries, const Response& response) const;

private:
  ParameterManager* param_manager_;

  std::vector<std::shared_ptr<AllreduceOp>> allreduce_ops_;
  std::vector<std::shared_ptr<AllgatherOp>> allgather_ops_;
  std::vector<std::shared_ptr<BroadcastOp>> broadcast_ops_;
  std::shared_ptr<JoinOp> join_op_;
  std::vector<std::shared_ptr<AllreduceOp>> adasum_ops_;
  std::shared_ptr<ErrorOp> error_op_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_OPERATION_MANAGER_H
