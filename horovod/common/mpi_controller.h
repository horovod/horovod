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

#ifndef HOROVOD_MPI_CONTROLLER_H
#define HOROVOD_MPI_CONTROLLER_H

#include "controller.h"
#include "mpi_context.h"

namespace horovod {
namespace common {

class MPIController : public Controller {
public:
  MPIController(MPIContext* mpi_ctx) : mpi_ctx_(mpi_ctx) {}

  void Initialize() override;

  int GetTypeSize(DataType dtype) override;

  void CrossRankBitwiseAnd(std::vector<long long>& bitvector,
                           int count) override;

  void CrossRankBitwiseOr(std::vector<long long>& bitvector,
                          int count) override;

  bool RecvReadyTensors(std::vector<std::string>& ready_to_reduce,
                        std::shared_ptr<MessageTable> message_table,
                        Timeline& timeline) override;

  void SendFinalTensors(ResponseList& response_list) override;

  void SendReadyTensors(RequestList& message_list) override;

  void RecvFinalTensors(ResponseList& response_list) override;

  void SynchronizeParameters(ParameterManager& para_manager) override;

  bool IsMpiThreadsSupported() const { return mpi_threads_supported_; }

protected:
  MPIContext* mpi_ctx_ = nullptr;

  // flag indicating whether MPI multi-threading is supported
  bool mpi_threads_supported_ = false;
};

} // namespace common
} // namespace horovod
#endif // HOROVOD_MPI_CONTROLLER_H
