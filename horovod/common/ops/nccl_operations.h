// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
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

#ifndef HOROVOD_NCCL_OPERATIONS_H
#define HOROVOD_NCCL_OPERATIONS_H

#include <nccl.h>

#include "../mpi_context.h"
#include "cuda_operations.h"

namespace horovod {
namespace common {

struct NCCLContext {
  std::unordered_map<std::vector<int32_t>, ncclComm_t> nccl_comms;

  void ErrorCheck(std::string op_name, ncclResult_t nccl_result);

  void ShutDown();
};

class NCCLAllreduce : public CUDAAllreduce {
public:
  NCCLAllreduce(NCCLContext* nccl_context, MPIContext* mpi_context,
                CUDAContext* cuda_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

protected:
  void InitNCCLComm(const std::vector<TensorTableEntry>& entries, const std::vector<int32_t>& nccl_device_map);

  virtual void PopulateNCCLCommStrategy(int& nccl_rank, int& nccl_size,
                                        Communicator& nccl_id_bcast_comm);

  NCCLContext* nccl_context_;
  ncclComm_t* nccl_comm_;

  MPIContext* mpi_context_;
};

class NCCLHierarchicalAllreduce : public NCCLAllreduce {
public:
  NCCLHierarchicalAllreduce(NCCLContext* nccl_context, MPIContext* mpi_context,
                            CUDAContext* cuda_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

private:
  void PopulateNCCLCommStrategy(int& nccl_rank, int& nccl_size,
                                Communicator& nccl_id_bcast_comm) override;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_NCCL_OPERATIONS_H
