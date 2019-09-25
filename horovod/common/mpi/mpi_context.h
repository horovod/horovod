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

#ifndef HOROVOD_MPI_CONTEXT_H
#define HOROVOD_MPI_CONTEXT_H

#include <iostream>
#include <memory>
#include <vector>

#include "../common.h"
#include "../half.h"
#include "../logging.h"

namespace horovod {
namespace common {

// Base class for managing MPI environment. Can be derived if other frameworks
// (like DDL) are able to manage MPI environment.
class MPIContextManager {
public:
  // Initialize MPI environment with required multi-threads support level.
  virtual void EnvInitialize(int mpi_threads_required);

  // Finalize MPI environment.
  virtual void EnvFinalize();
};

struct MPIContext {

  void Enable() {
    enabled_ = true;
    LOG(DEBUG) << "MPI context enabled.";
  };

  bool IsEnabled() { return enabled_; }

  // Take an argument of context manager pointer that will take care of
  // initialization of MPI environment.
  void Initialize(const std::vector<int>& ranks,
                  MPIContextManager& ctx_manager);

  // Take an argument of context manager pointer that will take care of
  // finalization of MPI environment.
  void Finalize(MPIContextManager& ctx_manager);
  MPI_Datatype GetMPIDataType(std::shared_ptr<Tensor> tensor);

  MPI_Datatype GetMPIDataType(DataType dtype);

  MPI_Op GetMPISumOp(DataType dtype);

  MPI_Comm GetMPICommunicator(Communicator comm);

  int GetMPITypeSize(DataType dtype);

  // Flag indicating whether mpi is enabled.
  bool enabled_ = false;

  // MPI custom data type for float16.
  MPI_Datatype mpi_float16_t;
  MPI_Op mpi_float16_sum;

  // Private MPI communicator for Horovod to ensure no collisions with other
  // threads using MPI.
  MPI_Comm mpi_comm;

  // Node-local communicator.
  MPI_Comm local_comm;

  // Cross-node communicator for hierarchical allreduce.
  MPI_Comm cross_comm;

  // MPI Window used for shared memory allgather
  MPI_Win window;

  // Whether mpi context should be finalize.
  bool should_finalize = false;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_MPI_CONTEXT_H
