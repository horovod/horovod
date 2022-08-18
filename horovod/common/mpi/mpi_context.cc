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

#include "mpi_context.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

#include "../common.h"
#include "../half.h"
#include "../logging.h"

namespace horovod {
namespace common {

MPI_Datatype MPIContext::GetMPIDataType(const std::shared_ptr<Tensor> tensor) const {
  return GetMPIDataType(tensor->dtype());
}

MPI_Datatype MPIContext::GetMPIDataType(const DataType dtype) const {
  switch (dtype) {
  case HOROVOD_UINT8:
    return MPI_UINT8_T;
  case HOROVOD_INT8:
    return MPI_INT8_T;
  case HOROVOD_UINT16:
    return MPI_UINT16_T;
  case HOROVOD_INT16:
    return MPI_INT16_T;
  case HOROVOD_INT32:
    return MPI_INT32_T;
  case HOROVOD_INT64:
    return MPI_INT64_T;
  case HOROVOD_FLOAT16:
    return mpi_float16_t;
  case HOROVOD_FLOAT32:
    return MPI_FLOAT;
  case HOROVOD_FLOAT64:
    return MPI_DOUBLE;
  case HOROVOD_BOOL:
    return MPI_C_BOOL;
  default:
    throw std::logic_error("Type " + DataType_Name(dtype) +
                           " is not supported in MPI mode.");
  }
}

MPI_Op MPIContext::GetMPISumOp(DataType dtype) const {
  return dtype == HOROVOD_FLOAT16 ? mpi_float16_sum : MPI_SUM;
}

MPI_Op MPIContext::GetMPIMinOp(DataType dtype) const {
  return dtype == HOROVOD_FLOAT16 ? mpi_float16_min : MPI_MIN;
}

MPI_Op MPIContext::GetMPIMaxOp(DataType dtype) const {
  return dtype == HOROVOD_FLOAT16 ? mpi_float16_max : MPI_MAX;
}

MPI_Op MPIContext::GetMPIProdOp(DataType dtype) const {
  return dtype == HOROVOD_FLOAT16 ? mpi_float16_prod : MPI_PROD;
}

MPI_Comm MPIContext::GetMPICommunicator(Communicator comm) const {
  switch (comm) {
  case GLOBAL:
    return mpi_comm;
  case LOCAL:
    return local_comm;
  case CROSS:
    return cross_comm;
  default:
    throw std::logic_error("Communicator " + CommunicatorName(comm) +
                           " is not supported in MPI mode.");
  }
}

int MPIContext::GetMPITypeSize(DataType dtype) const {
  int out;
  MPI_Type_size(GetMPIDataType(dtype), &out);
  return out;
}

namespace {

void CreateMPIFloat16TypeAndOps(MPI_Datatype& mpi_float16_t,
                                MPI_Op& mpi_float16_sum,
                                MPI_Op& mpi_float16_min,
                                MPI_Op& mpi_float16_max,
                                MPI_Op& mpi_float16_prod) {
  // Create custom MPI float16 data type.
  MPI_Type_contiguous(2, MPI_BYTE, &mpi_float16_t);
  MPI_Type_commit(&mpi_float16_t);

  // Create custom MPI float16 summation op.
  MPI_Op_create(&float16_sum, 1, &mpi_float16_sum);

  // Create custom MPI float16 min op.
  MPI_Op_create(&float16_min, 1, &mpi_float16_min);

  // Create custom MPI float16 max op.
  MPI_Op_create(&float16_max, 1, &mpi_float16_max);

  // Create custom MPI float16 prod op.
  MPI_Op_create(&float16_prod, 1, &mpi_float16_prod);
}

void CreateMPILocalAndCrossComm(MPI_Comm mpi_comm, MPI_Comm& local_comm,
                                MPI_Comm& cross_comm) {
  // Create local comm, Determine local rank by querying the local communicator.
  MPI_Comm_split_type(mpi_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &local_comm);

  // Get ranks corresponding to mpi_comm and local_comm for cross comm establishment.
  int local_rank, all_rank;
  MPI_Comm_rank(mpi_comm, &all_rank);
  MPI_Comm_rank(local_comm, &local_rank);

  // Create cross node communicator.
  MPI_Comm_split(mpi_comm, local_rank, all_rank, &cross_comm);
}

} // namespace

void MPIContext::Initialize(MPIContextManager& ctx_manager) {

  if (!enabled_) {
    return;
  }
  // Initialize MPI if it was not initialized. This must happen on the
  // background thread, since not all MPI implementations support being called
  // from multiple threads.
  //
  // In some cases MPI library has multi-threading support, but it slows down
  // certain components, e.g. OpenIB BTL in OpenMPI gets disabled if
  // MPI_THREAD_MULTIPLE is requested.
  //
  // By default, we will ask for multiple threads, so other libraries like
  // mpi4py can be used together with Horovod if multi-threaded MPI is
  // installed.
  auto mpi_threads_disable = std::getenv(HOROVOD_MPI_THREADS_DISABLE);
  int required = MPI_THREAD_MULTIPLE;
  if (mpi_threads_disable != nullptr &&
      std::strtol(mpi_threads_disable, nullptr, 10) > 0) {
    required = MPI_THREAD_SINGLE;
  }
  int is_mpi_initialized = 0;
  MPI_Initialized(&is_mpi_initialized);
  if (is_mpi_initialized) {
    int provided;
    MPI_Query_thread(&provided);
    if (provided < MPI_THREAD_MULTIPLE) {
      LOG(WARNING)
          << "MPI has already been initialized without "
             "multi-threading support (MPI_THREAD_MULTIPLE). This will "
             "likely cause a segmentation fault.";
    }
  } else {
    // MPI environment has not been created, using manager to initialize.
    ctx_manager.EnvInitialize(required);
    should_finalize = true;
  }

  if (!ranks_.empty()) {
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group work_group;
    MPI_Group_incl(world_group, ranks_.size(), ranks_.data(), &work_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, work_group, 0, &(global_comm));
    if (global_comm == MPI_COMM_NULL) {
      LOG(WARNING) << "Unable to create global Horovod communicator, using "
                      "MPI_COMM_WORLD instead.";
      global_comm = MPI_COMM_WORLD;
    }
    MPI_Group_free(&world_group);
    MPI_Group_free(&work_group);
  } else if (global_comm == MPI_COMM_NULL) {
    // No ranks were given and no communicator provided to horovod_init() so use
    // MPI_COMM_WORLD
    LOG(DEBUG) << "Using MPI_COMM_WORLD as global communicator.";
    MPI_Comm_dup(MPI_COMM_WORLD, &global_comm);
  }

  MPI_Comm_dup(global_comm, &(mpi_comm));

  CreateMPILocalAndCrossComm(mpi_comm, local_comm, cross_comm);

  CreateMPIFloat16TypeAndOps(mpi_float16_t, mpi_float16_sum, mpi_float16_min,
                             mpi_float16_max, mpi_float16_prod);
}

void MPIContext::InitializeForProcessSet(const MPIContext& global_context,
                                         const std::vector<int>& ranks) {
  assert(global_context.IsEnabled());
  assert(global_context.global_comm != MPI_COMM_NULL);

  enabled_ = true;
  should_finalize = false;
  MPI_Comm_dup(global_context.global_comm, &global_comm);
  if (ranks.empty()) {
    MPI_Comm_dup(global_comm, &(mpi_comm));
  } else {
    // Create mpi_comm for this process set.
    MPI_Group world_group;
    MPI_Comm_group(global_comm, &world_group);
    MPI_Group work_group;
    MPI_Group_incl(world_group, ranks.size(), ranks.data(), &work_group);
    MPI_Comm_create_group(global_comm, work_group, 0, &(mpi_comm));
    MPI_Group_free(&world_group);
    MPI_Group_free(&work_group);
  }
  if (mpi_comm == MPI_COMM_NULL) {
    // This process does not belong to the group.
    local_comm = MPI_COMM_NULL;
    cross_comm = MPI_COMM_NULL;
  } else {
    CreateMPILocalAndCrossComm(mpi_comm, local_comm, cross_comm);
  }

  CreateMPIFloat16TypeAndOps(mpi_float16_t, mpi_float16_sum, mpi_float16_min,
                             mpi_float16_max, mpi_float16_prod);
}

void MPIContext::Finalize(MPIContextManager& ctx_manager) {
  if (!enabled_) {
    return;
  }
  FinalizeWithoutEnv();
  if (should_finalize) {
    ctx_manager.EnvFinalize();
  }
}

void MPIContext::FinalizeWithoutEnv() {
   if (!enabled_) {
    return;
  }
  // It is OK to call MPI_Comm_free multiple times on multiple handles to the same communicator object
  if (global_comm != MPI_COMM_NULL && global_comm != MPI_COMM_WORLD) {
    MPI_Comm_free(&global_comm);
  }
  if (mpi_comm != MPI_COMM_NULL && mpi_comm != MPI_COMM_WORLD) {
    MPI_Comm_free(&mpi_comm);
  }
  if (local_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&local_comm);
  }
  if (cross_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&cross_comm);
  }
  if (mpi_float16_t != MPI_DATATYPE_NULL) {
    MPI_Type_free(&mpi_float16_t);
  }
  if (mpi_float16_sum != MPI_OP_NULL) {
    MPI_Op_free(&mpi_float16_sum);
  }
  if (mpi_float16_min != MPI_OP_NULL) {
    MPI_Op_free(&mpi_float16_min);
  }
  if (mpi_float16_max != MPI_OP_NULL) {
    MPI_Op_free(&mpi_float16_max);
  }
  if (mpi_float16_prod != MPI_OP_NULL) {
    MPI_Op_free(&mpi_float16_prod);
  }
}

void MPIContextManager::EnvInitialize(int mpi_threads_required) {
  int mpi_threads_provided;
  MPI_Init_thread(nullptr, nullptr, mpi_threads_required,
                  &mpi_threads_provided);
}

void MPIContextManager::EnvFinalize() {
  int is_mpi_finalized = 0;
  MPI_Finalized(&is_mpi_finalized);
  if (!is_mpi_finalized) {
    MPI_Finalize();
  }
}

} // namespace common
} // namespace horovod
