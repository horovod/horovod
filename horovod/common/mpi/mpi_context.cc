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

#include <iostream>
#include <memory>
#include <vector>

#include "../common.h"
#include "../half.h"
#include "../logging.h"

namespace horovod {
namespace common {

MPI_Datatype MPIContext::GetMPIDataType(const std::shared_ptr<Tensor> tensor) {
  return GetMPIDataType(tensor->dtype());
}

MPI_Datatype MPIContext::GetMPIDataType(const DataType dtype) {
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

MPI_Op MPIContext::GetMPISumOp(DataType dtype) {
  // 根据数据类型来获取不同的SUMOP，如果我们的数据是float16的话，那我们
  // 可以用mpi_float16_sum，如果不是的一般就用MPI_SUM
  return dtype == HOROVOD_FLOAT16 ? mpi_float16_sum : MPI_SUM;
}

MPI_Comm MPIContext::GetMPICommunicator(Communicator comm) {
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

int MPIContext::GetMPITypeSize(DataType dtype) {
  int out;
  MPI_Type_size(GetMPIDataType(dtype), &out);
  return out;
}

void MPIContext::Initialize(const std::vector<int>& ranks,
                            MPIContextManager& ctx_manager) {

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
  // 这里默认是采用多线程的形式的，除非关闭了这个多线程的开关
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

  if (!ranks.empty()) {
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group work_group;
    MPI_Group_incl(world_group, ranks.size(), ranks.data(), &work_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, work_group, 0, &(mpi_comm));
    if (mpi_comm == MPI_COMM_NULL) {
      LOG(WARNING) << "Unable to create Horovod communicator, using "
                      "MPI_COMM_WORLD instead.";
      mpi_comm = MPI_COMM_WORLD;
    }
    MPI_Group_free(&world_group);
    MPI_Group_free(&work_group);
  } else if (!mpi_comm) {
    // No ranks were given and no communicator provided to horovod_init() so use
    // MPI_COMM_WORLD
    LOG(DEBUG) << "Using MPI_COMM_WORLD as a communicator.";
    // 使用MPI_COMM_WORLD作为通信子
    MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm);
  }

  // Create local comm, Determine local rank by querying the local communicator.
  // 获取本地通信子，这样后面才能获取本地的rank
  MPI_Comm_split_type(mpi_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &local_comm);

  // Get local rank and world rank for cross comm establishment.
  // 只有MPI上下文context初始化之后才可以获取local_rank和world_rank
  int local_rank, world_rank;
  MPI_Comm_rank(mpi_comm, &world_rank);
  MPI_Comm_rank(local_comm, &local_rank);
  if (local_rank == world_rank){
    LOG(DEBUG) << "local rank == world rank =" << local_rank;
  }

  // Create cross node communicator.
  MPI_Comm_split(mpi_comm, local_rank, world_rank, &cross_comm);

  // Create custom MPI float16 data type.
  MPI_Type_contiguous(2, MPI_BYTE, &mpi_float16_t);
  MPI_Type_commit(&mpi_float16_t);

  // Create custom MPI float16 summation op.
  MPI_Op_create(&float16_sum, 1, &mpi_float16_sum);
}

void MPIContext::Finalize(MPIContextManager& ctx_manager) {
  // 将获取的通信子和float16相关的通信op销毁
  if (!enabled_) {
    return;
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

  if (should_finalize) {
    ctx_manager.EnvFinalize();
  }
}

void MPIContextManager::EnvInitialize(int mpi_threads_required) {
  int mpi_threads_provided;
  // 这里需要注意一个问题，就是之前看的MPI教程中我们是用MPI_Init函数来初始化MPI的
  // 但是如果程序中有多线程的话，则我们可以用MPI_Init_thread函数，注意这个mpi_threads_required
  // 有如下的用法
  // The valid values for the level of thread support are:
  // MPI_THREAD_SINGLE
  // Only one thread will execute.
  // MPI_THREAD_FUNNELED
  // The process may be multi-threaded, but only the main thread will make MPI calls (all MPI calls are funneled to the main thread).
  // MPI_THREAD_SERIALIZED
  // The process may be multi-threaded, and multiple threads may make MPI calls, but only one at a time: 
  // MPI calls are not made concurrently from two distinct threads (all MPI calls are serialized).
  // MPI_THREAD_MULTIPLE
  // Multiple threads may call MPI, with no restrictions.
  // 默认直接用MPI_THREAD_MULTIPLE，因为保不齐是否是多线程调用MPI服务
  MPI_Init_thread(nullptr, nullptr, mpi_threads_required,
                  &mpi_threads_provided);
}

void MPIContextManager::EnvFinalize() {
  // 用于回收和释放MPI资源
  int is_mpi_finalized = 0;
  MPI_Finalized(&is_mpi_finalized); //防止MPI资源已经被释放了，所以需要加一个检查判断
  // 如果MPI资源没有被释放，这个is_mpi_finalized返回就是false，默认就是0，这样我们才
  // 可以去调用MPI_Finalize函数进行释放操作
  if (!is_mpi_finalized) {
    MPI_Finalize();
  }
}

} // namespace common
} // namespace horovod
