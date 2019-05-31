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

#ifndef HOROVOD_OPERATIONS_H
#define HOROVOD_OPERATIONS_H

#include <functional>

#include "common.h"
#define OMPI_SKIP_MPICXX
#include "mpi.h"

namespace horovod {
namespace common {

// The number of elements held by fusion buffer and hierarchical
// allreduce size is always a multiple of FUSION_BUFFER_ATOMIC_UNIT
#define FUSION_BUFFER_ATOMIC_UNIT 64

// Horovod knobs.
#define HOROVOD_MPI_THREADS_DISABLE "HOROVOD_MPI_THREADS_DISABLE"
#define HOROVOD_TIMELINE "HOROVOD_TIMELINE"
#define HOROVOD_TIMELINE_MARK_CYCLES "HOROVOD_TIMELINE_MARK_CYCLES"
#define HOROVOD_AUTOTUNE "HOROVOD_AUTOTUNE"
#define HOROVOD_AUTOTUNE_LOG "HOROVOD_AUTOTUNE_LOG"
#define HOROVOD_FUSION_THRESHOLD "HOROVOD_FUSION_THRESHOLD"
#define HOROVOD_CYCLE_TIME "HOROVOD_CYCLE_TIME"
#define HOROVOD_STALL_CHECK_DISABLE "HOROVOD_STALL_CHECK_DISABLE"
#define HOROVOD_STALL_CHECK_TIME_SECONDS "HOROVOD_STALL_CHECK_TIME_SECONDS"
#define HOROVOD_STALL_SHUTDOWN_TIME_SECONDS "HOROVOD_STALL_SHUTDOWN_TIME_SECONDS"
#define HOROVOD_HIERARCHICAL_ALLREDUCE "HOROVOD_HIERARCHICAL_ALLREDUCE"
#define HOROVOD_HIERARCHICAL_ALLGATHER "HOROVOD_HIERARCHICAL_ALLGATHER"
#define HOROVOD_CACHE_CAPACITY "HOROVOD_CACHE_CAPACITY"
#define HOROVOD_MLSL_BGT_AFFINITY "HOROVOD_MLSL_BGT_AFFINITY"

// Check that Horovod is initialized.
Status CheckInitialized();

extern "C" {

// C interface to initialize Horovod.
void horovod_init(const int *ranks, int nranks);

// C interface to initialize Horovod with the given MPI communicator.
void horovod_init_comm(MPI_Comm comm);

// C interface to shut down Horovod.
void horovod_shutdown();

// C interface to get index of current Horovod process.
// Returns -1 if Horovod is not initialized.
int horovod_rank();

// C interface to get index of current Horovod process in the node it is on.
// Returns -1 if Horovod is not initialized.
int horovod_local_rank();

// C interface to return number of Horovod processes.
// Returns -1 if Horovod is not initialized.
int horovod_size();

// C interface to return number of Horovod processes in the node it is on.
// Returns -1 if Horovod is not initialized.
int horovod_local_size();

// C interface to return flag indicating whether MPI multi-threading is
// supported. Returns -1 if Horovod is not initialized.
int horovod_mpi_threads_supported();
}

Status EnqueueTensorAllreduce(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback);

Status EnqueueTensorAllgather(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback);

Status EnqueueTensorBroadcast(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output, int root_rank,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback);

} // namespace common
} // namespace horovod

#endif // HOROVOD_OPERATIONS_H
