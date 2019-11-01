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

#if HAVE_MPI
#define OMPI_SKIP_MPICXX
#include "mpi.h"
#endif

#include "common.h"

namespace horovod {
namespace common {

// Check that Horovod is initialized.
Status CheckInitialized();

extern "C" {

// C interface to initialize Horovod.
void horovod_init(const int *ranks, int nranks);

#if HAVE_MPI
// C interface to initialize Horovod with the given MPI communicator.
void horovod_init_comm(MPI_Comm comm);
#endif

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

// C interface to return flag indicating whether MPI is enabled.
bool horovod_mpi_enabled();

// C interface to return flag indicating whether Horovod was compiled with MPI support.
bool horovod_mpi_built();

// C interface to return flag indicating whether Gloo is enabled.
bool horovod_gloo_enabled();

// C interface to return flag indicating whether Horovod was compiled with Gloo support.
bool horovod_gloo_built();

// C interface to return flag indicating whether Horovod was compiled with NCCL support.
bool horovod_nccl_built();

// C interface to return flag indicating whether Horovod was compiled with DDL support.
bool horovod_ddl_built();

// C interface to return flag indicating whether Horovod was compiled with MLSL support.
bool horovod_mlsl_built();

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

Status EnqueueJoin(std::shared_ptr<OpContext> context,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback);

} // namespace common
} // namespace horovod

#endif // HOROVOD_OPERATIONS_H
