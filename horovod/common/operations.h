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

// C interface to initialize Horovod. Returns false on failure.
bool horovod_init(const int* ranks, int nranks, const int* process_set_ranks,
                  const int* process_set_sizes, int num_process_sets);

#if HAVE_MPI
// C interface to initialize Horovod with an array of existing MPI
// communicators. We will build matching process sets for these in addition to
// those defined via rank indices. Returns false on failure.
bool horovod_init_multi_comm(MPI_Comm* comm, int ncomms,
                             const int* process_set_ranks_via_ranks,
                             const int* process_set_sizes_via_ranks,
                             int num_process_sets_via_ranks);
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

// C interface to return integer indicating whether Horovod was compiled with NCCL support.
// Returns NCCL_VERSION_CODE if NCCL is available, else returns 0.
int horovod_nccl_built();

// C interface to return flag indicating whether Horovod was compiled with DDL support.
bool horovod_ddl_built();

// C interface to return flag indicating whether Horovod was compiled with CCL support.
bool horovod_ccl_built();

// C interface to return flag indicating whether Horovod was compiled with CUDA support.
bool horovod_cuda_built();

// C interface to return flag indicating whether Horovod was compiled with ROCm support.
bool horovod_rocm_built();

// C interface to return value of the ReduceOp::AVERAGE enum field.
int horovod_reduce_op_average();

// C interface to return value of the ReduceOp::SUM enum field.
int horovod_reduce_op_sum();

// C interface to return value of the ReduceOp::ADASUM enum field.
int horovod_reduce_op_adasum();

// C interface to return value of the ReduceOp::MIN enum field.
int horovod_reduce_op_min();

// C interface to return value of the ReduceOp::MAX enum field.
int horovod_reduce_op_max();

// C interface to return value of the ReduceOp::PRODUCT enum field.
int horovod_reduce_op_product();

extern const int HOROVOD_PROCESS_SET_ERROR_INIT;
extern const int HOROVOD_PROCESS_SET_ERROR_DYNAMIC;
extern const int HOROVOD_PROCESS_SET_ERROR_UNKNOWN_SET;
extern const int HOROVOD_PROCESS_SET_ERROR_FOREIGN_SET;
extern const int HOROVOD_PROCESS_SET_ERROR_EXISTING_SET;
extern const int HOROVOD_PROCESS_SET_ERROR_SHUTDOWN;

// C interface to register a new process set containing the given ranks
// (blocking). Returns positive process set id or an error code:
// HOROVOD_PROCESS_SET_ERROR_EXISTING_SET if a process set containing the
// same ranks (after sorting) has been added before,
// HOROVOD_PROCESS_SET_ERROR_INIT if Horovod is not initialized,
// HOROVOD_PROCESS_SET_ERROR_SHUTDOWN if Horovod is shutting down,
// HOROVOD_PROCESS_SET_ERROR_DYNAMIC if dynamic process sets are not enabled,
int horovod_add_process_set(const int *ranks, int nranks);

// C interface to deregister a previously registered process set (blocking).
// Returns process_set_id or an error code:
// HOROVOD_PROCESS_SET_ERROR_INIT if Horovod is not initialized,
// HOROVOD_PROCESS_SET_ERROR_SHUTDOWN if Horovod is shutting down,
// HOROVOD_PROCESS_SET_ERROR_DYNAMIC if dynamic process sets are not enabled,
// HOROVOD_PROCESS_SET_ERROR_UNKNOWN_SET if that process set is unknown,
int horovod_remove_process_set(int process_set_id);

// C interface to return the rank of this process counted in the specified
// process set or an error code:
// HOROVOD_PROCESS_SET_ERROR_INIT if Horovod is not initialized,
// HOROVOD_PROCESS_SET_ERROR_FOREIGN_SET if the process is not part of this set,
// HOROVOD_PROCESS_SET_ERROR_UNKNOWN_SET if the process set is unknown,
int horovod_process_set_rank(int process_set_id);

// C interface to return the size of the specified process set or an error code:
// HOROVOD_PROCESS_SET_ERROR_INIT if Horovod is not initialized,
// HOROVOD_PROCESS_SET_ERROR_UNKNOWN_SET if the process set is unknown,
int horovod_process_set_size(int process_set_id);

// C interface to return 0 or 1 depending on whether the current process is
// included in the specified process set or an error code:
// HOROVOD_PROCESS_SET_ERROR_INIT if Horovod is not initialized,
// HOROVOD_PROCESS_SET_ERROR_UNKNOWN_SET if the process set is unknown,
int horovod_process_set_included(int process_set_id);

// C interface to return the current number of process sets.
int horovod_number_of_process_sets();

// C interface to assign the ids of all process sets to the preallocated array.
void horovod_process_set_ids(int* ids_prealloc);

// C interface to assign the ranks belonging to the process sets with the given
// id to the preallocated array. Returns 0 or an error code:
// HOROVOD_PROCESS_SET_ERROR_INIT if Horovod is not initialized,
// HOROVOD_PROCESS_SET_ERROR_UNKNOWN_SET if the process set is unknown,
int horovod_process_set_ranks(int id, int* ranks_prealloc);

#if HAVE_MPI
// C interface to return process set id corresponding to processes belonging
// to this MPI communicator or an error code:
// HOROVOD_PROCESS_SET_ERROR_INIT if Horovod is not initialized,
// HOROVOD_PROCESS_SET_ERROR_UNKNOWN_SET if there is no process set
// corresponding to this communicator.
int horovod_comm_process_set(MPI_Comm comm);
#endif // HAVE_MPI

}

Status EnqueueTensorAllreduce(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              ReadyEventList ready_event_list,
                              std::string name, int device,
                              StatusCallback callback,
                              ReduceOp reduce_op = ReduceOp::SUM,
                              double prescale_factor = 1.0,
                              double postscale_factor = 1.0,
                              int32_t process_set_id = 0);

Status EnqueueTensorAllreduces(std::vector<std::shared_ptr<OpContext>>& contexts,
                               std::vector<std::shared_ptr<Tensor>>& tensors,
                               std::vector<std::shared_ptr<Tensor>>& outputs,
                               std::vector<ReadyEventList>& ready_event_lists,
                               std::vector<std::string>& names,
                               int device,
                               std::vector<StatusCallback>& callbacks,
                               ReduceOp reduce_op = ReduceOp::SUM,
                               double prescale_factor = 1.0,
                               double postscale_factor = 1.0,
                               int32_t process_set_id = 0);

Status EnqueueTensorAllgather(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              ReadyEventList ready_event_list,
                              const std::string& name, int device,
                              StatusCallback callback,
                              int32_t process_set_id = 0);

Status
EnqueueTensorAllgathers(std::vector<std::shared_ptr<OpContext>>& contexts,
                        std::vector<std::shared_ptr<Tensor>>& tensors,
                        std::vector<ReadyEventList>& ready_event_lists,
                        std::vector<std::string>& names, int device,
                        std::vector<StatusCallback>& callbacks,
                        int32_t process_set_id = 0);

Status EnqueueTensorBroadcast(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output, int root_rank,
                              ReadyEventList ready_event_list,
                              const std::string& name, int device,
                              StatusCallback callback,
                              int32_t process_set_id = 0);

Status EnqueueTensorAlltoall(std::shared_ptr<OpContext> context,
                             std::shared_ptr<Tensor> tensor,
                             std::shared_ptr<Tensor> splits,
                             ReadyEventList ready_event_list,
                             const std::string& name, int device,
                             StatusCallback callback,
                             int32_t process_set_id = 0);

Status EnqueueTensorReducescatter(
    std::shared_ptr<OpContext> context, std::shared_ptr<Tensor> tensor,
    ReadyEventList ready_event_list, const std::string& name, int device,
    StatusCallback callback, ReduceOp reduce_op = ReduceOp::SUM,
    int32_t process_set_id = 0, double prescale_factor = 1.0,
    double postscale_factor = 1.0);

Status EnqueueTensorReducescatters(
    std::vector<std::shared_ptr<OpContext>>& contexts,
    std::vector<std::shared_ptr<Tensor>>& tensors,
    std::vector<ReadyEventList>& ready_event_lists,
    std::vector<std::string>& names, int device,
    std::vector<StatusCallback>& callbacks, ReduceOp reduce_op = ReduceOp::SUM,
    int32_t process_set_id = 0, double prescale_factor = 1.0,
    double postscale_factor = 1.0);

Status EnqueueJoin(std::shared_ptr<OpContext> context,
                   std::shared_ptr<Tensor> output_last_joined_rank,
                   ReadyEventList ready_event_list,
                   const std::string& name, int device,
                   StatusCallback callback,
                   int32_t process_set_id = 0);

Status EnqueueBarrier(StatusCallback callback,
                   int32_t process_set_id = 0);

} // namespace common
} // namespace horovod

#endif // HOROVOD_OPERATIONS_H
