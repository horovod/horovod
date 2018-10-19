// Copyright 2018 Intel Corporation
// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2018 Uber Technologies, Inc.
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

#include <assert.h>
#include <atomic>
#include <cstring>
#include <mpi.h>
#include <mutex>
#include <queue>
#include <string>
#include <sys/syscall.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include "hashes.h"
#include "mlsl.hpp"
#include "mpi_message.h"
#include "operations.h"

/*
 * MLSL provides such collective ops as:
 *      – MLSL::AllReduce:
 *          Perform allreduce on a Tensor, returning the sum
 *          across all MLSL processes in the global distribution.
 *      – MLSL::AllGatherv:
 *          Perform allgather on a Tensor, returning the concatenation of
 *          the tensor on the first dimension across all MLSL processes in the
 *          global communicator.
 *      - MLSL::Bcast:
 *          Perform broadcast on a Tensor, broadcasting Tensor
 *          value from root rank to all other ranks.
 *
 * AllGatherv and Bcast can use MLSL::DT_BYTE type because they don't deal with
 * math and just collect/distribute the data.
 *
 * For better performance one should correctly set HVD_MLSL_BGT_AFFINITY
 * environment variable to pin BackgroundThread so that it wouldn't
 * interfere with other threads.
 *
 */



namespace horovod {
namespace common {

namespace {

#define COMMUNICATOR_TABLE_MAX 1000

#define NULL_VAL nullptr

#define RANK_ZERO 0

#define ERROR 0
#define INFO  1
#define DEBUG 2
#define TRACE 3

#define GET_TID() syscall(SYS_gettid)

#define MLSL_LOG(log_lvl, fmt, ...)					\
  do {									\
    if (log_lvl <= mlsl_log_lvl)					\
    {									\
       char time_buf[20]; /*2016:07:21 14:47:39*/			\
       GetTime(time_buf, 20);						\
       switch (log_lvl) {						\
       case ERROR:							\
         printf("%s: ERROR: (%ld): %s:%u " fmt "\n", time_buf, GET_TID(),\
            __FUNCTION__, __LINE__, ##__VA_ARGS__);			\
         break;							\
       case INFO:							\
         printf("(%ld): %s:%u " fmt "\n", GET_TID(),			\
            __FUNCTION__, __LINE__, ##__VA_ARGS__);			\
         break;							\
       case DEBUG:							\
       case TRACE:							\
         printf("%s: (%ld): %s:%u " fmt "\n", time_buf, GET_TID(),	\
            __FUNCTION__, __LINE__, ##__VA_ARGS__);			\
       default:							\
         assert(0);							\
       }								\
       fflush(stdout);							\
    }									\
  } while (0)

void GetTime(char* buf, size_t bufSize)
{
   time_t timer;
   struct tm* timeInfo = 0;
   time(&timer);
   timeInfo = localtime(&timer);
   assert(timeInfo);
   strftime(buf, bufSize, "%Y:%m:%d %H:%M:%S", timeInfo);
}

enum HvdRequestType
{
  HRT_ALLREDUCE = 0,
  HRT_BCAST     = 1,
  HRT_ALLGATHER = 2,
};

static int mlsl_log_lvl = ERROR;

/* Table storing Tensors to be reduced, keyed by unique name.
 * This table contains everything necessary to do the reduction. */
typedef struct {
  // Name of the tensor.
  std::string tensor_name;
  // Operation context.
  std::shared_ptr<OpContext> context;
  // Input tensor.
  std::shared_ptr<Tensor> tensor;
  // Pre-allocated output tensor.
  std::shared_ptr<Tensor> output;
  // Root rank for broadcast operation.
  int root_rank = 0;
  // Request object for this tensor
  MLSL::CommReq *request;
  // Communicator
  MLSL::Distribution *comm;
  MLSL::Distribution *tmp_comm;
  // A callback to call with the status.
  StatusCallback callback;
  HvdRequestType req_type;
  size_t* size_vec;
} TensorTableEntry;

typedef std::vector<TensorTableEntry> TensorTable;

/* Table storing Tensor metadata on all non-coordinator ranks until
 * rank zero sends communicator for tensor. */
typedef std::unordered_map<std::string, MLSL::Distribution*> NameTable;

// The global state
struct HorovodGlobalState {
  /* An atomic boolean which is set to true when background thread is started.
   * This ensures that only one background thread is spawned */
  std::atomic_flag initialize_flag = ATOMIC_FLAG_INIT;

  // A mutex that needs to be used whenever operations are done.
  std::mutex mutex;

  // Tensors waiting to be allreduced or allgathered.
  TensorTable tensor_table;

  // Background thread running communication.
  std::thread background_thread;

  // Whether the background thread should shutdown.
  std::atomic_bool shut_down {false};

  // Whether MLSL_Init has been completed on the background thread.
  std::atomic_bool initialization_done {false};

  int rank = 0;
  int size = 1;

  ~HorovodGlobalState() {
    /* Make sure that the destructor of the background thread is safe to
     * call. If a thread is still joinable (not detached or complete) its
     * destructor cannot be called */
    if (background_thread.joinable()) {
      shut_down = true;
      background_thread.join();
    }
  }
};

// All the Horovod state that must be stored globally per-process.
static  HorovodGlobalState horovod_global;

static const Status NOT_INITIALIZED_ERROR = Status::PreconditionError(
    "Horovod has not been initialized; use hvd.init().");

static const Status SHUT_DOWN_ERROR = Status::Aborted(
    "Horovod has been shut down. This has been caused by an exception on one "
    "of the rank or an attempt to allreduce, allgather or broadcast a tensor "
    "after one of the ranks has finished execution.");

/* Current implementation of reduction operation supports
 * only float and double data types. For other collective
 * operations a data type of the same size is used */

MLSL::DataType GetMLSLDataType(const std::shared_ptr<Tensor> tensor) {
  switch (tensor->dtype()) {
  case HOROVOD_UINT8:
    return MLSL::DT_BYTE;
  case HOROVOD_FLOAT32:
    return MLSL::DT_FLOAT;
  case HOROVOD_FLOAT64:
    return MLSL::DT_DOUBLE;
  default:
    throw std::logic_error("Type " + MPIDataType_Name(tensor->dtype()) +
                           " is not supported in MLSL.");
  }
}

static void server_affinity_set(int affinity) {
  cpu_set_t cpuset;
  pthread_t current_thread = pthread_self();

  __CPU_ZERO_S(sizeof(cpu_set_t), &cpuset);
  __CPU_SET_S(affinity, sizeof(cpu_set_t), &cpuset);

  if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0)
      perror("setaffinity failed\n");

  // Check if we set the affinity correctly
  if (pthread_getaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0)
      perror("sched_getaffinity failed\n");

  for (int core_idx = 0; core_idx < __CPU_SETSIZE; core_idx++)
       if (__CPU_ISSET_S(core_idx, sizeof(cpu_set_t), &cpuset))
           MLSL_LOG(DEBUG, "BG-thread affinity %d\n", core_idx);
}

void* BackgroundThreadLoop(HorovodGlobalState& state) {

  char* hvd_mlsl_bg_thread_env = NULL;
  int bg_thread_affinity = 0;
  if ((hvd_mlsl_bg_thread_env = getenv("HVD_MLSL_BGT_AFFINITY")) != NULL)
      bg_thread_affinity = atoi(hvd_mlsl_bg_thread_env);

  server_affinity_set(bg_thread_affinity);

  char* log_lvl_env = NULL;
  if ((log_lvl_env = getenv("HVD_MLSL_LOG_LVL")) != NULL)
     mlsl_log_lvl = atoi(log_lvl_env);

  MLSL_LOG(DEBUG, "BG-thread start\n");

  // Initialize MLSL
  MLSL::Environment::GetEnv().Init(NULL, NULL);

  // Get rank to determine if we are rank zero.
  size_t rank = MLSL::Environment::GetEnv().GetProcessIdx();
  bool is_coordinator = rank == 0;

  // Get comm size to determine how many tensors to wait for before reducing.
  size_t size = MLSL::Environment::GetEnv().GetProcessCount();

  state.rank = rank;
  state.size = size;
  state.initialization_done = true;

  MLSL::Distribution* global_dist = MLSL::Environment::GetEnv().CreateDistribution(size, 1);

  std::queue<MLSL::Distribution*> comm_table;
  int next_free_communicator = 0;

  // Special communicator for communicating communicator info
  MPI_Comm newcomm_world;
  MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &newcomm_world);

  // The coordinator sends a SHUTDOWN signal to trigger shutdown.
  bool should_shut_down = false;

  TensorTable current_table;

  NameTable recvname_table;

  do {

    // Lock and create private copy of tensor table
    {
       std::lock_guard<std::mutex> guard(state.mutex);
       for (unsigned int i = 0; i < state.tensor_table.size(); ++i)
            current_table.push_back(state.tensor_table[i]);

       state.tensor_table.clear();
    }

    /* Background thread prepares a pool of predefined communicators organized as a queue.
     * When coordinator is ready to post a collective operation for some tensors it distributes
     * next 'free' communicator idx along with tensor names to all other processes so that
     * every process could use the same communicator with specified tensor names.
     * All the other processes check msg from coordinator. If communicator to use has not been
     * sent by coordinator, they postpone the processing until the communicator idx is received */

    if (is_coordinator)
    {
       for (unsigned int i = 0; i < current_table.size(); ++i)
       {
          auto name = current_table[i].tensor_name;
          if (current_table[i].request == NULL_VAL && current_table[i].comm == NULL_VAL)
          {
            auto rn_it = recvname_table.find(name);
            if (rn_it != recvname_table.end())
            {
                current_table[i].comm = rn_it->second;
            }
            else
            {
              /* Use next free communicator and send tensor name to all other ranks.
               * Communicators are assigned in the order enforced by coordinator rank */
              for (unsigned int r = 1; r < size; r++)
              {
                 MPI_Send(name.c_str(), (int)name.length(), MPI_BYTE, r,
                        next_free_communicator, newcomm_world);
              }

              MLSL::Distribution *newcomm = MLSL::Environment::GetEnv().CreateDistribution(size, 1);
              comm_table.push(newcomm);
              current_table[i].comm = newcomm;
              recvname_table.emplace(name, newcomm);
              next_free_communicator++;
              /* We should be able to remove the below but
               * keep it for now in case this indicates an issue rather than big model */
              if (next_free_communicator == COMMUNICATOR_TABLE_MAX)
                MLSL_LOG(INFO, "Created more than 1K distributions which may indicate an issue if model is not that large");
            }
            MLSL_LOG(DEBUG, "NOW Coord comm assigned %d %p %lu %s\n",
                     next_free_communicator - 1, current_table[i].comm, name.length(), name.c_str());
          }
       }
    }
    else
    {
      // Check for messages from coordinator
      MPI_Status status;
      int test_flag;

      do {
         test_flag = 0;

         MPI_Iprobe(RANK_ZERO, next_free_communicator, newcomm_world, &test_flag, &status);
         if (test_flag)
         {
            int msg_length;
            MPI_Get_count(&status, MPI_BYTE, &msg_length);

            // Get tensor name from MPI into an std::string.
            char* buffer = new char[msg_length];
            MPI_Recv(buffer, msg_length, MPI_BYTE, RANK_ZERO, next_free_communicator, newcomm_world, &status);
            std::string received_name(buffer, (size_t)msg_length);
            delete[] buffer;

            MLSL::Distribution *newcomm = MLSL::Environment::GetEnv().CreateDistribution(size, 1);
            comm_table.push(newcomm);
            next_free_communicator++;
            recvname_table.emplace(received_name, newcomm);
            MLSL_LOG(DEBUG, "NCoord comm added:  %s - %p\n", received_name.c_str(), newcomm);
        }
      } while (test_flag == 1);

      for (unsigned int i = 0; i < current_table.size(); ++i)
      {
        auto name = current_table[i].tensor_name;
        if (current_table[i].request == NULL_VAL && current_table[i].comm == NULL_VAL)
        {
            auto rn_it = recvname_table.find(name);
            if (rn_it != recvname_table.end())
            {
                current_table[i].comm = rn_it->second;
                MLSL_LOG(DEBUG, "NOW NCoord comm assigned %p %lu %s\n", current_table[i].comm, name.length(), name.c_str());
            }
        }
      }
    }

    bool op_completed = false;
    std::vector<TensorTableEntry>::size_type i = 0;

    while (i < current_table.size())
    {
      op_completed = false;
      if (current_table[i].request == NULL_VAL && current_table[i].comm != NULL_VAL)
      {
        // Issue op if communicator assigned for this tensor
        MLSL_LOG(DEBUG, "NOW Allreduce %s %p %p %d %p %p %ld\n",
                 current_table[i].tensor_name.c_str(), current_table[i].tensor->data(),
                 current_table[i].comm, current_table[i].tensor->dtype(), current_table[i].request,
                 NULL_VAL, current_table[i].tensor->size());

        if (current_table[i].req_type == HRT_BCAST)
        {
            current_table[i].request = current_table[i].comm->Bcast(state.rank == current_table[i].root_rank ? (void*) current_table[i].tensor->data() : (void*) current_table[i].output->data(),
                                                                   (int) current_table[i].tensor->size(), MLSL::DT_BYTE, current_table[i].root_rank, MLSL::GT_DATA);
        }
        else if (current_table[i].req_type == HRT_ALLREDUCE)
        {
            current_table[i].request = current_table[i].comm->AllReduce((void*) current_table[i].tensor->data(), (void*) current_table[i].output->data(),
                                                                       (int) current_table[i].tensor->shape().num_elements(), GetMLSLDataType(current_table[i].tensor), MLSL::RT_SUM, MLSL::GT_DATA);
        }
        else if (current_table[i].req_type == HRT_ALLGATHER && current_table[i].size_vec != NULL)
        {
            size_t mySize = current_table[i].tensor->shape().dim_size(0);
            std::vector<size_t> rcvCounts(size);
            for (unsigned int j = 0; j < size; j++)
                rcvCounts[j] = sizeof(size_t);

            current_table[i].request = current_table[i].comm->AllGatherv(&mySize, sizeof(size_t), (void*) current_table[i].size_vec, rcvCounts.data(), MLSL::DT_BYTE, MLSL::GT_DATA);
            current_table[i].tmp_comm = current_table[i].comm;
        }

        current_table[i].comm = NULL_VAL;
        MLSL_LOG(DEBUG, "NOW Allreduce done %s %p %p %d %p %p %ld\n",
                 current_table[i].tensor_name.c_str(), current_table[i].tensor->data(),
                 current_table[i].comm, current_table[i].tensor->dtype(), current_table[i].request,
                 NULL_VAL, current_table[i].tensor->size());
      }
      else if (current_table[i].request != NULL_VAL && current_table[i].comm == NULL_VAL)
      {
        bool testflag = 0;

        MLSL::Environment::GetEnv().Test(current_table[i].request, &testflag);

        if (testflag)
        {
            if (current_table[i].req_type == HRT_ALLGATHER && current_table[i].size_vec != NULL)
            {
                TensorShape single_slice_shape;
                size_t total_first_dimension_size = 0;
                size_t total_size_of_other_dims = 1;
                for (int j = 1; j < current_table[i].tensor->shape().dims(); j++)
                {
                    single_slice_shape.AddDim(current_table[i].tensor->shape().dim_size(j));
                    total_size_of_other_dims *= current_table[i].tensor->shape().dim_size(j);
                }

                std::vector<size_t> rcounts(size);
                size_t element_bytesize = current_table[i].tensor->size()/current_table[i].tensor->shape().num_elements();
                for (unsigned int j = 0; j < size; j++)
                {
                    rcounts[j] = current_table[i].size_vec[j] * total_size_of_other_dims * element_bytesize;
                    total_first_dimension_size += current_table[i].size_vec[j] ;
                }

                /* Allgather output will have shape of:
                 * (sum of first dimension of every tensor) x (tensor slice shape) */
                TensorShape output_shape;
                output_shape.AddDim((int64_t)total_first_dimension_size);
                output_shape.AppendShape(single_slice_shape);

                Status status = current_table[i].context->AllocateOutput(output_shape, &current_table[i].output);
                if (!status.ok())
                {
                    current_table[i].callback(status);
                    return NULL;
                }

                current_table[i].request = current_table[i].tmp_comm->AllGatherv((void*) current_table[i].tensor->data(),
                                                                                 current_table[i].tensor->size(),
                                                                                 (void*) current_table[i].output->data(), rcounts.data(),
                                                                                 MLSL::DT_BYTE, MLSL::GT_DATA);

                delete current_table[i].size_vec;
                current_table[i].size_vec = NULL;
                current_table[i].comm = NULL_VAL;
                MLSL_LOG(DEBUG, "Test success for intermediate AllGather call %s %p\n",
                                current_table[i].tensor_name.c_str(), current_table[i].tensor->data());
            }
            else
            {
                // Remove request from request_table and execute tensor callback.
                current_table[i].callback(Status::OK());
                current_table[i].request = NULL_VAL;
                op_completed = true;
                MLSL_LOG(DEBUG, "Test success %s %p\n",
                current_table[i].tensor_name.c_str(), current_table[i].tensor->data());
            }
        }
      }
      if (op_completed == true)
      {
        current_table.erase(current_table.begin() + i);
        break;
      } else {
        ++i;
      }
    }// do

    volatile bool check_shut_down = state.shut_down;
    if (check_shut_down == true)
    {
      should_shut_down = check_shut_down;
      global_dist->Barrier(MLSL::GT_GLOBAL);
      for (unsigned int i = 0; i < current_table.size(); ++i)
      {
        if (current_table[i].request != NULL_VAL)
        {
            should_shut_down = false;
            break;
        }
      }
    }
  } while (should_shut_down == false);


  MLSL_LOG(DEBUG, "BG-thread comm destroy\n");
  // Destroy MLSL communicators
  MLSL::Environment::GetEnv().DeleteDistribution(global_dist);

  while (!comm_table.empty())
  {
      MLSL::Distribution *comm = comm_table.front();
      comm_table.pop();
      MLSL::Environment::GetEnv().DeleteDistribution(comm);
  }

  MLSL_LOG(DEBUG, "BG-thread MLSL Finalize\n");
  MLSL::Environment::GetEnv().Finalize();

  return NULL;
}

/* Start Horovod background thread. Ensure that this is
 * only done once no matter how many times this function is called */
void InitializeHorovodOnce()
{
  // Ensure background thread is only started once.
  if (!horovod_global.initialize_flag.test_and_set())
  {
    horovod_global.background_thread =
         std::thread(BackgroundThreadLoop, std::ref(horovod_global));
  }

  MLSL_LOG(DEBUG, "BG-thread spawn done\n");

  // Wait to ensure that the background thread has finished initializing MPI.
  while (!horovod_global.initialization_done)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  MLSL_LOG(DEBUG, "BG-thread init done\n");

  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  MLSL_LOG(DEBUG, "PID %d on %s ready for attach\n", getpid(), hostname);
}

} //namespace

// Check that Horovod is initialized.
Status CheckInitialized()
{
  if (!horovod_global.initialization_done)
  {
    return NOT_INITIALIZED_ERROR;
  }
  return Status::OK();
}

extern "C" {

void horovod_init(const int* ranks, int nranks) { InitializeHorovodOnce(); }

void horovod_init_comm(MPI_Comm comm)
{
  InitializeHorovodOnce();
}

void horovod_shutdown()
{
  if (horovod_global.background_thread.joinable())
  {
    horovod_global.shut_down = true;
    horovod_global.background_thread.join();
    // Reset the initialization flag to allow restarting with horovod_init(...)
    horovod_global.initialize_flag.clear();
    horovod_global.shut_down = false;
  }

}

int horovod_rank()
{
  if (!horovod_global.initialization_done)
  {
    return -1;
  }
  return horovod_global.rank;
}

int horovod_local_rank()
{
  return horovod_rank();
}

int horovod_size()
{
  if (!horovod_global.initialization_done)
  {
    return -1;
  }
  return horovod_global.size;
}
} // extern "C"

/* MLSL must be initialized and the background thread must be running before
 * this function is called */
Status EnqueueTensorAllreduce(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback) {
  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.output = output;
  e.request = NULL_VAL;
  e.comm = NULL_VAL;
  e.callback = callback;
  e.req_type = HRT_ALLREDUCE;

  MLSL_LOG(DEBUG, "Enqueue allreduce for tensor with address: %p\n", e.tensor->data());

  std::lock_guard<std::mutex> guard(horovod_global.mutex);
  if (!horovod_global.shut_down) {
    horovod_global.tensor_table.push_back(e);
    return Status::OK();
  } else {
    return SHUT_DOWN_ERROR;
  }

}

Status EnqueueTensorAllgather(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback) {
  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.request = NULL_VAL;
  e.comm = NULL_VAL;
  e.callback = callback;
  e.req_type = HRT_ALLGATHER;
  e.size_vec = new size_t[horovod_global.size];

  MLSL_LOG(DEBUG, "Enqueue allgather for tensor with address: %p\n", e.tensor->data());

  std::lock_guard<std::mutex> guard(horovod_global.mutex);
  if (!horovod_global.shut_down) {
    horovod_global.tensor_table.push_back(e);
    return Status::OK();
  } else {
    return SHUT_DOWN_ERROR;
  }
}

Status EnqueueTensorBroadcast(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output, int root_rank,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback) {
  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.output = output;
  e.root_rank = root_rank;
  e.request = NULL_VAL;
  e.comm = NULL_VAL;
  e.callback = callback;
  e.req_type = HRT_BCAST;

  MLSL_LOG(DEBUG, "Enqueue bcast for tensor with address: %p\n", e.tensor->data());

  std::lock_guard<std::mutex> guard(horovod_global.mutex);
  if (!horovod_global.shut_down) {
    horovod_global.tensor_table.push_back(e);
    return Status::OK();
  } else {
    return SHUT_DOWN_ERROR;
  }
}

} // namespace common
} // namespace horovod
