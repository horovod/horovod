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

#include <atomic>
#include <cassert>
#include <cstring>
#include <queue>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#if HAVE_CUDA
#include <cuda_runtime.h>
#endif

#if HAVE_NCCL
#include <nccl.h>
#endif

#if HAVE_DDL
#include <ddl.hpp>
#endif

#define OMPI_SKIP_MPICXX
#include "fusion_buffer_manager.h"
#include "half.h"
#include "hashes.h"
#include "mpi.h"
#include "message.h"
#include "operations.h"
#include "parameter_manager.h"
#include "timeline.h"
#include "logging.h"

/*
 * Allreduce, Allgather and Broadcast Ops.
 *
 * This module implements MPI ops for allgather, allreduce and broadcast, which
 * do optimized gathers, reductions and broadcasts and can take advantage of
 * hardware-optimized communication libraries through the MPI implementation.
 *
 * The primary logic of the allreduce, allgather and broadcast are in MPI and
 * NCCL implementations. The background thread which facilitates MPI operations
 * is run in BackgroundThreadLoop(). The provided ops are:
 *      – HorovodAllreduce:
 *          Perform an allreduce on a Tensor, returning the sum
 *          across all MPI processes in the global communicator.
 *      – HorovodAllgather:
 *          Perform an allgather on a Tensor, returning the concatenation of
 *          the tensor on the first dimension across all MPI processes in the
 *          global communicator.
 *      - HorovodBroadcast:
 *          Perform a broadcast on a Tensor, broadcasting Tensor
 *          value from root rank to all other ranks.
 *
 * Additionally, this library provides C APIs to initialize Horovod and query
 * rank, local rank and world size.  These are used in Python directly through
 * ctypes.
 */

namespace horovod {
namespace common {

namespace {

// Table storing Tensors to be reduced, keyed by unique name.
// This table contains everything necessary to do the reduction.
struct TensorTableEntry {
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
  // Event indicating that data is ready.
  std::shared_ptr<ReadyEvent> ready_event;
  // GPU to do reduction on, or CPU_DEVICE_ID in case of CPU.
  int device = CPU_DEVICE_ID;
  // A callback to call with the status.
  StatusCallback callback;
};
using TensorTable = std::unordered_map<std::string, TensorTableEntry>;

// Table for storing Tensor metadata on rank zero. This is used for error
// checking, stall checking and size calculations, as well as determining
// when a reduction is ready to be done (when all nodes are ready to do it).
using MessageTable = std::unordered_map<
    std::string,
    std::tuple<std::vector<Request>, std::chrono::steady_clock::time_point>>;

// The global state required for the MPI ops.
//
// MPI is a library that stores a lot of global per-program state and often
// requires running on a single thread. As a result, we have to have a single
// background thread responsible for all MPI operations, and communicate with
// that background thread through global state.
struct HorovodGlobalState {
  // An atomic boolean which is set to true when background thread is started.
  // This ensures that only one background thread is spawned.
  std::atomic_flag initialize_flag = ATOMIC_FLAG_INIT;

  // A mutex that needs to be used whenever MPI operations are done.
  std::mutex mutex;

  // Tensors waiting to be allreduced or allgathered.
  TensorTable tensor_table;

  // Queue of requests waiting to be sent to the coordinator node.
  std::queue<Request> message_queue;

  // Background thread running MPI communication.
  std::thread background_thread;

  // Whether the background thread should shutdown.
  std::atomic_bool shut_down{false};

  // Whether Horovod should finalize MPI (only if it has initialized it).
  bool should_finalize = false;

  // Only exists on the coordinator node (rank zero). Maintains a count of
  // how many nodes are ready to allreduce every tensor (keyed by tensor
  // name) and time point when tensor started allreduce op.
  std::unique_ptr<MessageTable> message_table;

  // Time point when coordinator last checked for stalled tensors.
  std::chrono::steady_clock::time_point last_stall_check;

  // Flag indicating whether to perform stall tensor check.
  bool perform_stall_check = true;

  // Timeline writer.
  Timeline timeline;

  // Flag indicating whether to mark cycles in the timeline.
  bool mark_cycles_in_timeline = false;

  ParameterManager param_manager;

  // Encapsulates the fusion buffers, handles resizing and auto-tuning of buffer
  // size.
  FusionBufferManager fusion_buffer;

  // Time point when last cycle started.
  std::chrono::steady_clock::time_point last_cycle_start;

  // Whether MPI_Init has been completed on the background thread.
  std::atomic_bool initialization_done{false};

  // The MPI rank, local rank, size, local size, flag indicating whether MPI
  // multi-threading is supported, ranks from which the MPI communicator will
  // be made and the communicator itself.
  int rank = 0;
  int local_rank = 0;
  int cross_rank = 0;
  int size = 1;
  int local_size = 1;
  int cross_size = 1;
  bool mpi_threads_supported = false;
  bool is_homogeneous = false;
  std::vector<int> ranks;

  // COMM_WORLD ranks of processes running on this node.
  std::vector<int> local_comm_ranks;

  // Numbers of ranks running per node
  std::vector<int> local_sizes;

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

  // Pointer to shared buffer for allgather
  void* shared_buffer = nullptr;

  // Current shared buffer size
  int64_t shared_buffer_size = 0;

// The CUDA stream used for data transfers and within-allreduce operations.
// A naive implementation would use the TensorFlow StreamExecutor CUDA
// stream. However, the allreduce and allgather require doing memory copies
// and kernel executions (for accumulation of values on the GPU). However,
// the subsequent operations must wait for those operations to complete,
// otherwise MPI (which uses its own stream internally) will begin the data
// transfers before the CUDA calls are complete. In order to wait for those
// CUDA operations, if we were using the TensorFlow stream, we would have to
// synchronize that stream; however, other TensorFlow threads may be
// submitting more work to that stream, so synchronizing on it can cause the
// allreduce to be delayed, waiting for compute totally unrelated to it in
// other parts of the graph. Overlaying memory transfers and compute during
// backpropagation is crucial for good performance, so we cannot use the
// TensorFlow stream, and must use our own stream.
#if HAVE_CUDA
  std::unordered_map<int, cudaStream_t> streams;
#endif
#if HAVE_NCCL
  std::unordered_map<std::vector<int32_t>, ncclComm_t> nccl_comms;
#endif

  // Will be set to true after initialization when ddl is used
  bool ddl_initialized = false;
  int32_t ddl_local_device_id = 0;

// We reuse CUDA events as it appears that their creation carries non-zero cost.
#if HAVE_CUDA
  std::unordered_map<int, std::queue<cudaEvent_t>> cuda_events;
  std::mutex cuda_events_mutex;
#endif

  ~HorovodGlobalState() {
    // Make sure that the destructor of the background thread is safe to
    // call. If a thread is still joinable (not detached or complete) its
    // destructor cannot be called.
    if (background_thread.joinable()) {
      shut_down = true;
      background_thread.join();
    }
  }
};

// All the Horovod state that must be stored globally per-process.
HorovodGlobalState horovod_global;

// For clarify in argument lists.
#define RANK_ZERO 0

// Stall-check warning time
#define STALL_WARNING_TIME std::chrono::seconds(60)

const Status NOT_INITIALIZED_ERROR = Status::PreconditionError(
    "Horovod has not been initialized; use hvd.init().");

const Status SHUT_DOWN_ERROR = Status::UnknownError(
    "Horovod has been shut down. This was caused by an exception on one of the "
    "ranks or an attempt to allreduce, allgather or broadcast a tensor after "
    "one of the ranks finished execution. If the shutdown was caused by an "
    "exception, you should see the exception in the log before the first "
    "shutdown message.");

const Status DUPLICATE_NAME_ERROR = Status::InvalidArgument(
    "Requested to allreduce, allgather, or broadcast a tensor with the same "
    "name as another tensor that is currently being processed.  If you want "
    "to request another tensor, use a different tensor name.");

#define OP_ERROR(entries, error_message)                                       \
  {                                                                            \
    for (auto& e : (entries)) {                                                \
      timeline.End(e.tensor_name, nullptr);                                    \
      e.callback(Status::UnknownError(error_message));                         \
    }                                                                          \
    return;                                                                    \
  }

// Store the Request for a name, and return whether the total count of
// Requests for that tensor is now equal to the MPI size (and thus we are
// ready to reduce the tensor).
bool IncrementTensorCount(std::unique_ptr<MessageTable>& message_table,
                          const Request& msg, int mpi_size) {
  auto& name = msg.tensor_name();
  auto& timeline = horovod_global.timeline;
  auto table_iter = message_table->find(name);
  if (table_iter == message_table->end()) {
    std::vector<Request> messages = {msg};
    messages.reserve(static_cast<unsigned long>(mpi_size));
    auto now = std::chrono::steady_clock::now();
    message_table->emplace(name, std::make_tuple(std::move(messages), now));
    table_iter = message_table->find(name);
    timeline.NegotiateStart(name, msg.request_type());
  } else {
    std::vector<Request>& messages = std::get<0>(table_iter->second);
    messages.push_back(msg);
  }

  timeline.NegotiateRankReady(name, msg.request_rank());

  std::vector<Request>& messages = std::get<0>(table_iter->second);
  int count = (int)messages.size();
  bool ready_to_reduce = count == mpi_size;
  if (ready_to_reduce) {
    timeline.NegotiateEnd(name);
  }
  return ready_to_reduce;
}

// Once a tensor is ready to be reduced, the coordinator sends a Response
// instructing all ranks to start the reduction to all ranks. The Response
// also contains error messages in case the submitted Requests were not
// valid (for example, contained mismatched shapes or types).
//
// Constructing the Response, thus, requires a whole lot of error checking.
Response ConstructResponse(std::unique_ptr<MessageTable>& message_table,
                           std::string name) {
  bool error = false;
  auto it = message_table->find(name);
  assert(it != message_table->end());

  std::vector<Request>& requests = std::get<0>(it->second);
  assert(requests.size() > 0);

  std::ostringstream error_message_stream;

  // Check that all data types of tensors being reduced, gathered or broadcasted
  // are identical.
  auto data_type = requests[0].tensor_type();
  for (unsigned int i = 1; i < requests.size(); ++i) {
    auto request_type = requests[i].tensor_type();
    if (data_type != request_type) {
      error = true;
      error_message_stream << "Mismatched data types: One rank had type "
                           << DataType_Name(data_type)
                           << ", but another rank had type "
                           << DataType_Name(request_type) << ".";
      break;
    }
  }

  // Check that all requested operations are the same
  auto message_type = requests[0].request_type();
  for (unsigned int i = 1; i < requests.size(); ++i) {
    if (error) {
      break;
    }

    auto request_type = requests[i].request_type();
    if (message_type != request_type) {
      error = true;
      error_message_stream << "Mismatched MPI operations: One rank did an "
                           << Request::RequestType_Name(message_type)
                           << ", but another rank did an "
                           << Request::RequestType_Name(request_type) << ".";
      break;
    }
  }

  // If we are doing an allreduce or broadcast, check that all tensor shapes are
  // identical.
  if (message_type == Request::ALLREDUCE ||
      message_type == Request::BROADCAST) {
    TensorShape tensor_shape;
    for (auto dim : requests[0].tensor_shape()) {
      tensor_shape.AddDim(dim);
    }
    for (unsigned int i = 1; i < requests.size(); ++i) {
      if (error) {
        break;
      }

      TensorShape request_shape;
      for (auto dim : requests[i].tensor_shape()) {
        request_shape.AddDim(dim);
      }
      if (tensor_shape != request_shape) {
        error = true;
        error_message_stream
            << "Mismatched " << Request::RequestType_Name(message_type)
            << " tensor shapes: One rank sent a tensor of shape "
            << tensor_shape.DebugString()
            << ", but another rank sent a tensor of shape "
            << request_shape.DebugString() << ".";
        break;
      }
    }
  }

  // If we are doing an allgather, make sure all but the first dimension are
  // the same. The first dimension may be different and the output tensor is
  // the sum of the first dimension. Collect the sizes by rank.
  std::vector<int64_t> tensor_sizes(requests.size());
  if (message_type == Request::ALLGATHER) {
    TensorShape tensor_shape;
    for (auto dim : requests[0].tensor_shape()) {
      tensor_shape.AddDim(dim);
    }

    if (tensor_shape.dims() == 0) {
      error = true;
      error_message_stream << "Rank zero tried to "
                           << Request::RequestType_Name(message_type)
                           << " a rank-zero tensor.";
    } else {
      tensor_sizes[requests[0].request_rank()] = tensor_shape.dim_size(0);
    }

    for (unsigned int i = 1; i < requests.size(); ++i) {
      if (error) {
        break;
      }

      TensorShape request_shape;
      for (auto dim : requests[i].tensor_shape()) {
        request_shape.AddDim(dim);
      }
      if (tensor_shape.dims() != request_shape.dims()) {
        error = true;
        error_message_stream
            << "Mismatched " << Request::RequestType_Name(message_type)
            << " tensor shapes: One rank sent a tensor of rank "
            << tensor_shape.dims()
            << ", but another rank sent a tensor of rank "
            << request_shape.dims() << ".";
        break;
      }

      bool dim_mismatch = false;
      for (int dim = 1; dim < tensor_shape.dims(); ++dim) {
        if (tensor_shape.dim_size(dim) != request_shape.dim_size(dim)) {
          error = true;
          error_message_stream
              << "Mismatched " << Request::RequestType_Name(message_type)
              << " tensor shapes: One rank sent a tensor with dimension " << dim
              << " equal to " << tensor_shape.dim_size(dim)
              << ", but another rank sent a tensor with dimension " << dim
              << " equal to " << request_shape.dim_size(dim) << ".";
          dim_mismatch = true;
          break;
        }
      }
      if (dim_mismatch) {
        break;
      }

      tensor_sizes[requests[i].request_rank()] = request_shape.dim_size(0);
    }
  }

  // If we are doing a broadcast, check that all root ranks are identical.
  if (message_type == Request::BROADCAST) {
    int first_root_rank = requests[0].root_rank();
    for (unsigned int i = 1; i < requests.size(); ++i) {
      if (error) {
        break;
      }

      int this_root_rank = requests[i].root_rank();
      if (first_root_rank != this_root_rank) {
        error = true;
        error_message_stream
            << "Mismatched " << Request::RequestType_Name(message_type)
            << " root ranks: One rank specified root rank " << first_root_rank
            << ", but another rank specified root rank " << this_root_rank
            << ".";
        break;
      }
    }
  }

  bool first_device_is_cpu = requests[0].device() == CPU_DEVICE_ID;
  for (unsigned int i = 1; i < requests.size(); ++i) {
    if (error) {
      break;
    }

    bool this_device_is_cpu = requests[i].device() == CPU_DEVICE_ID;
    if (first_device_is_cpu != this_device_is_cpu) {
      error = true;
      error_message_stream
          << "Mismatched " << Request::RequestType_Name(message_type)
          << " CPU/GPU device selection: One rank specified device "
          << (first_device_is_cpu ? "CPU" : "GPU")
          << ", but another rank specified device "
          << (this_device_is_cpu ? "CPU" : "GPU") << ".";
      break;
    }
  }
  std::vector<int32_t> devices(requests.size());
  for (auto& request : requests) {
    devices[request.request_rank()] = request.device();
  }

  Response response;
  response.add_tensor_name(name);
  if (error) {
    std::string error_message = error_message_stream.str();
    response.set_response_type(Response::ERROR);
    response.set_error_message(error_message);
  } else if (message_type == Request::ALLGATHER) {
    response.set_response_type(Response::ALLGATHER);
    for (auto dim : tensor_sizes) {
      response.add_tensor_size(dim);
    }
  } else if (message_type == Request::ALLREDUCE) {
    response.set_response_type(Response::ALLREDUCE);
  } else if (message_type == Request::BROADCAST) {
    response.set_response_type(Response::BROADCAST);
  }
  response.set_devices(devices);

  // Clear all queued up requests for this name. They are now taken care of
  // by the constructed MPI response.
  message_table->erase(it);

  return response;
}

MPI_Datatype GetMPIDataType(const std::shared_ptr<Tensor> tensor) {
  switch (tensor->dtype()) {
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
    return horovod_global.mpi_float16_t;
  case HOROVOD_FLOAT32:
    return MPI_FLOAT;
  case HOROVOD_FLOAT64:
    return MPI_DOUBLE;
  case HOROVOD_BOOL:
    return MPI_C_BOOL;
  default:
    throw std::logic_error("Type " + DataType_Name(tensor->dtype()) +
                           " is not supported in MPI mode.");
  }
}

// Return the total byte size of the final allgathered output tensor
int64_t TotalByteSizeOfAllgatherOutput(const std::vector<int64_t> &tensor_sizes,
                                       const TensorTableEntry entry) {
  int64_t total_dimension_size = 0;
  for (auto sz : tensor_sizes) {
    total_dimension_size += sz;
  }
  // Every tensor participating in Allgather operation may have
  // different first dimension size, but the rest of dimensions are same
  // for all tensors.  Here we get shape of tensor sliced by first
  // dimension. Allgather output will have shape of: (sum of first
  // dimension of every tensor) x (tensor slice shape).
  int64_t total_count_of_output_entries = total_dimension_size;
  for (int i = 1; i < entry.tensor->shape().dims(); ++i) {
    total_count_of_output_entries *= entry.tensor->shape().dim_size(i);
  }
  int element_size;
  MPI_Type_size(GetMPIDataType(entry.tensor), &element_size);
  int64_t total_byte_size_of_output =
      total_count_of_output_entries * element_size;

  return total_byte_size_of_output;
}

#if HAVE_NCCL
ncclDataType_t GetNCCLDataType(const std::shared_ptr<Tensor> tensor) {
  switch (tensor->dtype()) {
  case HOROVOD_INT32:
    return ncclInt32;
  case HOROVOD_INT64:
    return ncclInt64;
  case HOROVOD_FLOAT16:
    return ncclFloat16;
  case HOROVOD_FLOAT32:
    return ncclFloat32;
  case HOROVOD_FLOAT64:
    return ncclFloat64;
  default:
    throw std::logic_error("Type " + DataType_Name(tensor->dtype()) +
                           " is not supported in NCCL mode.");
  }
}
#endif

#if HAVE_DDL
DDL_Type GetDDLDataType(const std::shared_ptr<Tensor> tensor) {
  switch (tensor->dtype()) {
  case HOROVOD_FLOAT32:
    return DDL_TYPE_FLOAT;
  default:
    throw std::logic_error("Type " + DataType_Name(tensor->dtype()) +
                           " is not supported in DDL mode.");
  }
}
#endif

#define MPI_CHECK(entries, op_name, op)                                        \
  {                                                                            \
    auto mpi_result = (op);                                                    \
    if (mpi_result != MPI_SUCCESS) {                                           \
      for (auto& e : (entries)) {                                              \
        timeline.End(e.tensor_name, nullptr);                                  \
        e.callback(Status::UnknownError(                                       \
            std::string(op_name) + " failed, see MPI output for details."));   \
      }                                                                        \
      return;                                                                  \
    }                                                                          \
  }

#define CUDA_CHECK(entries, op_name, op)                                       \
  {                                                                            \
    auto cuda_result = (op);                                                   \
    if (cuda_result != cudaSuccess) {                                          \
      for (auto& e : (entries)) {                                              \
        timeline.End(e.tensor_name, nullptr);                                  \
        e.callback(Status::UnknownError(std::string(op_name) + " failed: " +   \
                                        cudaGetErrorString(cuda_result)));     \
      }                                                                        \
      return;                                                                  \
    }                                                                          \
  }

#define NCCL_CHECK(entries, op_name, op)                                       \
  {                                                                            \
    auto nccl_result = (op);                                                   \
    if (nccl_result != ncclSuccess) {                                          \
      for (auto& e : (entries)) {                                              \
        timeline.End(e.tensor_name, nullptr);                                  \
        e.callback(Status::UnknownError(std::string(op_name) + " failed: " +   \
                                        ncclGetErrorString(nccl_result)));     \
      }                                                                        \
      return;                                                                  \
    }                                                                          \
  }

#define DDL_CHECK(entries, op_name, op)                                        \
  {                                                                            \
    auto ddl_result = (op);                                                    \
    if (ddl_result != DDL_SUCCESS) {                                           \
      for (auto& e : (entries)) {                                              \
        timeline.End(e.tensor_name, nullptr);                                  \
        e.callback(Status::UnknownError(std::string(op_name) + " failed."));   \
      }                                                                        \
      return;                                                                  \
    }                                                                          \
  }

// This event management code is only used with CUDA
#if HAVE_CUDA
cudaError_t GetCudaEvent(cudaEvent_t* event) {
  int device;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }

  auto& mutex = horovod_global.cuda_events_mutex;
  {
    std::lock_guard<std::mutex> guard(mutex);
    auto& queue = horovod_global.cuda_events[device];
    if (!queue.empty()) {
      *event = queue.front();
      queue.pop();
      return cudaSuccess;
    }
  }

  return cudaEventCreateWithFlags(event, cudaEventBlockingSync |
                                             cudaEventDisableTiming);
}

cudaError_t ReleaseCudaEvent(cudaEvent_t event) {
  int device;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }

  auto& mutex = horovod_global.cuda_events_mutex;
  {
    std::lock_guard<std::mutex> guard(mutex);
    auto& queue = horovod_global.cuda_events[device];
    queue.push(event);
  }

  return cudaSuccess;
}

#define RECORD_EVENT(entries, event_queue, name, stream)                       \
  {                                                                            \
    cudaEvent_t event;                                                         \
    CUDA_CHECK(entries, "GetCudaEvent", GetCudaEvent(&event))                  \
    CUDA_CHECK(entries, "cudaEventRecord", cudaEventRecord(event, stream))     \
    (event_queue).emplace(name, event);                                        \
  }

#define WAIT_FOR_EVENTS(entries, timeline, event_queue)                        \
  {                                                                            \
    while (!(event_queue).empty()) {                                           \
      std::string name;                                                        \
      cudaEvent_t event;                                                       \
      std::tie(name, event) = (event_queue).front();                           \
      (event_queue).pop();                                                     \
      if (name != "") {                                                        \
        ACTIVITY_START_ALL(entries, timeline, name)                            \
      }                                                                        \
      CUDA_CHECK(entries, "cudaEventSynchronize", cudaEventSynchronize(event)) \
      if (name != "") {                                                        \
        ACTIVITY_END_ALL(entries, timeline)                                    \
      }                                                                        \
      CUDA_CHECK(entries, "ReleaseCudaEvent", ReleaseCudaEvent(event))         \
    }                                                                          \
  }
#endif

#define ACTIVITY_START_ALL(entries, timeline, activity)                        \
  {                                                                            \
    for (auto& e : (entries)) {                                                \
      (timeline).ActivityStart(e.tensor_name, activity);                       \
    }                                                                          \
  }

#define ACTIVITY_END_ALL(entries, timeline)                                    \
  {                                                                            \
    for (auto& e : (entries)) {                                                \
      (timeline).ActivityEnd(e.tensor_name);                                   \
    }                                                                          \
  }

int64_t TensorFusionThresholdBytes() {
  int64_t proposed_fusion_threshold =
      horovod_global.param_manager.TensorFusionThresholdBytes();

  // If the cluster is homogeneous and hierarchical allreduce is enabled,
  // adjust buffer size to make sure it is divisible by local_size to improve
  // performance.
  if (horovod_global.is_homogeneous &&
      horovod_global.param_manager.HierarchicalAllreduce()) {
    // Assume the worst-case data type float64, since if it is divisible with
    // float64, it will be divisible for other types too.

    // Ensuring that fusion buffer can hold a number of elements divisible by
    // FUSION_BUFFER_ATOMIC_UNIT for performance
    int mpi_double_size;
    MPI_Type_size(MPI_DOUBLE, &mpi_double_size);
    int64_t div =
        horovod_global.local_size * mpi_double_size * FUSION_BUFFER_ATOMIC_UNIT;
    return ((proposed_fusion_threshold + div - 1) / div) * div;
  }

  return proposed_fusion_threshold;
}

// Process a Response by doing a reduction, a gather, a broadcast, or
// raising an error.
void PerformOperation(TensorTable& tensor_table, Response response) {
  std::vector<TensorTableEntry> entries;
  // Reserve to save re-allocation costs, as we know the size before.
  entries.reserve(response.tensor_names().size());
  {
    // Lock on the tensor table.
    std::lock_guard<std::mutex> guard(horovod_global.mutex);
    for (auto& name : response.tensor_names()) {
      // We should never fail at finding this key in the tensor table.
      auto iter = tensor_table.find(name);
      assert(iter != tensor_table.end());

      assert(response.response_type() == Response::ALLREDUCE ||
             response.response_type() == Response::ALLGATHER ||
             response.response_type() == Response::BROADCAST ||
             response.response_type() == Response::ERROR);

      entries.push_back(iter->second);

      // Clear the tensor table of this tensor and its callbacks; the rest of
      // this function takes care of it.
      tensor_table.erase(iter);
    }
  }

  auto& timeline = horovod_global.timeline;
  for (auto& e : entries) {
    timeline.Start(e.tensor_name, response.response_type());
  }

  if (entries.size() > 1) {
    auto first_entry = entries[0];
    // Note: it is OK for different entries to come from different frameworks
    // since buffer allocated here is guaranteed to survive at least till the
    // end of this operation.
    Status status = horovod_global.fusion_buffer.InitializeBuffer(
        TensorFusionThresholdBytes(), first_entry.device, first_entry.context,
        [&]() { ACTIVITY_START_ALL(entries, timeline, INIT_FUSION_BUFFER) },
        [&]() { ACTIVITY_END_ALL(entries, timeline) });
    if (!status.ok()) {
      for (auto& e : entries) {
        timeline.End(e.tensor_name, nullptr);
        e.callback(status);
      }
      return;
    }
  }

  // On GPU data readiness is signalled by ready_event.
  std::vector<TensorTableEntry> waiting_tensors;
  for (auto& e : entries) {
    if (e.ready_event != nullptr) {
      timeline.ActivityStart(e.tensor_name, WAIT_FOR_DATA);
      waiting_tensors.push_back(e);
    }
  }
  while (!waiting_tensors.empty()) {
    for (auto it = waiting_tensors.begin(); it != waiting_tensors.end();) {
      if (it->ready_event->Ready()) {
        timeline.ActivityEnd(it->tensor_name);
        timeline.ActivityStart(it->tensor_name, WAIT_FOR_OTHER_TENSOR_DATA);
        it = waiting_tensors.erase(it);
      } else {
        ++it;
      }
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(100));
  }
  for (auto& e : entries) {
    if (e.ready_event != nullptr) {
      timeline.ActivityEnd(e.tensor_name);
    }
  }

  Status status;
  if (response.response_type() == Response::ALLGATHER) {

    // Sizes of subcomponents of each entry from all ranks
    auto** entry_component_sizes = new int64_t*[entries.size()];

    // Offset of each subcomponent of every entry in the final buffer after
    // allgatherv
    auto** entry_component_offsets = new int64_t*[entries.size()];

    auto* recvcounts = new int[horovod_global.size]();
    auto* displcmnts = new int[horovod_global.size]();

    for (size_t ec = 0; ec < entries.size(); ++ec) {
      entry_component_sizes[ec] = new int64_t[horovod_global.size]();
      entry_component_offsets[ec] = new int64_t[horovod_global.size]();
    }

    auto& first_entry = entries[0];

    ACTIVITY_START_ALL(entries, timeline, ALLOCATE_OUTPUT)
    for (size_t ec = 0; ec < entries.size(); ++ec) {
      auto& e = entries[ec];
      // Every tensor participating in Allgather operation may have different
      // first dimension size, but the rest of dimensions are same for all
      // tensors.  Here we get shape of tensor sliced by first dimension.
      TensorShape single_slice_shape;
      for (int i = 1; i < e.tensor->shape().dims(); ++i) {
        single_slice_shape.AddDim(e.tensor->shape().dim_size(i));
      }

      // Copy tensor sizes from the Response into a vector of int64_t
      // and compute total size.  This is size of first dimension.
      int64_t total_entry_dimension_size = 0;
      for (int rc = 0; rc < horovod_global.size; ++rc) {
        auto component_size =
            response.tensor_sizes()[ec * horovod_global.size + rc];
        total_entry_dimension_size += component_size;
        recvcounts[rc] += component_size * single_slice_shape.num_elements();
        entry_component_sizes[ec][rc] =
            component_size * single_slice_shape.num_elements();
      }

      // Allgather output will have shape of:
      // (sum of first dimension of every tensor) x (tensor slice shape).
      TensorShape output_shape;
      output_shape.AddDim((int64_t)total_entry_dimension_size);
      output_shape.AppendShape(single_slice_shape);

      status = e.context->AllocateOutput(output_shape, &e.output);
      if (!status.ok()) {
        timeline.End(e.tensor_name, nullptr);
        e.callback(status);
        return;
      }
    }
    ACTIVITY_END_ALL(entries, timeline)

    for (int rc = 0; rc < horovod_global.size; ++rc) {
      if (rc == 0) {
        displcmnts[rc] = 0;
      } else {
        displcmnts[rc] = displcmnts[rc - 1] + recvcounts[rc - 1];
      }
    }

    unsigned int rank_displacement = 0;
    for (int rc = 0; rc < horovod_global.size; ++rc) {
      for (size_t ec = 0; ec < entries.size(); ++ec) {
        if (ec == 0) {
          entry_component_offsets[ec][rc] = rank_displacement;
        } else {
          entry_component_offsets[ec][rc] =
              entry_component_offsets[ec - 1][rc] +
              entry_component_sizes[ec - 1][rc];
        }
      }
      rank_displacement += recvcounts[rc];
    }

    int element_size;
    MPI_Type_size(GetMPIDataType(first_entry.tensor), &element_size);
    int64_t total_size = displcmnts[horovod_global.size - 1] +
                         recvcounts[horovod_global.size - 1];

    int64_t total_size_in_bytes = total_size * element_size;

#if HOROVOD_GPU_ALLGATHER != 'M' // 'M' stands for MPI
    if (horovod_global.param_manager.HierarchicalAllgather()) {
      // If shared buffer is not initialized or is not large enough, reallocate
      if (horovod_global.shared_buffer == nullptr ||
          horovod_global.shared_buffer_size < total_size_in_bytes) {
        if (horovod_global.shared_buffer != nullptr) {
          MPI_Win_fence(0, horovod_global.window);
          MPI_Win_free(&horovod_global.window);
          horovod_global.shared_buffer = nullptr;
        }
        int64_t window_size =
            horovod_global.local_rank == 0 ? total_size_in_bytes : 0;

        // Allocate shared memory, give each rank their respective pointer
        ACTIVITY_START_ALL(entries, timeline, ALLOCATE_SHARED_BUFFER)
        MPI_Win_allocate_shared(
            window_size, element_size, MPI_INFO_NULL, horovod_global.local_comm,
            &horovod_global.shared_buffer, &horovod_global.window);

        if (horovod_global.local_rank != 0) {
          int disp_unit;
          MPI_Aint winsize;
          MPI_Win_shared_query(horovod_global.window, 0, &winsize, &disp_unit,
                               &horovod_global.shared_buffer);
        }
        horovod_global.shared_buffer_size = total_size_in_bytes;
        ACTIVITY_END_ALL(entries, timeline)
      }

      // Compute cross-node allgather displacements and recvcounts for
      // homogeneous/parallelized case
      auto* cross_recvcounts = new int[horovod_global.cross_size]();
      auto* cross_displcmnts = new int[horovod_global.cross_size]();

      if (horovod_global.is_homogeneous) {
        for (int i = 0; i < horovod_global.cross_size; ++i) {
          cross_recvcounts[i] = recvcounts[horovod_global.local_size * i +
                                           horovod_global.local_rank];
          cross_displcmnts[i] = displcmnts[horovod_global.local_size * i +
                                           horovod_global.local_rank];
        }
      } else if (horovod_global.local_rank == 0) {
        // In this case local rank 0 will allgather with all local data
        int offset = 0;
        for (int i = 0; i < horovod_global.cross_size; ++i) {
          for (int j = offset; j < offset + horovod_global.local_sizes[i];
               ++j) {
            cross_recvcounts[i] += recvcounts[j];
          }
          cross_displcmnts[i] = displcmnts[offset];
          offset += horovod_global.local_sizes[i];
        }
      }

      ACTIVITY_START_ALL(entries, timeline, MEMCPY_IN_SHARED_BUFFER)
      for (size_t ec = 0; ec < entries.size(); ++ec) {
        auto& e = entries[ec];
        void* shared_buffer_at_offset =
            (uint8_t*)horovod_global.shared_buffer +
            entry_component_offsets[ec][horovod_global.rank] * element_size;

        // CPU copy to shared buffer
        memcpy(shared_buffer_at_offset, e.tensor->data(),
               (size_t)(entry_component_sizes[ec][horovod_global.rank] *
                        element_size));
      }
      MPI_CHECK(entries, "MPI_Barrier", MPI_Barrier(horovod_global.mpi_comm));
      ACTIVITY_END_ALL(entries, timeline)

      // Perform the cross-node allgather. If the cluster is homogeneous all
      // local ranks participate, otherwise local rank 0 handles all data
      ACTIVITY_START_ALL(entries, timeline, MPI_CROSS_ALLGATHER)
      if (horovod_global.is_homogeneous || horovod_global.local_rank == 0) {
        MPI_CHECK(entries, "MPI_Allgatherv",
                  MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                                 horovod_global.shared_buffer, cross_recvcounts,
                                 cross_displcmnts,
                                 GetMPIDataType(first_entry.tensor),
                                 horovod_global.cross_comm))
      }
      MPI_CHECK(entries, "MPI_Barrier", MPI_Barrier(horovod_global.mpi_comm));
      ACTIVITY_END_ALL(entries, timeline)

      // Copy memory out of the fusion buffer.
      ACTIVITY_START_ALL(entries, timeline, MEMCPY_OUT_FUSION_BUFFER)
      for (size_t ec = 0; ec < entries.size(); ++ec) {
        auto& e = entries[ec];
        int64_t copy_offset = 0;
        for (int rc = 0; rc < horovod_global.size; ++rc) {
          auto entry_component_size = entry_component_sizes[ec][rc];
          std::memcpy((void*)((uint8_t*)e.output->data() + copy_offset),
                      (void*)((uint8_t*)horovod_global.shared_buffer +
                              entry_component_size * element_size),
                      (size_t)entry_component_size * element_size);
          copy_offset += entry_component_size * element_size;
        }
      }
      MPI_CHECK(entries, "MPI_Barrier", MPI_Barrier(horovod_global.mpi_comm));
      ACTIVITY_END_ALL(entries, timeline)

      // Free the buffers
      delete[] cross_displcmnts;
      delete[] cross_recvcounts;

    } else {
#endif
      // Data is at the CPU and hierarchical allgather is disabled, or
      // Data is at the GPU and HOROVOD_GPU_ALLGATHER == MPI
      if (entries.size() > 1) {
        auto& buffer = horovod_global.fusion_buffer.GetBuffer(
            first_entry.device, first_entry.context->framework());
        auto buffer_data = buffer->AccessData(first_entry.context);

        int64_t total_num_elements = 0;

        // Copy memory into the fusion buffer. Then the input data of each
        // process is assumed to be in the area where that process would
        // receive its own contribution to the receive buffer.
        ACTIVITY_START_ALL(entries, timeline, MEMCPY_IN_FUSION_BUFFER)
        int64_t offset = displcmnts[horovod_global.rank] * element_size;
        for (auto& e : entries) {
          void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
          std::memcpy(buffer_data_at_offset, e.tensor->data(),
                      (size_t)e.tensor->size());
          offset += e.tensor->size();
          total_num_elements += e.tensor->shape().num_elements();
        }
        ACTIVITY_END_ALL(entries, timeline)

        ACTIVITY_START_ALL(entries, timeline, MPI_ALLGATHER)
        MPI_CHECK(entries, "MPI_Allgatherv",
                  MPI_Allgatherv(MPI_IN_PLACE, (int)total_num_elements,
                                 GetMPIDataType(first_entry.tensor),
                                 (void*)buffer_data, recvcounts, displcmnts,
                                 GetMPIDataType(first_entry.tensor),
                                 horovod_global.mpi_comm))
        ACTIVITY_END_ALL(entries, timeline)

        ACTIVITY_START_ALL(entries, timeline, MEMCPY_OUT_FUSION_BUFFER)
        // Copy memory out of the fusion buffer.
        for (size_t ec = 0; ec < entries.size(); ++ec) {
          auto& e = entries[ec];
          int64_t copy_offset = 0;
          for (int rc = 0; rc < horovod_global.size; ++rc) {
            std::memcpy((void*)((uint8_t*)e.output->data() + copy_offset),
                        (void*)((uint8_t*)buffer_data +
                                entry_component_offsets[ec][rc] * element_size),
                        (size_t)entry_component_sizes[ec][rc] * element_size);

            copy_offset += entry_component_sizes[ec][rc] * element_size;
          }
        }
        ACTIVITY_END_ALL(entries, timeline)

      } else if (entries.size() == 1) {
        ACTIVITY_START_ALL(entries, timeline, MPI_ALLGATHER)
        MPI_CHECK(
            entries, "MPI_Allgatherv",
            MPI_Allgatherv(first_entry.tensor->data(),
                           (int)first_entry.tensor->shape().num_elements(),
                           GetMPIDataType(first_entry.tensor),
                           (void*)first_entry.output->data(), recvcounts,
                           displcmnts, GetMPIDataType(first_entry.tensor),
                           horovod_global.mpi_comm))
        ACTIVITY_END_ALL(entries, timeline)
      }

      delete[] recvcounts;
      delete[] displcmnts;

      for (size_t ec = 0; ec < entries.size(); ++ec) {
        delete[] entry_component_sizes[ec];
        delete[] entry_component_offsets[ec];
      }
      delete[] entry_component_sizes;
      delete[] entry_component_offsets;

#if HOROVOD_GPU_ALLGATHER != 'M' // 'M' stands for MPI
    }
#endif

    for (auto& e : entries) {
      timeline.End(e.tensor_name, e.output);
      e.callback(Status::OK());
    }

  } else if (response.response_type() == Response::ALLREDUCE) {
    auto& first_entry = entries[0];
#if HAVE_CUDA
    bool on_gpu = first_entry.device != CPU_DEVICE_ID;
    if (on_gpu) {
      CUDA_CHECK(entries, "cudaSetDevice", cudaSetDevice(first_entry.device))

      // Ensure stream is in the map before executing reduction.
      cudaStream_t& stream = horovod_global.streams[first_entry.device];
      if (stream == nullptr) {
        int greatest_priority;
        CUDA_CHECK(entries, "cudaDeviceGetStreamPriorityRange",
                   cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority))
        CUDA_CHECK(entries, "cudaStreamCreateWithPriority",
                   cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking,
                                                greatest_priority))
      }
    }
#endif

// 'N' stands for NCCL and 'D' for DDL
#if HOROVOD_GPU_ALLREDUCE == 'N' || HOROVOD_GPU_ALLREDUCE == 'D'
    if (on_gpu) {
      auto stream = horovod_global.streams[first_entry.device];
      auto event_queue = std::queue<std::pair<std::string, cudaEvent_t>>();

      // Determine GPU IDs of the devices participating in this communicator.
      std::vector<int32_t> nccl_device_map;
      if (horovod_global.param_manager.HierarchicalAllreduce()) {
        // Reserve before for-loop, to save on reallocation cost.
        nccl_device_map.reserve(horovod_global.local_comm_ranks.size());
        for (int rank : horovod_global.local_comm_ranks) {
          nccl_device_map.push_back(response.devices()[rank]);
        }
      } else {
        nccl_device_map = response.devices();
      }

#if HOROVOD_GPU_ALLREDUCE == 'N'
      // Ensure NCCL communicator is in the map before executing reduction.
      ncclComm_t& nccl_comm = horovod_global.nccl_comms[nccl_device_map];
      if (nccl_comm == nullptr) {
        ACTIVITY_START_ALL(entries, timeline, INIT_NCCL)

        int nccl_rank, nccl_size;
        MPI_Comm nccl_id_bcast_comm;
        if (horovod_global.param_manager.HierarchicalAllreduce()) {
          nccl_rank = horovod_global.local_rank;
          nccl_size = horovod_global.local_size;
          nccl_id_bcast_comm = horovod_global.local_comm;
        } else {
          nccl_rank = horovod_global.rank;
          nccl_size = horovod_global.size;
          nccl_id_bcast_comm = horovod_global.mpi_comm;
        }

        ncclUniqueId nccl_id;
        if (nccl_rank == 0) {
          NCCL_CHECK(entries, "ncclGetUniqueId", ncclGetUniqueId(&nccl_id))
        }

        MPI_CHECK(entries, "MPI_Bcast",
                  MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0,
                            nccl_id_bcast_comm));

        ncclComm_t new_nccl_comm;
        NCCL_CHECK(
            entries, "ncclCommInitRank",
            ncclCommInitRank(&new_nccl_comm, nccl_size, nccl_id, nccl_rank))
        nccl_comm = new_nccl_comm;

        // Barrier helps NCCL to synchronize after initialization and avoid
        // deadlock that we've been seeing without it.
        MPI_CHECK(entries, "MPI_Barrier", MPI_Barrier(horovod_global.mpi_comm));

        ACTIVITY_END_ALL(entries, timeline)
      }
#elif HOROVOD_GPU_ALLREDUCE == 'D'
      if (!horovod_global.ddl_initialized) {
        // Initialize DDL
        auto ddl_options = std::getenv("DDL_OPTIONS");
        if (ddl_options == nullptr) {
          OP_ERROR(entries,
                   "DDL_OPTIONS env variable needs to be set to use DDL.")
        }
        DDL_CHECK(entries, "ddl_init", ddl_init(ddl_options))
        horovod_global.ddl_initialized = true;
        horovod_global.ddl_local_device_id = first_entry.device;
      } else if (horovod_global.ddl_local_device_id != first_entry.device) {
        OP_ERROR(entries,
                 "DDL does not support more than one GPU device per process.")
      }
#endif

      if (timeline.Initialized()) {
        RECORD_EVENT(entries, event_queue, QUEUE, stream)
      }

      // If entries.size() > 1, we copy tensors into fusion buffer before
      // allreduce, and distribute results of allreduce back into target
      // tensors after allreduce.

      const void* fused_input_data;
      void* buffer_data;
      int64_t num_elements = 0;
      size_t buffer_len;
      if (entries.size() > 1) {
        // Access the fusion buffer.
        auto& buffer = horovod_global.fusion_buffer.GetBuffer(
            first_entry.device, first_entry.context->framework());
        buffer_data =
            const_cast<void*>(buffer->AccessData(first_entry.context));

        // Copy memory into the fusion buffer.
        int64_t offset = 0;
        for (auto& e : entries) {
          void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
          CUDA_CHECK(entries, "cudaMemcpyAsync",
                     cudaMemcpyAsync(buffer_data_at_offset, e.tensor->data(),
                                     (size_t)e.tensor->size(),
                                     cudaMemcpyDeviceToDevice, stream))
          offset += e.tensor->size();
        }

        buffer_len = (size_t)offset;

        if (timeline.Initialized() || horovod_global.ddl_initialized) {
          RECORD_EVENT(entries, event_queue, MEMCPY_IN_FUSION_BUFFER, stream)
        }

        // Set the input data to originate from the buffer.
        fused_input_data = buffer_data;

        // Perform the reduction on the fusion buffer.
        for (auto& e : entries) {
          num_elements += e.tensor->shape().num_elements();
        }

      } else {
        fused_input_data = first_entry.tensor->data();
        buffer_data = (void*)first_entry.output->data();
        num_elements = first_entry.tensor->shape().num_elements();
        buffer_len = (size_t)first_entry.output->size();

        if (horovod_global.ddl_initialized) {
          // Copy input buffer content to output buffer
          // because DDL only supports in-place allreduce
          CUDA_CHECK(entries, "cudaMemcpyAsync",
                     cudaMemcpyAsync(buffer_data, fused_input_data, buffer_len,
                                     cudaMemcpyDeviceToDevice, stream))
          RECORD_EVENT(entries, event_queue, MEMCPY_IN_FUSION_BUFFER, stream)
        }
      }

      void* host_buffer = nullptr;

#if HOROVOD_GPU_ALLREDUCE == 'D'
      // Synchronize.
      WAIT_FOR_EVENTS(entries, timeline, event_queue)
      DDL_Type ddl_data_type;
      try {
        ddl_data_type = GetDDLDataType(first_entry.tensor);
      } catch (const std::logic_error& ex) {
        OP_ERROR(entries, ex.what())
      }
      DDL_CHECK(entries, "ddl_allreduce",
                ddl_allreduce(buffer_data, (size_t)num_elements, ddl_data_type,
                              DDL_OP_SUM))
#else
      if (horovod_global.param_manager.HierarchicalAllreduce()) {
        int element_size;
        MPI_Type_size(GetMPIDataType(first_entry.tensor), &element_size);

        // If cluster is homogeneous and we are using fusion buffer, include
        // dummy elements from the buffer (if necessary) to make sure the data
        // is divisible by local_size. This is always possible since we
        // set the fusion buffer size divisible by local_size.
        if (horovod_global.is_homogeneous && entries.size() > 1) {
          // Making sure the number of elements is divisible by
          // FUSION_BUFFER_ATOMIC_UNIT for improved performance
          int div = horovod_global.local_size * FUSION_BUFFER_ATOMIC_UNIT;
          num_elements = ((num_elements + div - 1) / div) * div;
          buffer_len = num_elements * element_size;
        }

        // Split the elements into two groups: num_elements_per_rank*local_size,
        // and num_elements_remaining. Cross-node reduction for the first group
        // is done by all local_rank's in parallel, while for the second group
        // it it is only done by the root_rank. If the cluster is not
        // homogeneous first group is zero, and root_rank is 0.

        // Homogeneous case:
        // For the part of data divisible by local_size, perform NCCL
        // ReduceScatter - Parallelized MPI Allreduce - NCCL Allgather. For the
        // non-divisible part (if any), do NCCL Reduce (at rank local_size-1),
        // MPI Allreduce (across rank (local_size-1)'s), and NCCL Bcast

        int64_t num_elements_per_rank =
            horovod_global.is_homogeneous
                ? num_elements / horovod_global.local_size
                : 0;

        size_t buffer_len_per_rank = element_size * num_elements_per_rank;

        void* buffer_data_at_rank_offset =
            (uint8_t*)buffer_data +
            buffer_len_per_rank * horovod_global.local_rank;

        int64_t num_elements_remaining =
            horovod_global.is_homogeneous
                ? num_elements % horovod_global.local_size
                : num_elements;

        size_t buffer_len_remaining = element_size * num_elements_remaining;

        void* buffer_data_remainder =
            (uint8_t*)buffer_data +
            buffer_len_per_rank * horovod_global.local_size;

        void* fused_input_data_remainder =
            (uint8_t*)fused_input_data +
            buffer_len_per_rank * horovod_global.local_size;

        int root_rank =
            horovod_global.is_homogeneous ? horovod_global.local_size - 1 : 0;
        bool is_root_rank = horovod_global.local_rank == root_rank;

        int64_t total_num_elements =
            is_root_rank ? num_elements_per_rank + num_elements_remaining
                         : num_elements_per_rank;
        int64_t total_buffer_len =
            is_root_rank ? buffer_len_per_rank + buffer_len_remaining
                         : buffer_len_per_rank;

        if (num_elements_per_rank > 0) {
          NCCL_CHECK(entries, "ncclReduceScatter",
                     ncclReduceScatter(fused_input_data,
                                       buffer_data_at_rank_offset,
                                       (size_t)num_elements_per_rank,
                                       GetNCCLDataType(first_entry.tensor),
                                       ncclSum, nccl_comm, stream))

          if (timeline.Initialized()) {
            RECORD_EVENT(entries, event_queue, NCCL_REDUCESCATTER, stream)
          }
        }

        if (num_elements_remaining > 0) {
          // Reduce the remaining data at local_size-1 to append to
          // existing buffer
          NCCL_CHECK(entries, "ncclReduce",
                     ncclReduce(fused_input_data_remainder,
                                buffer_data_remainder,
                                (size_t)num_elements_remaining,
                                GetNCCLDataType(first_entry.tensor), ncclSum,
                                root_rank, nccl_comm, stream))

          if (timeline.Initialized()) {
            RECORD_EVENT(entries, event_queue, NCCL_REDUCE, stream)
          }
        }

        if (horovod_global.is_homogeneous || is_root_rank) {
          // cudaHostAlloc is significantly slower than malloc.  Pre-allocating
          // a buffer is not safe since the tensor can be arbitrarily large.
          host_buffer = malloc(total_buffer_len);

          // Synchronize.
          WAIT_FOR_EVENTS(entries, timeline, event_queue)

          // According to https://docs.nvidia.com/cuda/cuda-runtime-api/
          // api-sync-behavior.html#api-sync-behavior__memcpy-async,
          // cudaMemcpyAsync is synchronous with respect to the host, so we
          // memcpy (effectively) synchronously to generate an accurate timeline
          ACTIVITY_START_ALL(entries, timeline, MEMCPY_IN_HOST_BUFFER)
          CUDA_CHECK(entries, "cudaMemcpyAsync",
                     cudaMemcpyAsync(host_buffer, buffer_data_at_rank_offset,
                                     total_buffer_len, cudaMemcpyDeviceToHost,
                                     stream))
          ACTIVITY_END_ALL(entries, timeline)

          ACTIVITY_START_ALL(entries, timeline, MPI_ALLREDUCE)
          MPI_CHECK(entries, "MPI_Allreduce",
                    MPI_Allreduce(MPI_IN_PLACE, host_buffer,
                                  (int)total_num_elements,
                                  GetMPIDataType(first_entry.tensor),
                                  first_entry.tensor->dtype() == HOROVOD_FLOAT16
                                      ? horovod_global.mpi_float16_sum
                                      : MPI_SUM,
                                  horovod_global.cross_comm))
          ACTIVITY_END_ALL(entries, timeline)

          ACTIVITY_START_ALL(entries, timeline, MEMCPY_OUT_HOST_BUFFER)
          CUDA_CHECK(entries, "cudaMemcpyAsync",
                     cudaMemcpyAsync(buffer_data_at_rank_offset, host_buffer,
                                     total_buffer_len, cudaMemcpyHostToDevice,
                                     stream))
          ACTIVITY_END_ALL(entries, timeline)
        }

        if (num_elements_per_rank > 0) {
          NCCL_CHECK(entries, "ncclAllGather",
                     ncclAllGather(buffer_data_at_rank_offset, buffer_data,
                                   (size_t)num_elements_per_rank,
                                   GetNCCLDataType(first_entry.tensor),
                                   nccl_comm, stream))

          if (timeline.Initialized()) {
            RECORD_EVENT(entries, event_queue, NCCL_ALLGATHER, stream)
          }
        }
        if (num_elements_remaining > 0) {
          NCCL_CHECK(entries, "ncclBcast",
                     ncclBcast(buffer_data_remainder,
                               (size_t)num_elements_remaining,
                               GetNCCLDataType(first_entry.tensor), root_rank,
                               nccl_comm, stream))

          if (timeline.Initialized()) {
            RECORD_EVENT(entries, event_queue, NCCL_BCAST, stream)
          }
        }
      } else {
        NCCL_CHECK(entries, "ncclAllReduce",
                   ncclAllReduce(fused_input_data, buffer_data,
                                 (size_t)num_elements,
                                 GetNCCLDataType(first_entry.tensor), ncclSum,
                                 nccl_comm, stream))
        if (timeline.Initialized()) {
          RECORD_EVENT(entries, event_queue, NCCL_ALLREDUCE, stream)
        }
      }
#endif

      if (entries.size() > 1) {
        // Copy memory out of the fusion buffer.
        int64_t offset = 0;
        for (auto& e : entries) {
          void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
          CUDA_CHECK(entries, "cudaMemcpyAsync",
                     cudaMemcpyAsync((void*)e.output->data(),
                                     buffer_data_at_offset,
                                     (size_t)e.tensor->size(),
                                     cudaMemcpyDeviceToDevice, stream))
          offset += e.tensor->size();
        }
        if (timeline.Initialized()) {
          RECORD_EVENT(entries, event_queue, MEMCPY_OUT_FUSION_BUFFER, stream)
        }
      }

      // Use completion marker via event because it's faster than
      // blocking cudaStreamSynchronize() in this thread.
      RECORD_EVENT(entries, event_queue, "", stream)

      // TODO: use thread pool or single thread for callbacks
      std::thread finalizer_thread([entries, first_entry, host_buffer, response,
                                    event_queue, &timeline]() mutable {
        CUDA_CHECK(entries, "cudaSetDevice", cudaSetDevice(first_entry.device))

        WAIT_FOR_EVENTS(entries, timeline, event_queue)

        if (host_buffer != nullptr) {
          free(host_buffer);
        }

        for (auto& e : entries) {
          timeline.End(e.tensor_name, e.output);
          e.callback(Status::OK());
        }
      });
      finalizer_thread.detach();
      return;
    }
#endif

    if (entries.size() > 1) {
      // Access the fusion buffer.
      auto& buffer = horovod_global.fusion_buffer.GetBuffer(
          first_entry.device, first_entry.context->framework());
      auto buffer_data = buffer->AccessData(first_entry.context);

      // Copy memory into the fusion buffer.
      ACTIVITY_START_ALL(entries, timeline, MEMCPY_IN_FUSION_BUFFER)
      int64_t offset = 0;
      for (auto& e : entries) {
        void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
#if HAVE_CUDA
        if (on_gpu) {
          CUDA_CHECK(entries, "cudaMemcpyAsync",
                     cudaMemcpyAsync(
                         buffer_data_at_offset, e.tensor->data(),
                         (size_t)e.tensor->size(), cudaMemcpyDeviceToDevice,
                         horovod_global.streams[first_entry.device]))
        } else {
#endif
          std::memcpy(buffer_data_at_offset, e.tensor->data(),
                      (size_t)e.tensor->size());
#if HAVE_CUDA
        }
#endif
        offset += e.tensor->size();
      }
#if HAVE_CUDA
      if (on_gpu) {
        CUDA_CHECK(
            entries, "cudaStreamSynchronize",
            cudaStreamSynchronize(horovod_global.streams[first_entry.device]))
      }
#endif
      ACTIVITY_END_ALL(entries, timeline)

      ACTIVITY_START_ALL(entries, timeline, MPI_ALLREDUCE)
      int64_t num_elements = 0;
      for (auto& e : entries) {
        num_elements += e.tensor->shape().num_elements();
      }
      MPI_CHECK(entries, "MPI_Allreduce",
                MPI_Allreduce(MPI_IN_PLACE, (void*)buffer_data,
                              (int)num_elements,
                              GetMPIDataType(first_entry.tensor),
                              first_entry.tensor->dtype() == HOROVOD_FLOAT16
                                  ? horovod_global.mpi_float16_sum
                                  : MPI_SUM,
                              horovod_global.mpi_comm))
      ACTIVITY_END_ALL(entries, timeline)

      // Copy memory out of the fusion buffer.
      ACTIVITY_START_ALL(entries, timeline, MEMCPY_OUT_FUSION_BUFFER)
      offset = 0;
      for (auto& e : entries) {
        void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
#if HAVE_CUDA
        if (on_gpu) {
          CUDA_CHECK(entries, "cudaMemcpyAsync",
                     cudaMemcpyAsync(
                         (void*)e.output->data(), buffer_data_at_offset,
                         (size_t)e.tensor->size(), cudaMemcpyDeviceToDevice,
                         horovod_global.streams[first_entry.device]))
        } else {
#endif
          std::memcpy((void*)e.output->data(), buffer_data_at_offset,
                      (size_t)e.tensor->size());
#if HAVE_CUDA
        }
#endif
        offset += e.tensor->size();
      }
#if HAVE_CUDA
      if (on_gpu) {
        CUDA_CHECK(
            entries, "cudaStreamSynchronize",
            cudaStreamSynchronize(horovod_global.streams[first_entry.device]))
      }
#endif
      ACTIVITY_END_ALL(entries, timeline)
    } else {
      auto& e = first_entry;
      ACTIVITY_START_ALL(entries, timeline, MPI_ALLREDUCE)
      const void* sendbuf = e.tensor->data() == e.output->data()
                                ? MPI_IN_PLACE
                                : e.tensor->data();
      MPI_CHECK(entries, "MPI_Allreduce",
                MPI_Allreduce(sendbuf, (void*)e.output->data(),
                              (int)e.tensor->shape().num_elements(),
                              GetMPIDataType(e.tensor),
                              first_entry.tensor->dtype() == HOROVOD_FLOAT16
                                  ? horovod_global.mpi_float16_sum
                                  : MPI_SUM,
                              horovod_global.mpi_comm))
      ACTIVITY_END_ALL(entries, timeline)
    }

    for (auto& e : entries) {
      timeline.End(e.tensor_name, e.output);
      e.callback(Status::OK());
    }
  } else if (response.response_type() == Response::BROADCAST) {
    assert(entries.size() == 1);
    auto e = entries[0];

    // On root rank, MPI_Bcast sends data, on other ranks it receives data.
    void* data_ptr;
    if (horovod_global.rank == e.root_rank) {
      data_ptr = (void*)e.tensor->data();
    } else {
      data_ptr = (void*)e.output->data();
    }

    ACTIVITY_START_ALL(entries, timeline, MPI_BCAST)
    MPI_CHECK(entries, "MPI_Bcast",
              MPI_Bcast(data_ptr, (int)e.tensor->shape().num_elements(),
                        GetMPIDataType(e.tensor), e.root_rank,
                        horovod_global.mpi_comm))
    ACTIVITY_END_ALL(entries, timeline)

    timeline.End(e.tensor_name, e.output);
    e.callback(Status::OK());
  } else if (response.response_type() == Response::ERROR) {
    assert(entries.size() == 1);
    auto e = entries[0];

    status = Status::PreconditionError(response.error_message());
    timeline.End(e.tensor_name, nullptr);
    e.callback(status);
  }
}

// Report Tensors that were submitted to be reduced, gathered or broadcasted by
// some ranks but not others and are waiting for long time to get processed.
void CheckForStalledTensors(HorovodGlobalState& state) {
  bool preamble = false;
  auto now = std::chrono::steady_clock::now();
  for (auto& m : *state.message_table) {
    auto tensor_name = m.first;
    std::vector<Request>& messages = std::get<0>(m.second);
    std::chrono::steady_clock::time_point start_at = std::get<1>(m.second);

    if (now - start_at > STALL_WARNING_TIME) {
      std::stringstream message;
      if (!preamble) {
       message << "One or more tensors were submitted to be "
                  "reduced, gathered or broadcasted by subset of ranks and "
                  "are waiting for remainder of ranks for more than "
               << std::chrono::duration_cast<std::chrono::seconds>(
                   STALL_WARNING_TIME)
                   .count()
               << " seconds. "
               << "This may indicate that different ranks are trying to "
                  "submit different tensors or that only subset of ranks is "
                  "submitting tensors, which will cause deadlock. "
               << std::endl << "Stalled ops:" ;
        preamble = true;
      }
      message << tensor_name;
      message << " [missing ranks:";
      std::unordered_set<int32_t> ready_ranks;
      bool missing_preamble = false;
      for (auto msg_iter = messages.begin(); msg_iter != messages.end();
           ++msg_iter) {
        ready_ranks.insert(msg_iter->request_rank());
      }
      for (int32_t rank = 0; rank < state.size; ++rank) {
        if (ready_ranks.find(rank) == ready_ranks.end()) {
          if (!missing_preamble) {
            message << " ";
            missing_preamble = true;
          } else {
            message << ", ";
          }
          message << rank;
        }
      }
      message << "]";
      LOG(WARNING) << message.str();
    }
  }
}

// The MPI background thread loop coordinates all the MPI processes and the
// tensor reductions. The design of the communicator mechanism is limited by a
// few considerations:
//
//      1. Some MPI implementations require all MPI calls to happen from a
//      single thread. Since TensorFlow may use several threads for graph
//      processing, this means we must have our own dedicated thread for dealing
//      with MPI.
//      2. We want to gracefully handle errors, when MPI processes do not
//      properly agree upon what should happen (such as mismatched types or
//      shapes). To do so requires the MPI processes to know about the shapes
//      and types of the relevant tensors on the other processes.
//      3. The MPI reductions and gathers should be able to happen in parallel
//      with other ongoing operations. This means that they cannot be blocking
//      ops, but rather must be async ops, the execution of which happens on a
//      separate thread.
//      4. We cannot guarantee that all the MPI processes reduce their tensors
//      in the same order, so we cannot dispatch one thread per tensor,
//      otherwise we may end up dispatching many blocked threads and never make
//      progress if we have a thread pool limit.
bool RunLoopOnce(HorovodGlobalState& state, bool is_coordinator);
void BackgroundThreadLoop(HorovodGlobalState& state) {
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
  int provided;
  int is_mpi_initialized = 0;
  MPI_Initialized(&is_mpi_initialized);
  if (is_mpi_initialized) {
    MPI_Query_thread(&provided);
    if (provided < MPI_THREAD_MULTIPLE) {
      LOG(WARNING) << "MPI has already been initialized without "
                      "multi-threading support (MPI_THREAD_MULTIPLE). This will "
                      "likely cause a segmentation fault.";
    }
  } else {
    MPI_Init_thread(NULL, NULL, required, &provided);
    state.should_finalize = true;
  }

  if (state.ranks.size() > 0) {
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group work_group;
    MPI_Group_incl(world_group, state.ranks.size(), &(state.ranks[0]),
                   &work_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, work_group, 0, &(state.mpi_comm));
    if (state.mpi_comm == MPI_COMM_NULL) {
      LOG(WARNING) << "Unable to create Horovod communicator, using "
                      "MPI_COMM_WORLD instead.";
      state.mpi_comm = MPI_COMM_WORLD;
    }
    MPI_Group_free(&world_group);
    MPI_Group_free(&work_group);
  } else if (!state.mpi_comm) {
    // No ranks were given and no communicator provided to horovod_init() so use
    // MPI_COMM_WORLD
    MPI_Comm_dup(MPI_COMM_WORLD, &(horovod_global.mpi_comm));
  }

  // Get MPI rank to determine if we are rank zero.
  int rank;
  MPI_Comm_rank(state.mpi_comm, &rank);
  bool is_coordinator = rank == 0;

  // Get MPI size to determine how many tensors to wait for before reducing.
  int size;
  MPI_Comm_size(state.mpi_comm, &size);
  if (is_coordinator) {
    LOG(INFO) << "Started Horovod with " << size << " processes";
  }

  // Determine local rank by querying the local communicator.
  MPI_Comm local_comm;
  MPI_Comm_split_type(state.mpi_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &local_comm);
  int local_rank, local_size;
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_size(local_comm, &local_size);
  std::vector<int> local_comm_ranks((size_t)local_size);
  local_comm_ranks[local_rank] = rank;
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, local_comm_ranks.data(), 1,
                MPI_INT, local_comm);

  // Determine if cluster is homogeneous, i.e., if every node has the same
  // local_size
  auto local_sizes = new int[size];
  MPI_Allgather(&local_size, 1, MPI_INT, local_sizes, 1, MPI_INT,
                state.mpi_comm);

  bool is_homogeneous = true;
  for (int i = 0; i < size; ++i) {
    if (local_sizes[i] != local_size) {
      is_homogeneous = false;
      break;
    }
  }
  for (int i = 0; i < size; i += local_sizes[i]) {
    state.local_sizes.push_back(local_sizes[i]);
  }

  delete[] local_sizes;
  state.is_homogeneous = is_homogeneous;

  // Set up cross-communicator in case of hierarchical allreduce.
  MPI_Comm cross_comm;
  MPI_Comm_split(state.mpi_comm, local_rank, rank, &cross_comm);
  int cross_rank, cross_size;
  MPI_Comm_rank(cross_comm, &cross_rank);
  MPI_Comm_size(cross_comm, &cross_size);

  // Create custom MPI float16 data type.
  MPI_Datatype mpi_float16_t;
  MPI_Type_contiguous(2, MPI_BYTE, &mpi_float16_t);
  MPI_Type_commit(&mpi_float16_t);

  // Create custom MPI float16 summation op.
  MPI_Op mpi_float16_sum;
  MPI_Op_create(&float16_sum, 1, &mpi_float16_sum);

  // Create custom datatypes for the parameter manager.
  state.param_manager.CreateMpiTypes();

  state.rank = rank;
  state.local_rank = local_rank;
  state.cross_rank = cross_rank;
  state.size = size;
  state.local_size = local_size;
  state.cross_size = cross_size;
  state.local_comm = local_comm;
  state.cross_comm = cross_comm;
  state.mpi_float16_t = mpi_float16_t;
  state.mpi_float16_sum = mpi_float16_sum;
  state.mpi_threads_supported = (provided == MPI_THREAD_MULTIPLE);
  state.local_comm_ranks = local_comm_ranks;

  // Open the timeline file on coordinator.
  auto horovod_timeline = std::getenv(HOROVOD_TIMELINE);
  if (is_coordinator && horovod_timeline != nullptr) {
    state.timeline.Initialize(std::string(horovod_timeline),
                              static_cast<unsigned int>(size));
  }

  auto horovod_timeline_mark_cycles = std::getenv(HOROVOD_TIMELINE_MARK_CYCLES);
  if (horovod_timeline_mark_cycles != nullptr &&
      std::strtol(horovod_timeline_mark_cycles, nullptr, 10) > 0) {
    state.mark_cycles_in_timeline = true;
  }

  // Override Tensor Fusion threshold, if it's set.
  state.param_manager.SetTensorFusionThresholdBytes(64 * 1024 * 1024);
  auto horovod_fusion_threshold = std::getenv(HOROVOD_FUSION_THRESHOLD);
  if (horovod_fusion_threshold != nullptr) {
    int64_t threshold = std::strtol(horovod_fusion_threshold, nullptr, 10);
    state.param_manager.SetTensorFusionThresholdBytes(threshold, true);
  }

  // Override the cycle time.
  state.param_manager.SetCycleTimeMs(5);
  auto horovod_cycle_time = std::getenv(HOROVOD_CYCLE_TIME);
  if (horovod_cycle_time != nullptr) {
    state.param_manager.SetCycleTimeMs(std::strtof(horovod_cycle_time, nullptr),
                                       true);
  }

  // Disable stall check.
  auto horovod_stall_check_disable = std::getenv(HOROVOD_STALL_CHECK_DISABLE);
  if (horovod_stall_check_disable != nullptr &&
      std::strtol(horovod_stall_check_disable, nullptr, 10) > 0) {
    state.perform_stall_check = false;
  }

  // Set flag for hierarchical allgather. Ignore if Horovod is running on a
  // single node.
  auto horovod_hierarchical_allgather =
      std::getenv(HOROVOD_HIERARCHICAL_ALLGATHER);
  state.param_manager.SetHierarchicalAllgather(false);
  if (horovod_hierarchical_allgather != nullptr) {
    bool value = std::strtol(horovod_hierarchical_allgather, nullptr, 10) > 0 &&
                 (size != local_size);
    state.param_manager.SetHierarchicalAllgather(value, true);
  }
  // Set flag for hierarchical allreduce. Ignore if Horovod is running on a
  // single node.
  auto horovod_hierarchical_allreduce =
      std::getenv(HOROVOD_HIERARCHICAL_ALLREDUCE);
  state.param_manager.SetHierarchicalAllreduce(false);
  if (horovod_hierarchical_allreduce != nullptr) {
    bool value = std::strtol(horovod_hierarchical_allreduce, nullptr, 10) > 0 &&
                 (size != local_size);
    state.param_manager.SetHierarchicalAllreduce(value, true);
  }

#if HOROVOD_GPU_ALLREDUCE != 'N' && HOROVOD_GPU_ALLREDUCE != 'D'
  // Hierarchical allreduce is not supported without NCCL or DDL
  state.param_manager.SetHierarchicalAllreduce(false, true);
#endif

  // Issue warning if hierarchical allreduce is enabled in heterogeneous cluster
  if (is_coordinator &&
      (state.param_manager.HierarchicalAllreduce() ||
       state.param_manager.HierarchicalAllgather()) &&
      !state.is_homogeneous) {
    std::cerr
        << "WARNING: Using different number of ranks per node might cause "
           "performance loss in hierarchical allgather and "
           "hierarchical allreduce. Consider assigning the same "
           "number of ranks to each node, or disabling hierarchical "
           "allgather and hierarchical allreduce.";
  }

  // Enable auto-tuning.
  auto horovod_autotune = std::getenv(HOROVOD_AUTOTUNE);
  if (horovod_autotune != nullptr &&
      std::strtol(horovod_autotune, nullptr, 10) > 0) {
    auto horovod_autotune_log = std::getenv(HOROVOD_AUTOTUNE_LOG);
    state.param_manager.Initialize(rank, RANK_ZERO, state.mpi_comm,
                                   horovod_autotune_log != nullptr
                                       ? std::string(horovod_autotune_log)
                                       : "");
    state.param_manager.SetAutoTuning(true);
  }

  // Initialize the tensor count table. No tensors are available yet.
  if (is_coordinator) {
    state.message_table = std::unique_ptr<MessageTable>(new MessageTable());
  }

  // Signal that initialization is completed.
  state.initialization_done = true;

  LOG(INFO, rank) << "Horovod Initialized";

  // Iterate until shutdown.
  while (RunLoopOnce(state, is_coordinator))
    ;

  LOG(DEBUG, rank) << "Shutting down background thread";

  // Signal that shutdown has been requested.
  state.shut_down = true;

  // TODO: init.cu:645 WARN Cuda failure 'driver shutting down'
  //#if HAVE_NCCL
  //  for (auto it = horovod_global.streams.begin();
  //       it != horovod_global.streams.end(); ++it) {
  //    cudaStreamSynchronize(it->second);
  //  }
  //  for (auto it = horovod_global.nccl_comms.begin();
  //       it != horovod_global.nccl_comms.end(); ++it) {
  //    ncclCommDestroy(it->second);
  //  }
  //#endif

  // Notify all outstanding operations that Horovod has been shut down
  // and clear up the tensor table and message queue.
  std::vector<StatusCallback> callbacks;
  {
    std::lock_guard<std::mutex> guard(state.mutex);
    for (auto& e : state.tensor_table) {
      callbacks.emplace_back(e.second.callback);
    }
    state.tensor_table.clear();
    while (!state.message_queue.empty()) {
      state.message_queue.pop();
    }
  }
  for (auto& cb : callbacks) {
    cb(SHUT_DOWN_ERROR);
  }

  if (horovod_global.shared_buffer != nullptr) {
    MPI_Win_free(&horovod_global.window);
    horovod_global.shared_buffer = nullptr;
  }

  if (horovod_global.mpi_comm != MPI_COMM_NULL &&
      horovod_global.mpi_comm != MPI_COMM_WORLD) {
    MPI_Comm_free(&horovod_global.mpi_comm);
  }

  if (horovod_global.local_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&horovod_global.local_comm);
  }

  if (horovod_global.cross_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&horovod_global.cross_comm);
  }

  if (horovod_global.mpi_float16_t != MPI_DATATYPE_NULL) {
    MPI_Type_free(&horovod_global.mpi_float16_t);
  }

  if (horovod_global.mpi_float16_sum != MPI_OP_NULL) {
    MPI_Op_free(&horovod_global.mpi_float16_sum);
  }

  horovod_global.param_manager.FreeMpiTypes();

  if (horovod_global.should_finalize) {
#if HAVE_DDL
    // ddl_finalize calls MPI_Finalize
    ddl_finalize();
#else
    int is_mpi_finalized = 0;
    MPI_Finalized(&is_mpi_finalized);
    if (!is_mpi_finalized) {
      MPI_Finalize();
    }
#endif
  }
}

// The coordinator currently follows a master-worker paradigm. Rank zero acts
// as the master (the "coordinator"), whereas all other ranks are simply
// workers. Each rank runs its own background thread which progresses in ticks.
// In each tick, the following actions happen:
//
//      a) The workers send a Request to the coordinator, indicating what
//      they would like to do (which tensor they would like to gather and
//      reduce, as well as their shape and type). They repeat this for every
//      tensor that they would like to operate on.
//
//      b) The workers send an empty "DONE" message to the coordinator to
//      indicate that there are no more tensors they wish to operate on.
//
//      c) The coordinator receives the Requests from the workers, as well
//      as from its own TensorFlow ops, and stores them in a request table. The
//      coordinator continues to receive Request messages until it has
//      received MPI_SIZE number of empty "DONE" messages.
//
//      d) The coordinator finds all tensors that are ready to be reduced,
//      gathered, or all operations that result in an error. For each of those,
//      it sends a Response to all the workers. When no more Responses
//      are available, it sends a "DONE" response to the workers. If the process
//      is being shutdown, it instead sends a "SHUTDOWN" response.
//
//      e) The workers listen for Response messages, processing each one by
//      doing the required reduce or gather, until they receive a "DONE"
//      response from the coordinator. At that point, the tick ends.
//      If instead of "DONE" they receive "SHUTDOWN", they exit their background
//      loop.
bool RunLoopOnce(HorovodGlobalState& state, bool is_coordinator) {
  // This delay determines thread frequency and MPI message latency
  auto start_time = std::chrono::steady_clock::now();
  auto sleep_duration = state.last_cycle_start +
                        std::chrono::microseconds(
                            long(state.param_manager.CycleTimeMs() * 1000.)) -
                        start_time;
  if (sleep_duration > std::chrono::steady_clock::duration::zero()) {
    std::this_thread::sleep_for(sleep_duration);
  }
  state.last_cycle_start = std::chrono::steady_clock::now();

  if (state.mark_cycles_in_timeline) {
    // Mark start of the new cycle.
    state.timeline.MarkCycleStart();
  }

  // Copy the data structures from global state under this lock.
  // However, don't keep the lock for the rest of the loop, so that
  // enqueued stream callbacks can continue.
  std::queue<Request> message_queue;
  {
    std::lock_guard<std::mutex> guard(state.mutex);
    while (!state.message_queue.empty()) {
      Request message = state.message_queue.front();
      state.message_queue.pop();
      message_queue.push(message);
    }
  }

  if (!message_queue.empty()) {
    LOG(DEBUG, state.rank) << "Sent " << message_queue.size() << " messages";
  }

  // Flag indicating that the background thread should shut down.
  bool should_shut_down = state.shut_down;

  // Collect all tensors that are ready to be reduced. Record them in the
  // tensor count table (rank zero) or send them to rank zero to be
  // recorded (everyone else).
  std::vector<std::string> ready_to_reduce;
  if (is_coordinator) {
    while (!message_queue.empty()) {
      // Pop the first available message message
      Request message = message_queue.front();
      message_queue.pop();

      bool reduce =
          IncrementTensorCount(state.message_table, message, state.size);
      if (reduce) {
        ready_to_reduce.push_back(message.tensor_name());
      }
    }

    // Rank zero has put all its own tensors in the tensor count table.
    // Now, it should count all the tensors that are coming from other
    // ranks at this tick.

    // 1. Get message lengths from every rank.
    auto recvcounts = new int[state.size];
    recvcounts[0] = 0;
    MPI_Gather(MPI_IN_PLACE, 1, MPI_INT, recvcounts, 1, MPI_INT, RANK_ZERO,
               state.mpi_comm);

    // 2. Compute displacements.
    auto displcmnts = new int[state.size];
    size_t total_size = 0;
    for (int i = 0; i < state.size; ++i) {
      if (i == 0) {
        displcmnts[i] = 0;
      } else {
        displcmnts[i] = recvcounts[i - 1] + displcmnts[i - 1];
      }
      total_size += recvcounts[i];
    }

    // 3. Collect messages from every rank.
    auto buffer = new uint8_t[total_size];
    MPI_Gatherv(nullptr, 0, MPI_BYTE, buffer, recvcounts, displcmnts, MPI_BYTE,
                RANK_ZERO, state.mpi_comm);

    // 4. Process messages.
    for (int i = 1; i < state.size; ++i) {
      auto rank_buffer_ptr = buffer + displcmnts[i];
      RequestList received_message_list;
      RequestList::ParseFromBytes(received_message_list, rank_buffer_ptr);
      for (auto& received_message : received_message_list.requests()) {
        auto& received_name = received_message.tensor_name();

        bool reduce = IncrementTensorCount(state.message_table,
                                           received_message, state.size);
        if (reduce) {
          ready_to_reduce.push_back(received_name);
        }
      }
      if (received_message_list.shutdown()) {
        // Received SHUTDOWN request from one of the workers.
        should_shut_down = true;
      }
    }

    // 5. Free buffers.
    delete[] recvcounts;
    delete[] displcmnts;
    delete[] buffer;

    // At this point, rank zero should have a fully updated tensor count
    // table and should know all the tensors that need to be reduced or
    // gathered, and everyone else should have sent all their information
    // to rank zero. We can now do reductions and gathers; rank zero will
    // choose which ones and in what order, and will notify the other ranks
    // before doing each reduction.
    std::deque<Response> responses;
    for (auto& tensor_name : ready_to_reduce) {
      Response response =
          ConstructResponse(state.message_table, tensor_name);
      responses.push_back(std::move(response));
    }

    ResponseList response_list;
    response_list.set_shutdown(should_shut_down);
    {
      // Protect access to tensor table.
      std::lock_guard<std::mutex> guard(horovod_global.mutex);
      while (!responses.empty()) {

        auto response = responses.front();
        assert(response.tensor_names().size() == 1);
        responses.pop_front();
        int64_t tensor_size = 0;
        if (response.response_type() == Response::ResponseType::ALLREDUCE) {
          // Attempt to add more responses to this fused response.
          auto& entry = state.tensor_table[response.tensor_names()[0]];
          tensor_size = entry.tensor->size();

          std::deque<Response> skipped_responses;
          int64_t skipped_size = 0;
          while (!responses.empty()) {
            auto new_response = responses.front();
            assert(new_response.tensor_names().size() == 1);
            auto& new_entry =
                state.tensor_table[new_response.tensor_names()[0]];
            int64_t new_tensor_size = new_entry.tensor->size();

            if (response.response_type() == new_response.response_type() &&
                response.devices() == new_response.devices() &&
                entry.tensor->dtype() == new_entry.tensor->dtype() &&
                tensor_size + new_tensor_size <= TensorFusionThresholdBytes()) {
              // These tensors will fuse together well.
              tensor_size += new_tensor_size;
              response.add_tensor_name(new_response.tensor_names()[0]);
              responses.pop_front();
            } else {
              // In general, don't try to fuse additional tensors since they are usually
              // computed in order of requests and skipping tensors may mean
              // that the batch will have to wait longer while skipped tensors
              // could be reduced at that time. However, mixed-precision training may yield
              // requests of various dtype in a mixed-up sequence causing breakups
              // in fusion. To counter this some look ahead is allowed.

              skipped_size += new_tensor_size;
              if (tensor_size + skipped_size <= TensorFusionThresholdBytes()) {
                // Skip response and look ahead for more to fuse.
                skipped_responses.push_back(std::move(responses.front()));
                responses.pop_front();
              } else {
                break;
              }
            }
          }

          // Replace any skipped responses.
          while (!skipped_responses.empty()) {
            responses.push_front(std::move(skipped_responses.back()));
            skipped_responses.pop_back();
          }

        } else if (response.response_type() ==
                   Response::ResponseType::ALLGATHER) {
          // Attempt to add more responses to this fused response.
          auto& entry = state.tensor_table[response.tensor_names()[0]];

          // This is size of first dimension.
          int64_t total_byte_size_of_output =
              TotalByteSizeOfAllgatherOutput(response.tensor_sizes(), entry);

          std::deque<Response> skipped_responses;
          int64_t skipped_size = 0;
          while (!responses.empty()) {

            auto new_response = responses.front();
            assert(new_response.tensor_names().size() == 1);
            auto& new_entry =
                state.tensor_table[new_response.tensor_names()[0]];

            int64_t new_total_byte_size_of_output =
                TotalByteSizeOfAllgatherOutput(new_response.tensor_sizes(),
                                               new_entry);

            if (response.response_type() == new_response.response_type() &&
                response.devices() == new_response.devices() &&
                entry.tensor->dtype() == new_entry.tensor->dtype() &&
                total_byte_size_of_output + new_total_byte_size_of_output <=
                    TensorFusionThresholdBytes()) {

              // These tensors will fuse together well.
              total_byte_size_of_output += new_total_byte_size_of_output;
              response.add_allgather_response(new_response);
              responses.pop_front();

            } else {
              // In general, don't try to fuse additional tensors since they are usually
              // computed in order of requests and skipping tensors may mean
              // that the batch will have to wait longer while skipped tensors
              // could be reduced at that time. However, mixed-precision training may yield
              // requests of various dtype in a mixed-up sequence causing breakups
              // in fusion. To counter this some look ahead is allowed.

              skipped_size += new_total_byte_size_of_output;
              if (total_byte_size_of_output + skipped_size <=
                      TensorFusionThresholdBytes()) {
                // Skip response and look ahead for more to fuse.
                skipped_responses.push_back(std::move(responses.front()));
                responses.pop_front();
              } else {
                break;
              }
            }
          }

          // Replace any skipped responses.
          while (!skipped_responses.empty()) {
            responses.push_front(std::move(skipped_responses.back()));
            skipped_responses.pop_back();
          }

        }

        response_list.add_response(response);
        LOG(DEBUG) << "Created response of size " << tensor_size;
      }
    }

    if (!response_list.responses().empty()) {
      std::string tensors_ready;
      for (auto r : response_list.responses()) {
        tensors_ready += r.tensor_names_string() + "; " ;
      }
      LOG(TRACE) << "Sending ready responses as " << tensors_ready;
    }

    // Notify all nodes which tensors we'd like to reduce at this step.
    std::string encoded_response;
    ResponseList::SerializeToString(response_list, encoded_response);
    int encoded_response_length = (int)encoded_response.length() + 1;
    MPI_Bcast(&encoded_response_length, 1, MPI_INT, RANK_ZERO, state.mpi_comm);
    MPI_Bcast((void*)encoded_response.c_str(), encoded_response_length,
              MPI_BYTE, RANK_ZERO, state.mpi_comm);

    std::vector<std::string> tensor_names;
    int64_t total_tensor_size = 0;
    if (state.param_manager.IsAutoTuning()) {
      for (auto& response : response_list.responses()) {
        if (response.response_type() == Response::ResponseType::ALLREDUCE) {
          for (auto& tensor_name : response.tensor_names()) {
            tensor_names.push_back(tensor_name);
            auto& entry = state.tensor_table[tensor_name];
            total_tensor_size += entry.tensor->size();
          }
        }
      }
    }

    // Perform the collective operation. All nodes should end up performing
    // the same operation.
    for (auto& response : response_list.responses()) {
      LOG(TRACE, state.rank) << "Performing " << response.tensor_names_string();
      LOG(DEBUG, state.rank) << "Processing " << response.tensor_names().size() << " tensors";
      PerformOperation(state.tensor_table, response);
      LOG(TRACE, state.rank) << "Finished performing " << response.tensor_names_string();
    }

    // Check for stalled tensors.
    if (state.perform_stall_check &&
        std::chrono::steady_clock::now() - state.last_stall_check >
            STALL_WARNING_TIME) {
      CheckForStalledTensors(state);
      state.last_stall_check = std::chrono::steady_clock::now();
    }

    if (state.param_manager.IsAutoTuning()) {
      state.param_manager.Update(tensor_names, total_tensor_size);
    }
  } else {
    std::string encoded_message;
    RequestList message_list;
    message_list.set_shutdown(should_shut_down);
    while (!message_queue.empty()) {
      message_list.add_request(message_queue.front());
      message_queue.pop();
    }
    RequestList::SerializeToString(message_list, encoded_message);
    int encoded_message_length = (int)encoded_message.length() + 1;
    MPI_Gather(&encoded_message_length, 1, MPI_INT, nullptr, 1, MPI_INT,
               RANK_ZERO, state.mpi_comm);
    MPI_Gatherv((void*)encoded_message.c_str(), encoded_message_length,
                MPI_BYTE, nullptr, nullptr, nullptr, MPI_BYTE, RANK_ZERO,
                state.mpi_comm);

    int msg_length;
    MPI_Bcast(&msg_length, 1, MPI_INT, RANK_ZERO, state.mpi_comm);
    auto buffer = new uint8_t[msg_length];
    MPI_Bcast(buffer, msg_length, MPI_BYTE, RANK_ZERO, state.mpi_comm);
    ResponseList response_list;
    ResponseList::ParseFromBytes(response_list, buffer);
    delete[] buffer;

    std::vector<std::string> tensor_names;
    int64_t total_tensor_size = 0;
    if (state.param_manager.IsAutoTuning()) {
      for (auto& response : response_list.responses()) {
        if (response.response_type() == Response::ResponseType::ALLREDUCE) {
          for (auto& tensor_name : response.tensor_names()) {
            tensor_names.push_back(tensor_name);
            auto& entry = state.tensor_table[tensor_name];
            total_tensor_size += entry.tensor->size();
          }
        }
      }
    }

    // Perform the collective operation. All nodes should end up performing
    // the same operation.
    for (auto& response : response_list.responses()) {
      LOG(TRACE, state.rank) << "Performing " << response.tensor_names_string();
      LOG(DEBUG, state.rank) << "Processing " << response.tensor_names().size() << " tensors";
      PerformOperation(state.tensor_table, response);
      LOG(TRACE, state.rank) << "Finished performing " << response.tensor_names_string();
    }

    if (state.param_manager.IsAutoTuning()) {
      state.param_manager.Update(tensor_names, total_tensor_size);
    }

    if (response_list.shutdown()) {
      should_shut_down = true;
    }
  }

  return !should_shut_down;
}

// Start Horovod background thread. Ensure that this is
// only done once no matter how many times this function is called.
void InitializeHorovodOnce(const int* ranks, int nranks) {
  // Ensure background thread is only started once.
  if (!horovod_global.initialize_flag.test_and_set()) {
    for (int i = 0; i < nranks; ++i) {
      horovod_global.ranks.push_back(ranks[i]);
    }

    // Reset initialization flag
    horovod_global.initialization_done = false;

    horovod_global.background_thread =
        std::thread(BackgroundThreadLoop, std::ref(horovod_global));
  }

  // Wait to ensure that the background thread has finished initializing MPI.
  while (!horovod_global.initialization_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

} // namespace

Status CheckInitialized() {
  if (!horovod_global.initialization_done) {
    return NOT_INITIALIZED_ERROR;
  }
  return Status::OK();
}

extern "C" {

void horovod_init(const int* ranks, int nranks) {
  InitializeHorovodOnce(ranks, nranks);
}

void horovod_init_comm(MPI_Comm comm) {
  MPI_Comm_dup(comm, &(horovod_global.mpi_comm));
  InitializeHorovodOnce(NULL, 0);
}

void horovod_shutdown() {
  if (horovod_global.background_thread.joinable()) {
    horovod_global.shut_down = true;
    horovod_global.background_thread.join();
    // Reset the initialization flag to allow restarting with horovod_init(...)
    horovod_global.initialize_flag.clear();
    horovod_global.shut_down = false;
  }
}

int horovod_rank() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.rank;
}

int horovod_local_rank() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.local_rank;
}

int horovod_size() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.size;
}

int horovod_local_size() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.local_size;
}

int horovod_mpi_threads_supported() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.mpi_threads_supported ? 1 : 0;
}
}

// MPI must be initialized and the background thread must be running before
// this function is called.
Status EnqueueTensorAllreduce(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback) {
  Request message;
  message.set_request_rank(horovod_global.rank);
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(Request::ALLREDUCE);
  for (int i = 0; i < tensor->shape().dims(); ++i) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.output = output;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  std::lock_guard<std::mutex> guard(horovod_global.mutex);
  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  if (horovod_global.tensor_table.find(name) !=
      horovod_global.tensor_table.end()) {
    return DUPLICATE_NAME_ERROR;
  }
  horovod_global.tensor_table.emplace(name, std::move(e));
  horovod_global.message_queue.push(message);
  LOG(TRACE, horovod_global.rank) << "Enqueued " << name;
  return Status::OK();
}

// MPI must be initialized and the background thread must be running before
// this function is called.
Status EnqueueTensorAllgather(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback) {
  Request message;
  message.set_request_rank(horovod_global.rank);
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(Request::ALLGATHER);
  for (int i = 0; i < tensor->shape().dims(); ++i) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  std::lock_guard<std::mutex> guard(horovod_global.mutex);
  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  if (horovod_global.tensor_table.find(name) !=
      horovod_global.tensor_table.end()) {
    return DUPLICATE_NAME_ERROR;
  }
  horovod_global.tensor_table.emplace(name, std::move(e));
  horovod_global.message_queue.push(message);
  LOG(TRACE, horovod_global.rank) << "Enqueued " << name;
  return Status::OK();
}

// MPI must be initialized and the background thread must be running before
// this function is called.
Status EnqueueTensorBroadcast(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output, int root_rank,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback) {
  Request message;
  message.set_request_rank(horovod_global.rank);
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_root_rank(root_rank);
  message.set_device(device);
  message.set_request_type(Request::BROADCAST);
  for (int i = 0; i < tensor->shape().dims(); ++i) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.output = output;
  e.root_rank = root_rank;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  std::lock_guard<std::mutex> guard(horovod_global.mutex);
  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  if (horovod_global.tensor_table.find(name) !=
      horovod_global.tensor_table.end()) {
    return DUPLICATE_NAME_ERROR;
  }
  horovod_global.tensor_table.emplace(name, std::move(e));
  horovod_global.message_queue.push(message);
  LOG(TRACE, horovod_global.rank) << "Enqueued " << name;
  return Status::OK();
}

} // namespace common
} // namespace horovod
