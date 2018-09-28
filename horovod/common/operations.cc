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
#include "hashes.h"
#include "mpi.h"
#include "mpi_message.h"
#include "operations.h"
#include "timeline.h"

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
    std::tuple<std::vector<MPIRequest>, std::chrono::steady_clock::time_point>>;

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

  // Queue of MPI requests waiting to be sent to the coordinator node.
  std::queue<MPIRequest> message_queue;

  // Background thread running MPI communication.
  std::thread background_thread;

  // Whether the background thread should shutdown.
  bool shut_down = false;

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

  // Threshold for Tensor Fusion.  All tensors that occupy memory beyond this
  // threshold will be fused.
  int64_t tensor_fusion_threshold = 64 * 1024 * 1024;

  // Background thread cycle time in milliseconds.  Fractional numbers are
  // permitted.
  double cycle_time_ms = 5;

  // Time point when last cycle started.
  std::chrono::steady_clock::time_point last_cycle_start;

  // Memory buffers for Tensor Fusion.  They are keyed off device ID and
  // framework, and all are allocated tensor_fusion_threshold bytes if
  // initialized.
  std::unordered_map<std::tuple<int, Framework>,
                     std::shared_ptr<PersistentBuffer>>
      tensor_fusion_buffers;

  // Whether MPI_Init has been completed on the background thread.
  bool initialization_done = false;

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

  // MPI custom data type for float16.
  MPI_Datatype mpi_float16_t;

  // Private MPI communicator for Horovod to ensure no collisions with other
  // threads using MPI.
  MPI_Comm mpi_comm;

  // Node-local communicator.
  MPI_Comm local_comm;

  // Cross-node communicator for hierarchical allreduce.
  MPI_Comm cross_comm;

  // Do hierarchical allreduce with MPI + NCCL.
  bool hierarchical_allreduce = false;

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
// Event management code is only used in NCCL path.
#if HAVE_NCCL || HAVE_DDL
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

const Status SHUT_DOWN_ERROR = Status::Aborted(
    "Horovod has been shut down. This was caused by an exception on one of the "
    "ranks or an attempt to allreduce, allgather or broadcast a tensor after "
    "one of the ranks finished execution. If the shutdown was caused by an "
    "exception, you should see the exception in the log before the first "
    "shutdown message.");

#define OP_ERROR(entries, error_message)                                       \
  {                                                                            \
    for (auto& e : (entries)) {                                                \
      timeline.End(e.tensor_name, nullptr);                                    \
      e.callback(Status::UnknownError(error_message));                         \
    }                                                                          \
    return;                                                                    \
  }

// Store the MPIRequest for a name, and return whether the total count of
// MPIRequests for that tensor is now equal to the MPI size (and thus we are
// ready to reduce the tensor).
bool IncrementTensorCount(std::unique_ptr<MessageTable>& message_table,
                          MPIRequest msg, int mpi_size) {
  auto& name = msg.tensor_name();
  auto& timeline = horovod_global.timeline;
  auto table_iter = message_table->find(name);
  if (table_iter == message_table->end()) {
    std::vector<MPIRequest> messages = {msg};
    auto now = std::chrono::steady_clock::now();
    message_table->emplace(name, std::make_tuple(std::move(messages), now));
    table_iter = message_table->find(name);
    timeline.NegotiateStart(name, msg.request_type());
  } else {
    std::vector<MPIRequest>& messages = std::get<0>(table_iter->second);
    messages.push_back(msg);
  }

  timeline.NegotiateRankReady(name, msg.request_rank());

  std::vector<MPIRequest>& messages = std::get<0>(table_iter->second);
  int count = (int)messages.size();
  bool ready_to_reduce = count == mpi_size;
  if (ready_to_reduce) {
    timeline.NegotiateEnd(name);
  }
  return ready_to_reduce;
}

// Once a tensor is ready to be reduced, the coordinator sends an MPIResponse
// instructing all ranks to start the reduction to all ranks. The MPIResponse
// also contains error messages in case the submitted MPIRequests were not
// valid (for example, contained mismatched shapes or types).
//
// Constructing the MPIResponse, thus, requires a whole lot of error checking.
MPIResponse ConstructMPIResponse(std::unique_ptr<MessageTable>& message_table,
                                 std::string name) {
  bool error = false;
  auto it = message_table->find(name);
  assert(it != message_table->end());

  std::vector<MPIRequest>& requests = std::get<0>(it->second);
  assert(requests.size() > 0);

  std::ostringstream error_message_stream;

  // Check that all data types of tensors being reduced, gathered or broadcasted
  // are identical.
  auto data_type = requests[0].tensor_type();
  for (unsigned int i = 1; i < requests.size(); i++) {
    auto request_type = requests[i].tensor_type();
    if (data_type != request_type) {
      error = true;
      error_message_stream << "Mismatched data types: One rank had type "
                           << MPIDataType_Name(data_type)
                           << ", but another rank had type "
                           << MPIDataType_Name(request_type) << ".";
      break;
    }
  }

  // Check that all requested operations are the same
  auto message_type = requests[0].request_type();
  for (unsigned int i = 1; i < requests.size(); i++) {
    if (error) {
      break;
    }

    auto request_type = requests[i].request_type();
    if (message_type != request_type) {
      error = true;
      error_message_stream << "Mismatched MPI operations: One rank did an "
                           << MPIRequest::RequestType_Name(message_type)
                           << ", but another rank did an "
                           << MPIRequest::RequestType_Name(request_type) << ".";
      break;
    }
  }

  // If we are doing an allreduce or broadcast, check that all tensor shapes are
  // identical.
  if (message_type == MPIRequest::ALLREDUCE ||
      message_type == MPIRequest::BROADCAST) {
    TensorShape tensor_shape;
    for (auto dim : requests[0].tensor_shape()) {
      tensor_shape.AddDim(dim);
    }
    for (unsigned int i = 1; i < requests.size(); i++) {
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
            << "Mismatched " << MPIRequest::RequestType_Name(message_type)
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
  if (message_type == MPIRequest::ALLGATHER) {
    TensorShape tensor_shape;
    for (auto dim : requests[0].tensor_shape()) {
      tensor_shape.AddDim(dim);
    }

    if (tensor_shape.dims() == 0) {
      error = true;
      error_message_stream << "Rank zero tried to "
                           << MPIRequest::RequestType_Name(message_type)
                           << " a rank-zero tensor.";
    } else {
      tensor_sizes[requests[0].request_rank()] = tensor_shape.dim_size(0);
    }

    for (unsigned int i = 1; i < requests.size(); i++) {
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
            << "Mismatched " << MPIRequest::RequestType_Name(message_type)
            << " tensor shapes: One rank sent a tensor of rank "
            << tensor_shape.dims()
            << ", but another rank sent a tensor of rank "
            << request_shape.dims() << ".";
        break;
      }

      bool dim_mismatch = false;
      for (int dim = 1; dim < tensor_shape.dims(); dim++) {
        if (tensor_shape.dim_size(dim) != request_shape.dim_size(dim)) {
          error = true;
          error_message_stream
              << "Mismatched " << MPIRequest::RequestType_Name(message_type)
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
  if (message_type == MPIRequest::BROADCAST) {
    int first_root_rank = requests[0].root_rank();
    for (unsigned int i = 1; i < requests.size(); i++) {
      if (error) {
        break;
      }

      int this_root_rank = requests[i].root_rank();
      if (first_root_rank != this_root_rank) {
        error = true;
        error_message_stream
            << "Mismatched " << MPIRequest::RequestType_Name(message_type)
            << " root ranks: One rank specified root rank " << first_root_rank
            << ", but another rank specified root rank " << this_root_rank
            << ".";
        break;
      }
    }
  }

  bool first_device_is_cpu = requests[0].device() == CPU_DEVICE_ID;
  for (unsigned int i = 1; i < requests.size(); i++) {
    if (error) {
      break;
    }

    bool this_device_is_cpu = requests[i].device() == CPU_DEVICE_ID;
    if (first_device_is_cpu != this_device_is_cpu) {
      error = true;
      error_message_stream
          << "Mismatched " << MPIRequest::RequestType_Name(message_type)
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

  MPIResponse response;
  response.add_tensor_names(name);
  if (error) {
    std::string error_message = error_message_stream.str();
    response.set_response_type(MPIResponse::ERROR);
    response.set_error_message(error_message);
  } else if (message_type == MPIRequest::ALLGATHER) {
    response.set_response_type(MPIResponse::ALLGATHER);
    for (auto dim : tensor_sizes) {
      response.add_tensor_sizes(dim);
    }
  } else if (message_type == MPIRequest::ALLREDUCE) {
    response.set_response_type(MPIResponse::ALLREDUCE);
  } else if (message_type == MPIRequest::BROADCAST) {
    response.set_response_type(MPIResponse::BROADCAST);
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
    throw std::logic_error("Type " + MPIDataType_Name(tensor->dtype()) +
                           " is not supported in MPI mode.");
  }
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
    throw std::logic_error("Type " + MPIDataType_Name(tensor->dtype()) +
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
    throw std::logic_error("Type " + MPIDataType_Name(tensor->dtype()) +
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

// This event management code is only used in NCCL.
#if HAVE_NCCL || HAVE_DDL
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

// Process an MPIResponse by doing a reduction, a gather, a broadcast, or
// raising an error.
void PerformOperation(TensorTable& tensor_table, MPIResponse response) {
  std::vector<TensorTableEntry> entries;
  {
    // Lock on the tensor table.
    std::lock_guard<std::mutex> guard(horovod_global.mutex);

    for (auto& name : response.tensor_names()) {
      // We should never fail at finding this key in the tensor table.
      auto iter = tensor_table.find(name);
      assert(iter != tensor_table.end());

      assert(response.response_type() == MPIResponse::ALLREDUCE ||
             response.response_type() == MPIResponse::ALLGATHER ||
             response.response_type() == MPIResponse::BROADCAST ||
             response.response_type() == MPIResponse::ERROR);

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
    auto& buffer = horovod_global.tensor_fusion_buffers[std::make_tuple(
        first_entry.device, first_entry.context->framework())];
    if (buffer == nullptr) {
      ACTIVITY_START_ALL(entries, timeline, INIT_FUSION_BUFFER)

      // Lazily allocate persistent buffer for Tensor Fusion and keep it
      // forever per device.
      Status status = first_entry.context->AllocatePersistent(
          horovod_global.tensor_fusion_threshold, &buffer);
      if (!status.ok()) {
        for (auto& e : entries) {
          timeline.End(e.tensor_name, nullptr);
          e.callback(status);
        }
        return;
      }

      ACTIVITY_END_ALL(entries, timeline)
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
  if (response.response_type() == MPIResponse::ALLGATHER) {
    assert(entries.size() == 1);
    auto e = entries[0];

    // Copy tensor sizes from the MPI response into a vector of int64_t
    // and compute total size.  This is size of first dimension.
    std::vector<int64_t> tensor_sizes;
    int64_t total_dimension_size = 0;
    for (auto sz : response.tensor_sizes()) {
      tensor_sizes.push_back(sz);
      total_dimension_size += sz;
    }

    // Every tensor participating in Allgather operation may have different
    // first dimension size, but the rest of dimensions are same for all
    // tensors.  Here we get shape of tensor sliced by first dimension.
    TensorShape single_slice_shape;
    for (int i = 1; i < e.tensor->shape().dims(); ++i) {
      single_slice_shape.AddDim(e.tensor->shape().dim_size(i));
    }

    // Allgather output will have shape of:
    // (sum of first dimension of every tensor) x (tensor slice shape).
    TensorShape output_shape;
    output_shape.AddDim((int64_t)total_dimension_size);
    output_shape.AppendShape(single_slice_shape);

    ACTIVITY_START_ALL(entries, timeline, ALLOCATE_OUTPUT)
    status = e.context->AllocateOutput(output_shape, &e.output);
    if (!status.ok()) {
      timeline.End(e.tensor_name, nullptr);
      e.callback(status);
      return;
    }
    ACTIVITY_END_ALL(entries, timeline)

    // Tensors may have different first dimension, so we need to use
    // MPI_Allgatherv API that supports gathering arrays of different length.
    ACTIVITY_START_ALL(entries, timeline, MPI_ALLGATHER)
    auto* recvcounts = new int[tensor_sizes.size()];
    auto* displcmnts = new int[tensor_sizes.size()];
    for (unsigned int i = 0; i < tensor_sizes.size(); i++) {
      recvcounts[i] =
          (int)(single_slice_shape.num_elements() * tensor_sizes[i]);
      if (i == 0) {
        displcmnts[i] = 0;
      } else {
        displcmnts[i] = recvcounts[i - 1] + displcmnts[i - 1];
      }
    }
    auto result = MPI_Allgatherv(
        e.tensor->data(), (int)e.tensor->shape().num_elements(),
        GetMPIDataType(e.tensor), (void*)e.output->data(), recvcounts,
        displcmnts, GetMPIDataType(e.tensor), horovod_global.mpi_comm);
    delete[] recvcounts;
    delete[] displcmnts;
    MPI_CHECK(entries, "MPI_Allgatherv", result)
    ACTIVITY_END_ALL(entries, timeline)

    timeline.End(e.tensor_name, e.output);
    e.callback(Status::OK());

  } else if (response.response_type() == MPIResponse::ALLREDUCE) {
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
      if (horovod_global.hierarchical_allreduce) {
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
        if (horovod_global.hierarchical_allreduce) {
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
        auto& buffer = horovod_global.tensor_fusion_buffers[std::make_tuple(
            first_entry.device, first_entry.context->framework())];
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
      if (horovod_global.hierarchical_allreduce) {
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
                                  GetMPIDataType(first_entry.tensor), MPI_SUM,
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
      auto& buffer = horovod_global.tensor_fusion_buffers[std::make_tuple(
          first_entry.device, first_entry.context->framework())];
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
                              GetMPIDataType(first_entry.tensor), MPI_SUM,
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
                              GetMPIDataType(e.tensor), MPI_SUM,
                              horovod_global.mpi_comm))
      ACTIVITY_END_ALL(entries, timeline)
    }

    for (auto& e : entries) {
      timeline.End(e.tensor_name, e.output);
      e.callback(Status::OK());
    }
  } else if (response.response_type() == MPIResponse::BROADCAST) {
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
  } else if (response.response_type() == MPIResponse::ERROR) {
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
    std::vector<MPIRequest>& messages = std::get<0>(m.second);
    std::chrono::steady_clock::time_point start_at = std::get<1>(m.second);

    if (now - start_at > STALL_WARNING_TIME) {
      if (!preamble) {
        std::cerr << "WARNING: One or more tensors were submitted to be "
                     "reduced, gathered or broadcasted by subset of ranks and "
                     "are waiting for remainder of ranks for more than "
                  << std::chrono::duration_cast<std::chrono::seconds>(
                         STALL_WARNING_TIME)
                         .count()
                  << " seconds. ";
        std::cerr << "This may indicate that different ranks are trying to "
                     "submit different tensors or that only subset of ranks is "
                     "submitting tensors, which will cause deadlock. "
                  << std::endl;
        std::cerr << "Stalled ops:" << std::endl;
        preamble = true;
      }
      std::cerr << tensor_name;
      std::cerr << " [missing ranks:";
      std::unordered_set<int32_t> ready_ranks;
      bool missing_preamble = false;
      for (auto msg_iter = messages.begin(); msg_iter != messages.end();
           msg_iter++) {
        ready_ranks.insert(msg_iter->request_rank());
      }
      for (int32_t rank = 0; rank < state.size; rank++) {
        if (ready_ranks.find(rank) == ready_ranks.end()) {
          if (!missing_preamble) {
            std::cerr << " ";
            missing_preamble = true;
          } else {
            std::cerr << ", ";
          }
          std::cerr << rank;
        }
      }
      std::cerr << "]" << std::endl;
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
      std::cerr << "WARNING: MPI has already been initialized without "
                   "multi-threading support (MPI_THREAD_MULTIPLE). This will "
                   "likely cause a segmentation fault."
                << std::endl;
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
      std::cerr << "WARNING: Unable to create Horovod communicator, using "
                   "MPI_COMM_WORLD instead."
                << std::endl;
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
  for (int i = 0; i < size; i++) {
    if (local_sizes[i] != local_size) {
      is_homogeneous = false;
      break;
    }
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

  state.rank = rank;
  state.local_rank = local_rank;
  state.cross_rank = cross_rank;
  state.size = size;
  state.local_size = local_size;
  state.cross_size = cross_size;
  state.local_comm = local_comm;
  state.cross_comm = cross_comm;
  state.mpi_float16_t = mpi_float16_t;
  state.mpi_threads_supported = (provided == MPI_THREAD_MULTIPLE);
  state.local_comm_ranks = local_comm_ranks;

  // Open the timeline file on coordinator.
  auto horovod_timeline = std::getenv(HOROVOD_TIMELINE);
  if (is_coordinator && horovod_timeline != nullptr) {
    state.timeline.Initialize(std::string(horovod_timeline));
  }

  // Override the cycle time.
  auto horovod_cycle_time = std::getenv(HOROVOD_CYCLE_TIME);
  if (horovod_cycle_time != nullptr) {
    state.cycle_time_ms = std::strtof(horovod_cycle_time, nullptr);
  }

  // Disable stall check.
  auto horovod_stall_check_disable = std::getenv(HOROVOD_STALL_CHECK_DISABLE);
  if (horovod_stall_check_disable != nullptr &&
      std::strtol(horovod_stall_check_disable, nullptr, 10) > 0) {
    state.perform_stall_check = false;
  }

  // Set flag for hierarchical allreduce. Ignore if Horovod is running on a
  // single node.
  auto horovod_hierarchical_allreduce =
      std::getenv(HOROVOD_HIERARCHICAL_ALLREDUCE);
  if (horovod_hierarchical_allreduce != nullptr &&
      std::strtol(horovod_hierarchical_allreduce, nullptr, 10) > 0 &&
      (size != local_size)) {
    state.hierarchical_allreduce = true;
  }

  // Issue warning if hierarchical allreduce is enabled in heterogeneous cluster
  if (is_coordinator && state.hierarchical_allreduce && !state.is_homogeneous) {
    std::cerr
        << "WARNING: Using different number of ranks per node might hurt "
           "performance of hierarchical allreduce. Consider assigning the same "
           "number of ranks to each node or disabling hierarchical allreduce."
        << std::endl;
  }

  // Override Tensor Fusion threshold, if it's set.
  auto horovod_fusion_threshold = std::getenv("HOROVOD_FUSION_THRESHOLD");
  int64_t proposed_fusion_threshold = (horovod_fusion_threshold != nullptr) ?
        std::strtol(horovod_fusion_threshold, nullptr, 10) :
        state.tensor_fusion_threshold;

  // If the cluster is homogeneous and hierarchical allreduce is enabled,
  // adjust buffer size to make sure it is divisible by local_size to improve
  // performance.
  if (state.is_homogeneous && state.hierarchical_allreduce) {
    // Assume the worst-case data type float64, since if it is divisible with
    // float64, it will be divisible for other types too.

    // Ensuring that fusion buffer can hold a number of elements divisible by
    // FUSION_BUFFER_ATOMIC_UNIT for performance
    int mpi_double_size;
    MPI_Type_size(MPI_DOUBLE, &mpi_double_size);
    int64_t div = state.local_size * mpi_double_size * FUSION_BUFFER_ATOMIC_UNIT;
    state.tensor_fusion_threshold = ((proposed_fusion_threshold+div-1) / div) * div;
  } else {
    state.tensor_fusion_threshold = proposed_fusion_threshold;
  }

  // Initialize the tensor count table. No tensors are available yet.
  if (is_coordinator) {
    state.message_table = std::unique_ptr<MessageTable>(new MessageTable());
  }

  // Signal that initialization is completed.
  state.initialization_done = true;

  // Iterate until shutdown.
  while (RunLoopOnce(state, is_coordinator))
    ;

  // TODO: init.cu:645 WARN Cuda failure 'driver shutting down'
  //#if HAVE_NCCL
  //  for (auto it = horovod_global.streams.begin();
  //       it != horovod_global.streams.end(); it++) {
  //    cudaStreamSynchronize(it->second);
  //  }
  //  for (auto it = horovod_global.nccl_comms.begin();
  //       it != horovod_global.nccl_comms.end(); it++) {
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
}

// The coordinator currently follows a master-worker paradigm. Rank zero acts
// as the master (the "coordinator"), whereas all other ranks are simply
// workers. Each rank runs its own background thread which progresses in ticks.
// In each tick, the following actions happen:
//
//      a) The workers send an MPIRequest to the coordinator, indicating what
//      they would like to do (which tensor they would like to gather and
//      reduce, as well as their shape and type). They repeat this for every
//      tensor that they would like to operate on.
//
//      b) The workers send an empty "DONE" message to the coordinator to
//      indicate that there are no more tensors they wish to operate on.
//
//      c) The coordinator receives the MPIRequests from the workers, as well
//      as from its own TensorFlow ops, and stores them in a request table. The
//      coordinator continues to receive MPIRequest messages until it has
//      received MPI_SIZE number of empty "DONE" messages.
//
//      d) The coordinator finds all tensors that are ready to be reduced,
//      gathered, or all operations that result in an error. For each of those,
//      it sends an MPIResponse to all the workers. When no more MPIResponses
//      are available, it sends a "DONE" response to the workers. If the process
//      is being shutdown, it instead sends a "SHUTDOWN" response.
//
//      e) The workers listen for MPIResponse messages, processing each one by
//      doing the required reduce or gather, until they receive a "DONE"
//      response from the coordinator. At that point, the tick ends.
//      If instead of "DONE" they receive "SHUTDOWN", they exit their background
//      loop.
bool RunLoopOnce(HorovodGlobalState& state, bool is_coordinator) {
  // The coordinator sends a SHUTDOWN message to trigger shutdown.
  bool should_shut_down = false;

  // This delay determines thread frequency and MPI message latency
  auto sleep_duration =
      state.last_cycle_start +
      std::chrono::microseconds(long(state.cycle_time_ms * 1000.)) -
      std::chrono::steady_clock::now();
  if (sleep_duration > std::chrono::steady_clock::duration::zero()) {
    std::this_thread::sleep_for(sleep_duration);
  }
  state.last_cycle_start = std::chrono::steady_clock::now();

  // Copy the data structures from global state under this lock.
  // However, don't keep the lock for the rest of the loop, so that
  // enqueued stream callbacks can continue.
  std::queue<MPIRequest> message_queue;
  {
    std::lock_guard<std::mutex> guard(state.mutex);
    while (!state.message_queue.empty()) {
      MPIRequest message = state.message_queue.front();
      state.message_queue.pop();
      message_queue.push(message);
    }
  }

  // Collect all tensors that are ready to be reduced. Record them in the
  // tensor count table (rank zero) or send them to rank zero to be
  // recorded (everyone else).
  std::vector<std::string> ready_to_reduce;
  if (is_coordinator) {
    while (!message_queue.empty()) {
      // Pop the first available message message
      MPIRequest message = message_queue.front();
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
    for (int i = 0; i < state.size; i++) {
      if (i == 0) {
        displcmnts[i] = 0;
      } else {
        displcmnts[i] = recvcounts[i - 1] + displcmnts[i - 1];
      }
      total_size += recvcounts[i];
    }

    // 3. Collect messages from every rank.
    auto buffer = new char[total_size];
    MPI_Gatherv(nullptr, 0, MPI_BYTE, buffer, recvcounts, displcmnts, MPI_BYTE,
                RANK_ZERO, state.mpi_comm);

    // 4. Process messages.
    for (int i = 1; i < state.size; i++) {
      std::string received_data(buffer + displcmnts[i], (size_t)recvcounts[i]);

      MPIRequestList received_message_list;
      MPIRequestList::ParseFromString(received_message_list, received_data);
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
        state.shut_down = true;
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
    std::deque<MPIResponse> responses;
    for (auto& tensor_name : ready_to_reduce) {
      MPIResponse response =
          ConstructMPIResponse(state.message_table, tensor_name);
      responses.push_back(std::move(response));
    }

    MPIResponseList response_list;
    response_list.set_shutdown(state.shut_down);
    should_shut_down = state.shut_down;

    while (!responses.empty()) {
      auto response = responses.front();
      assert(response.tensor_names().size() == 1);
      responses.pop_front();

      if (response.response_type() == MPIResponse::ResponseType::ALLREDUCE) {
        // Attempt to add more responses to this fused response.
        auto& entry = state.tensor_table[response.tensor_names()[0]];
        int64_t tensor_size = entry.tensor->size();

        while (!responses.empty()) {
          auto new_response = responses.front();
          assert(new_response.tensor_names().size() == 1);
          auto& new_entry = state.tensor_table[new_response.tensor_names()[0]];
          int64_t new_tensor_size = new_entry.tensor->size();

          if (response.response_type() == new_response.response_type() &&
              response.devices() == new_response.devices() &&
              entry.tensor->dtype() == new_entry.tensor->dtype() &&
              tensor_size + new_tensor_size <= state.tensor_fusion_threshold) {
            // These tensors will fuse together well.
            tensor_size += new_tensor_size;
            response.add_tensor_names(new_response.tensor_names()[0]);
            responses.pop_front();
          } else {
            // Don't try to fuse additional tensors since they are usually
            // computed in order of requests and skipping tensors may mean
            // that the batch will have to wait longer while skipped tensors
            // could be reduced at that time.
            break;
          }
        }
      }

      response_list.add_responses(response);
    }

    // Notify all nodes which tensors we'd like to reduce at this step.
    std::string encoded_response;
    MPIResponseList::SerializeToString(response_list, encoded_response);
    int encoded_response_length = (int)encoded_response.length() + 1;
    MPI_Bcast(&encoded_response_length, 1, MPI_INT, RANK_ZERO, state.mpi_comm);
    MPI_Bcast((void*)encoded_response.c_str(), encoded_response_length,
              MPI_BYTE, RANK_ZERO, state.mpi_comm);

    // Perform the collective operation. All nodes should end up performing
    // the same operation.
    for (auto& response : response_list.responses()) {
      PerformOperation(state.tensor_table, response);
    }

    // Check for stalled tensors.
    if (state.perform_stall_check &&
        std::chrono::steady_clock::now() - state.last_stall_check >
            STALL_WARNING_TIME) {
      CheckForStalledTensors(state);
      state.last_stall_check = std::chrono::steady_clock::now();
    }
  } else {
    std::string encoded_message;
    MPIRequestList message_list;
    message_list.set_shutdown(state.shut_down);
    while (!message_queue.empty()) {
      message_list.add_requests(message_queue.front());
      message_queue.pop();
    }
    MPIRequestList::SerializeToString(message_list, encoded_message);
    int encoded_message_length = (int)encoded_message.length() + 1;
    MPI_Gather(&encoded_message_length, 1, MPI_INT, nullptr, 1, MPI_INT,
               RANK_ZERO, state.mpi_comm);
    MPI_Gatherv((void*)encoded_message.c_str(), encoded_message_length,
                MPI_BYTE, nullptr, nullptr, nullptr, MPI_BYTE, RANK_ZERO,
                state.mpi_comm);

    int msg_length;
    MPI_Bcast(&msg_length, 1, MPI_INT, RANK_ZERO, state.mpi_comm);
    auto buffer = new char[msg_length];
    MPI_Bcast(buffer, msg_length, MPI_BYTE, RANK_ZERO, state.mpi_comm);
    std::string received_message(buffer, (size_t)msg_length);
    MPIResponseList response_list;
    MPIResponseList::ParseFromString(response_list, received_message);
    delete[] buffer;

    // Perform the collective operation. All nodes should end up performing
    // the same operation.
    for (auto& response : response_list.responses()) {
      PerformOperation(state.tensor_table, response);
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
    for (int i = 0; i < nranks; i++) {
      horovod_global.ranks.push_back(ranks[i]);
    }

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
  MPIRequest message;
  message.set_request_rank(horovod_global.rank);
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(MPIRequest::ALLREDUCE);
  for (int i = 0; i < tensor->shape().dims(); i++) {
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
  if (!horovod_global.shut_down) {
    horovod_global.tensor_table.emplace(name, std::move(e));
    horovod_global.message_queue.push(message);
    return Status::OK();
  } else {
    return SHUT_DOWN_ERROR;
  }
}

// MPI must be initialized and the background thread must be running before
// this function is called.
Status EnqueueTensorAllgather(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback) {
  MPIRequest message;
  message.set_request_rank(horovod_global.rank);
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(MPIRequest::ALLGATHER);
  for (int i = 0; i < tensor->shape().dims(); i++) {
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
  if (!horovod_global.shut_down) {
    horovod_global.tensor_table.emplace(name, std::move(e));
    horovod_global.message_queue.push(message);
    return Status::OK();
  } else {
    return SHUT_DOWN_ERROR;
  }
}

// MPI must be initialized and the background thread must be running before
// this function is called.
Status EnqueueTensorBroadcast(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output, int root_rank,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback) {
  MPIRequest message;
  message.set_request_rank(horovod_global.rank);
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_root_rank(root_rank);
  message.set_device(device);
  message.set_request_type(MPIRequest::BROADCAST);
  for (int i = 0; i < tensor->shape().dims(); i++) {
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
  if (!horovod_global.shut_down) {
    horovod_global.tensor_table.emplace(name, std::move(e));
    horovod_global.message_queue.push(message);
    return Status::OK();
  } else {
    return SHUT_DOWN_ERROR;
  }
}

} // namespace common
} // namespace horovod
