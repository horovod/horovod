// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2017 Uber Technologies, Inc.
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

#include <queue>
#include <thread>
#include <unordered_map>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#define EIGEN_USE_THREADS

#if HAVE_CUDA
#include "tensorflow/stream_executor/stream.h"
#include <cuda_runtime.h>
#endif

#if HAVE_NCCL
#include <nccl.h>
#endif

#define OMPI_SKIP_MPICXX
#include "mpi.h"
#include "mpi_message.h"
#include "hash_vector.h"

/*
 * Allreduce, Allgather and Broadcast Ops for TensorFlow.
 *
 * TensorFlow natively provides inter-device communication through send and
 * receive ops and inter-node communication through Distributed TensorFlow,
 * based on the same send and receive abstractions. These end up being
 * insufficient for synchronous data-parallel training on HPC clusters where
 * Infiniband or other high-speed interconnects are available.  This module
 * implements MPI ops for allgather, allreduce and broadcast, which do
 * optimized gathers, reductions and broadcasts and can take advantage of
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

using namespace tensorflow;

namespace horovod {
namespace tensorflow {

namespace {

// Device ID used for CPU.
#define CPU_DEVICE_ID -1

// Use void pointer for ready event if CUDA is not present to avoid linking
// error.
#if HAVE_CUDA
#define GPU_EVENT_IF_CUDA perftools::gputools::Event*
#else
#define GPU_EVENT_IF_CUDA void*
#endif

// A callback to call after the MPI communication completes. Since the
// allreduce and allgather ops are asynchronous, this callback is what resumes
// computation after the reduction is completed.
typedef std::function<void(const Status&)> StatusCallback;

// Table storing Tensors to be reduced, keyed by unique name.
// This table contains everything necessary to do the reduction.
typedef struct {
  // Operation context.
  OpKernelContext* context;
  // Input tensor.
  Tensor tensor;
  // Pre-allocated output tensor.
  Tensor* output;
  // Root rank for broadcast operation.
  int root_rank;
  // Event indicating that data is ready.
  GPU_EVENT_IF_CUDA ready_event;
  // GPU to do reduction on, or CPU_DEVICE_ID in case of CPU.
  int device;
  // A callback to call with the status.
  StatusCallback callback;
} TensorTableEntry;
typedef std::unordered_map<std::string, TensorTableEntry> TensorTable;

// Table for storing Tensor metadata on rank zero. This is used for error
// checking, stall checking and size calculations, as well as determining
// when a reduction is ready to be done (when all nodes are ready to do it).
typedef std::unordered_map<
    std::string,
    std::tuple<std::vector<MPIRequest>, std::chrono::system_clock::time_point>>
    MessageTable;

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

  // Only exists on the coordinator node (rank zero). Maintains a count of
  // how many nodes are ready to allreduce every tensor (keyed by tensor
  // name) and time point when tensor started allreduce op.
  std::unique_ptr<MessageTable> message_table;

  // Time point when coordinator last checked for stalled tensors.
  std::chrono::system_clock::time_point last_stall_check;

  // Whether MPI_Init has been completed on the background thread.
  bool initialization_done = false;

  // The MPI rank, local rank, and size.
  int rank = 0;
  int local_rank = 0;
  int size = 1;

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
#if HAVE_NCCL
  std::unordered_map<int, cudaStream_t> streams;
  std::unordered_map<std::vector<int32_t>, ncclComm_t> nccl_comms;
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
static HorovodGlobalState horovod_global;

// For clarify in argument lists.
#define RANK_ZERO 0

// A tag used for all coordinator messaging.
#define TAG_NOTIFY 1

// Stall-check warning time
#define STALL_WARNING_TIME std::chrono::seconds(60)

// Store the MPIRequest for a name, and return whether the total count of
// MPIRequests for that tensor is now equal to the MPI size (and thus we are
// ready to reduce the tensor).
bool IncrementTensorCount(std::unique_ptr<MessageTable>& message_table,
                          MPIRequest msg, int mpi_size) {
  auto name = msg.tensor_name();
  auto table_iter = message_table->find(name);
  if (table_iter == message_table->end()) {
    std::vector<MPIRequest> messages = {msg};
    auto now = std::chrono::system_clock::now();
    message_table->emplace(name, std::make_tuple(std::move(messages), now));
    table_iter = message_table->find(name);
  } else {
    std::vector<MPIRequest>& messages = std::get<0>(table_iter->second);
    messages.push_back(msg);
  }

  std::vector<MPIRequest>& messages = std::get<0>(table_iter->second);
  int count = (int)messages.size();
  return count == mpi_size;
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
    for (auto it = requests[0].tensor_shape().begin();
         it != requests[0].tensor_shape().end(); it++) {
      tensor_shape.AddDim(*it);
    }
    for (unsigned int i = 1; i < requests.size(); i++) {
      if (error) {
        break;
      }

      TensorShape request_shape;
      for (auto it = requests[i].tensor_shape().begin();
           it != requests[i].tensor_shape().end(); it++) {
        request_shape.AddDim(*it);
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
  std::vector<size_t> tensor_sizes(requests.size());
  if (message_type == MPIRequest::ALLGATHER) {
    TensorShape tensor_shape;
    for (auto it = requests[0].tensor_shape().begin();
         it != requests[0].tensor_shape().end(); it++) {
      tensor_shape.AddDim(*it);
    }

    if (tensor_shape.dims() == 0) {
      error = true;
      error_message_stream << "Rank zero tried to "
                           << MPIRequest::RequestType_Name(message_type)
                           << " a rank-zero tensor.";
    } else {
      tensor_sizes[requests[0].request_rank()] =
          size_t(tensor_shape.dim_size(0));
    }

    for (unsigned int i = 1; i < requests.size(); i++) {
      if (error) {
        break;
      }

      TensorShape request_shape;
      for (auto it = requests[i].tensor_shape().begin();
           it != requests[i].tensor_shape().end(); it++) {
        request_shape.AddDim(*it);
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

      tensor_sizes[requests[i].request_rank()] =
          size_t(request_shape.dim_size(0));
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
  for (auto it = requests.begin(); it != requests.end(); it++) {
    devices[it->request_rank()] = it->device();
  }

  MPIResponse response;
  response.set_tensor_name(name);
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

Status GetMPIDataType(const Tensor tensor, MPI_Datatype* dtype) {
  switch (tensor.dtype()) {
  case DT_UINT8:
    *dtype = MPI_UINT8_T;
    return Status::OK();
  case DT_INT8:
    *dtype = MPI_INT8_T;
    return Status::OK();
  case DT_UINT16:
    *dtype = MPI_UINT16_T;
    return Status::OK();
  case DT_INT16:
    *dtype = MPI_INT16_T;
    return Status::OK();
  case DT_INT32:
    *dtype = MPI_INT32_T;
    return Status::OK();
  case DT_INT64:
    *dtype = MPI_INT64_T;
    return Status::OK();
  case DT_FLOAT:
    *dtype = MPI_FLOAT;
    return Status::OK();
  case DT_DOUBLE:
    *dtype = MPI_DOUBLE;
    return Status::OK();
  default:
    // This is not reachable normally since we specify acceptable
    // data types in Op definition.
    return errors::Internal("Invalid tensor type.");
  }
}

#if HAVE_NCCL
Status GetNCCLDataType(const Tensor tensor, ncclDataType_t* dtype) {
  switch (tensor.dtype()) {
  case DT_INT32:
    *dtype = ncclInt32;
    return Status::OK();
  case DT_INT64:
    *dtype = ncclInt64;
    return Status::OK();
  case DT_FLOAT:
    *dtype = ncclFloat32;
    return Status::OK();
  case DT_DOUBLE:
    *dtype = ncclFloat64;
    return Status::OK();
  default:
    // This is not reachable normally since we specify acceptable
    // data types in Op definition.
    return errors::Internal("Invalid tensor type.");
  }
}
#endif

#define MPI_CHECK(e, op_name, op)                                              \
  {                                                                            \
    auto mpi_result = (op);                                                    \
    if (mpi_result != MPI_SUCCESS) {                                           \
      e.callback(                                                              \
          errors::Unknown(op_name, " failed, see MPI output for details."));   \
      return;                                                                  \
    }                                                                          \
  }

#define CUDA_CHECK(e, op_name, op)                                             \
  {                                                                            \
    auto cuda_result = (op);                                                   \
    if (cuda_result != cudaSuccess) {                                          \
      e.callback(errors::Unknown(                                              \
          op_name, " failed: ", cudaGetErrorString(cuda_result)));             \
      return;                                                                  \
    }                                                                          \
  }

#define NCCL_CHECK(e, op_name, op)                                             \
  {                                                                            \
    auto nccl_result = (op);                                                   \
    if (nccl_result != ncclSuccess) {                                          \
      e.callback(errors::Unknown(                                              \
          op_name, " failed: ", ncclGetErrorString(nccl_result)));             \
      return;                                                                  \
    }                                                                          \
  }

// Process an MPIResponse by doing a reduction, a gather, a broadcast, or
// raising an error.
void PerformOperation(TensorTable& tensor_table, MPIResponse response) {
  TensorTableEntry e;
  {
    // Lock on the tensor table.
    std::lock_guard<std::mutex> guard(horovod_global.mutex);

    // We should never fail at finding this key in the tensor table.
    auto name = response.tensor_name();
    auto iter = tensor_table.find(name);
    assert(iter != tensor_table.end());

    assert(response.response_type() == MPIResponse::ALLREDUCE ||
           response.response_type() == MPIResponse::ALLGATHER ||
           response.response_type() == MPIResponse::BROADCAST ||
           response.response_type() == MPIResponse::ERROR);

    e = iter->second;

    // Clear the tensor table of this tensor and its callbacks; the rest of
    // this function takes care of it.
    tensor_table.erase(iter);
  }

#if HAVE_CUDA
  // On GPU data readiness is signalled by ready_event.
  if (e.ready_event != nullptr) {
    while (e.ready_event->PollForStatus() ==
           perftools::gputools::Event::Status::kPending) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(100));
    }
  }
#endif

  Status status;
  if (response.response_type() == MPIResponse::ALLGATHER) {
    // Copy tensor sizes from the MPI response into a vector of size_t
    // and compute total size.  This is size of first dimension.
    std::vector<size_t> tensor_sizes;
    size_t total_dimension_size = 0;
    for (auto it = response.tensor_sizes().begin();
         it != response.tensor_sizes().end(); it++) {
      tensor_sizes.push_back(size_t(*it));
      total_dimension_size += size_t(*it);
    }

    // Every tensor participating in Allgather operation may have different
    // first dimension size, but the rest of dimensions are same for all
    // tensors.  Here we get shape of tensor sliced by first dimension.
    TensorShape single_slice_shape;
    for (int i = 1; i < e.tensor.shape().dims(); i++) {
      single_slice_shape.AddDim(e.tensor.dim_size(i));
    }

    // Allgather output will have shape of:
    // (sum of first dimension of every tensor) x (tensor slice shape).
    TensorShape output_shape;
    output_shape.AddDim((int64)total_dimension_size);
    output_shape.AppendShape(single_slice_shape);

    status = e.context->allocate_output(0, output_shape, &e.output);
    if (!status.ok()) {
      e.callback(status);
      return;
    }

#if HAVE_CUDA
    // On GPU allocation is asynchronous, we need to wait for it to complete.
    auto device_context = e.context->op_device_context();
    if (device_context != nullptr) {
      device_context->stream()->BlockHostUntilDone();
    }
#endif

    MPI_Datatype dtype;
    status = GetMPIDataType(e.tensor, &dtype);
    if (!status.ok()) {
      e.callback(status);
      return;
    }

    // Tensors may have different first dimension, so we need to use
    // MPI_Allgatherv API that supports gathering arrays of different length.
    int* recvcounts = new int[tensor_sizes.size()];
    int* displcmnts = new int[tensor_sizes.size()];
    for (size_t i = 0; i < tensor_sizes.size(); i++) {
      recvcounts[i] =
          (int)(single_slice_shape.num_elements() * tensor_sizes[i]);
      if (i == 0) {
        displcmnts[i] = 0;
      } else {
        displcmnts[i] = recvcounts[i - 1] + displcmnts[i - 1];
      }
    }
    auto result = MPI_Allgatherv((const void*)e.tensor.tensor_data().data(),
                                 (int)e.tensor.NumElements(), dtype,
                                 (void*)e.output->tensor_data().data(),
                                 recvcounts, displcmnts, dtype, MPI_COMM_WORLD);
    delete[] recvcounts;
    delete[] displcmnts;
    MPI_CHECK(e, "MPI_Allgatherv", result)

    e.callback(Status::OK());

  } else if (response.response_type() == MPIResponse::ALLREDUCE) {
#if HOROVOD_GPU_ALLREDUCE == 'N' // 'N' stands for NCCL
    bool on_gpu = e.device != CPU_DEVICE_ID;
    if (on_gpu) {
      CUDA_CHECK(e, "cudaSetDevice", cudaSetDevice(e.device))

      // Ensure stream is in the map before executing reduction.
      cudaStream_t& stream = horovod_global.streams[e.device];
      if (stream == nullptr) {
        CUDA_CHECK(e, "cudaStreamCreate", cudaStreamCreate(&stream))
      }

      // Ensure NCCL communicator is in the map before executing reduction.
      ncclComm_t& nccl_comm = horovod_global.nccl_comms[response.devices()];
      if (nccl_comm == nullptr) {
        ncclUniqueId nccl_id;
        if (horovod_global.rank == 0) {
          NCCL_CHECK(e, "ncclGetUniqueId", ncclGetUniqueId(&nccl_id))
        }

        MPI_CHECK(e, "MPI_Bcast",
                  MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0,
                            MPI_COMM_WORLD));

        NCCL_CHECK(e, "ncclCommInitRank",
                   ncclCommInitRank(&nccl_comm, horovod_global.size, nccl_id,
                                    horovod_global.rank))

        // TODO: Rohit (NVIDIA): figure out why we need this sleep
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }

      ncclDataType_t dtype;
      status = GetNCCLDataType(e.tensor, &dtype);
      if (!status.ok()) {
        e.callback(status);
        return;
      }

      NCCL_CHECK(e, "ncclAllReduce",
                 ncclAllReduce((const void*)e.tensor.tensor_data().data(),
                               (void*)e.output->tensor_data().data(),
                               (size_t)e.tensor.NumElements(), dtype, ncclSum,
                               nccl_comm, stream))

      // Use completion marker via event because it's faster than
      // cudaStreamSynchronize().
      cudaEvent_t event;
      CUDA_CHECK(e, "cudaEventCreateWithFlags",
                 cudaEventCreateWithFlags(&event, cudaEventBlockingSync |
                                                      cudaEventDisableTiming))
      CUDA_CHECK(e, "cudaEventRecord",
                 cudaEventRecord(event, horovod_global.streams[e.device]))

      // TODO: use thread pool or single thread for callbacks
      std::thread finalizer_thread([e, event] {
        CUDA_CHECK(e, "cudaSetDevice", cudaSetDevice(e.device))
        CUDA_CHECK(e, "cudaEventSynchronize", cudaEventSynchronize(event))
        e.callback(Status::OK());
        cudaEventDestroy(event);
      });
      finalizer_thread.detach();
      return;
    }
#endif

    MPI_Datatype dtype;
    status = GetMPIDataType(e.tensor, &dtype);
    if (!status.ok()) {
      e.callback(status);
      return;
    }

    MPI_CHECK(e, "MPI_Allreduce",
              MPI_Allreduce((const void*)e.tensor.tensor_data().data(),
                            (void*)e.output->tensor_data().data(),
                            (int)e.tensor.NumElements(), dtype, MPI_SUM,
                            MPI_COMM_WORLD))

    e.callback(Status::OK());

  } else if (response.response_type() == MPIResponse::BROADCAST) {
    MPI_Datatype dtype;
    status = GetMPIDataType(e.tensor, &dtype);
    if (!status.ok()) {
      e.callback(status);
      return;
    }

    // On root rank, MPI_Bcast sends data, on other ranks it receives data.
    void* data_ptr;
    if (horovod_global.rank == e.root_rank) {
      data_ptr = (void*)e.tensor.tensor_data().data();
    } else {
      data_ptr = (void*)e.output->tensor_data().data();
    }

    MPI_CHECK(e, "MPI_Bcast",
              MPI_Bcast(data_ptr, (int)e.tensor.NumElements(), dtype,
                        e.root_rank, MPI_COMM_WORLD))

    e.callback(Status::OK());

  } else if (response.response_type() == MPIResponse::ERROR) {
    status = errors::FailedPrecondition(response.error_message());
    e.callback(status);
  }
}

// Report Tensors that were submitted to be reduced, gathered or broadcasted by
// some ranks but not others and are waiting for long time to get processed.
void CheckForStalledTensors(HorovodGlobalState& state) {
  bool preamble = false;
  auto now = std::chrono::system_clock::now();
  for (auto it = state.message_table->begin(); it != state.message_table->end();
       it++) {
    auto tensor_name = it->first;
    std::vector<MPIRequest>& messages = std::get<0>(it->second);
    std::chrono::system_clock::time_point start_at = std::get<1>(it->second);

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
                     "submitting tensors, which will cause deadlock. ";
        std::cerr << "Stalled ops: ";
        preamble = true;
      } else {
        std::cerr << ", ";
      }
      std::cerr << tensor_name;
      std::cerr << " [ready ranks:";
      for (auto msg_iter = messages.begin(); msg_iter != messages.end();
           msg_iter++) {
        if (msg_iter == messages.begin()) {
          std::cerr << " ";
        } else {
          std::cerr << ", ";
        }
        std::cerr << msg_iter->request_rank();
      }
      std::cerr << "]";
    }
  }
  if (preamble) {
    std::cerr << std::endl;
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
//
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
void BackgroundThreadLoop(HorovodGlobalState& state) {
  // Initialize MPI. This must happen on the background thread, since not all
  // MPI implementations support being called from multiple threads.
  MPI_Init(NULL, NULL);

  // Get MPI rank to determine if we are rank zero.
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  bool is_coordinator = rank == 0;

  // Get MPI size to determine how many tensors to wait for before reducing.
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Determine local rank by querying the local communicator.
  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &local_comm);
  int local_rank;
  MPI_Comm_rank(local_comm, &local_rank);

  state.rank = rank;
  state.local_rank = local_rank;
  state.size = size;
  state.initialization_done = true;

  // Initialize the tensor count table. No tensors are available yet.
  if (is_coordinator) {
    state.message_table = std::unique_ptr<MessageTable>(new MessageTable());
  }

  // The coordinator sends a SHUTDOWN message to trigger shutdown.
  bool should_shut_down = false;
  do {
    // This delay determines thread frequency and MPI message latency
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

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
    while (!message_queue.empty()) {
      // Pop the first available message message
      MPIRequest message = message_queue.front();
      message_queue.pop();

      if (is_coordinator) {
        bool reduce = IncrementTensorCount(state.message_table, message, size);
        if (reduce) {
          ready_to_reduce.push_back(message.tensor_name());
        }
      } else {
        std::string encoded_message;
        MPIRequest::SerializeToString(message, encoded_message);
        MPI_Send(encoded_message.c_str(), (int)encoded_message.length() + 1,
                 MPI_BYTE, RANK_ZERO, TAG_NOTIFY, MPI_COMM_WORLD);
      }
    }

    // Rank zero has put all its own tensors in the tensor count table.
    // Now, it should count all the tensors that are coming from other
    // ranks at this tick. It should keep getting tensors until it gets a
    // DONE message from all the other ranks.
    if (is_coordinator) {
      // Count of DONE messages. Keep receiving messages until the number
      // of messages is equal to the number of processes. Initialize to
      // one since the coordinator is effectively done.
      int completed_ranks = 1;
      while (completed_ranks != size) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, TAG_NOTIFY, MPI_COMM_WORLD, &status);

        // Find number of characters in message (including zero byte).
        int source_rank = status.MPI_SOURCE;
        int msg_length;
        MPI_Get_count(&status, MPI_BYTE, &msg_length);

        // If the length is zero, this is a DONE message.
        if (msg_length == 0) {
          completed_ranks++;
          MPI_Recv(NULL, 0, MPI_BYTE, source_rank, TAG_NOTIFY, MPI_COMM_WORLD,
                   &status);
          continue;
        }

        // Get tensor name from MPI into an std::string.
        char* buffer = new char[msg_length];
        MPI_Recv(buffer, msg_length, MPI_BYTE, source_rank, TAG_NOTIFY,
                 MPI_COMM_WORLD, &status);
        std::string received_data(buffer, (size_t)msg_length);
        delete[] buffer;

        MPIRequest received_message;
        MPIRequest::ParseFromString(received_message, received_data);
        auto received_name = received_message.tensor_name();

        bool reduce =
            IncrementTensorCount(state.message_table, received_message, size);
        if (reduce) {
          ready_to_reduce.push_back(received_name);
        }
      }

      // At this point, rank zero should have a fully updated tensor count
      // table and should know all the tensors that need to be reduced or
      // gathered, and everyone else should have sent all their information
      // to rank zero. We can now do reductions and gathers; rank zero will
      // choose which ones and in what order, and will notify the other ranks
      // before doing each reduction.
      for (size_t i = 0; i < ready_to_reduce.size(); i++) {
        // Notify all nodes which tensor we'd like to reduce at this step.
        auto name = ready_to_reduce[i];
        MPIResponse response = ConstructMPIResponse(state.message_table, name);

        std::string encoded_response;
        MPIResponse::SerializeToString(response, encoded_response);
        for (int r = 1; r < size; r++) {
          MPI_Send(encoded_response.c_str(), (int)encoded_response.length() + 1,
                   MPI_BYTE, r, TAG_NOTIFY, MPI_COMM_WORLD);
        }

        // Perform the collective operation. All nodes should end up performing
        // the same operation.
        PerformOperation(state.tensor_table, response);
      }

      // Notify all nodes that we are done with the reductions for this tick.
      MPIResponse done_response;
      should_shut_down = state.shut_down;
      done_response.set_response_type(should_shut_down ? MPIResponse::SHUTDOWN
                                                       : MPIResponse::DONE);
      std::string encoded_response;
      MPIResponse::SerializeToString(done_response, encoded_response);
      for (int r = 1; r < size; r++) {
        MPI_Send(encoded_response.c_str(), (int)encoded_response.length() + 1,
                 MPI_BYTE, r, TAG_NOTIFY, MPI_COMM_WORLD);
      }

      // Check for stalled tensors.
      if (std::chrono::system_clock::now() - state.last_stall_check >
          STALL_WARNING_TIME) {
        CheckForStalledTensors(state);
        state.last_stall_check = std::chrono::system_clock::now();
      }
    } else {
      // Notify the coordinator that this node is done sending messages.
      // A DONE message is encoded as a zero-length message.
      MPI_Send(NULL, 0, MPI_BYTE, RANK_ZERO, TAG_NOTIFY, MPI_COMM_WORLD);

      // Receive names for tensors to reduce from rank zero.
      // Once we receive a empty DONE message, stop waiting for more names.
      while (true) {
        MPI_Status status;
        MPI_Probe(0, TAG_NOTIFY, MPI_COMM_WORLD, &status);

        // Find number of characters in message (including zero byte).
        int msg_length;
        MPI_Get_count(&status, MPI_BYTE, &msg_length);

        // Get tensor name from MPI into an std::string.
        char* buffer = new char[msg_length];
        MPI_Recv(buffer, msg_length, MPI_BYTE, 0, TAG_NOTIFY, MPI_COMM_WORLD,
                 &status);
        std::string received_message(buffer, (size_t)msg_length);
        delete[] buffer;

        MPIResponse response;
        MPIResponse::ParseFromString(response, received_message);
        if (response.response_type() == MPIResponse::DONE) {
          // No more messages this tick
          break;
        } else if (response.response_type() == MPIResponse::SHUTDOWN) {
          // No more messages this tick, and the background thread should shut
          // down
          should_shut_down = true;
          break;
        } else {
          // Process the current message
          PerformOperation(state.tensor_table, response);
        }
      }
    }
  } while (!should_shut_down);

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
  MPI_Finalize();
}

// Start Horovod background thread. Ensure that this is
// only done once no matter how many times this function is called.
void InitializeHorovodOnce() {
  // Ensure background thread is only started once.
  if (!horovod_global.initialize_flag.test_and_set()) {
    horovod_global.background_thread =
        std::thread(BackgroundThreadLoop, std::ref(horovod_global));
  }

  // Wait to ensure that the background thread has finished initializing MPI.
  while (!horovod_global.initialization_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

// Check that Horovod is initialized.
Status CheckInitialized() {
  if (!horovod_global.initialization_done) {
    return errors::FailedPrecondition(
        "Horovod has not been initialized; use horovod.tensorflow.init().");
  }
  return Status::OK();
}

// C interface to initialize Horovod.
extern "C" void horovod_tensorflow_init() { InitializeHorovodOnce(); }

// C interface to get index of current Horovod process.
// Returns -1 if Horovod is not initialized.
extern "C" int horovod_tensorflow_rank() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.rank;
}

// C interface to get index of current Horovod process in the node it is on..
// Returns -1 if Horovod is not initialized.
extern "C" int horovod_tensorflow_local_rank() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.local_rank;
}

// C interface to return number of Horovod processes.
// Returns -1 if Horovod is not initialized.
extern "C" int horovod_tensorflow_size() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.size;
}

// Convert a TensorFlow DataType to our MPIDataType.
Status DataTypeToMPIType(DataType tf_dtype, MPIDataType* mpi_dtype) {
  switch (tf_dtype) {
  case DT_UINT8:
    *mpi_dtype = TF_MPI_UINT8;
    return Status::OK();
  case DT_INT8:
    *mpi_dtype = TF_MPI_INT8;
    return Status::OK();
  case DT_UINT16:
    *mpi_dtype = TF_MPI_UINT16;
    return Status::OK();
  case DT_INT16:
    *mpi_dtype = TF_MPI_INT16;
    return Status::OK();
  case DT_INT32:
    *mpi_dtype = TF_MPI_INT32;
    return Status::OK();
  case DT_INT64:
    *mpi_dtype = TF_MPI_INT64;
    return Status::OK();
  case DT_FLOAT:
    *mpi_dtype = TF_MPI_FLOAT32;
    return Status::OK();
  case DT_DOUBLE:
    *mpi_dtype = TF_MPI_FLOAT64;
    return Status::OK();
  default:
    return errors::Internal("Invalid tensor type.");
  }
}

// MPI must be initialized and the background thread must be running before
// this function is called.
void EnqueueTensorAllreduce(OpKernelContext* context, const Tensor& tensor,
                            Tensor* output, GPU_EVENT_IF_CUDA ready_event,
                            const std::string name, const int device,
                            StatusCallback callback) {
  MPIDataType dtype;
  Status status = DataTypeToMPIType(tensor.dtype(), &dtype);
  if (!status.ok()) {
    callback(status);
    return;
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPIRequest message;
  message.set_request_rank(rank);
  message.set_tensor_name(name);
  message.set_tensor_type(dtype);
  message.set_device(device);
  message.set_request_type(MPIRequest::ALLREDUCE);
  for (int i = 0; i < tensor.shape().dims(); i++) {
    message.add_tensor_shape(tensor.shape().dim_size(i));
  }

  TensorTableEntry e;
  e.context = context;
  e.tensor = tensor;
  e.output = output;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  std::lock_guard<std::mutex> guard(horovod_global.mutex);
  horovod_global.tensor_table.emplace(name, std::move(e));
  horovod_global.message_queue.push(message);
}

// MPI must be initialized and the background thread must be running before
// this function is called.
void EnqueueTensorAllgather(OpKernelContext* context, const Tensor& tensor,
                            GPU_EVENT_IF_CUDA ready_event,
                            const std::string name, const int device,
                            StatusCallback callback) {
  MPIDataType dtype;
  Status status = DataTypeToMPIType(tensor.dtype(), &dtype);
  if (!status.ok()) {
    callback(status);
    return;
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPIRequest message;
  message.set_request_rank(rank);
  message.set_tensor_name(name);
  message.set_tensor_type(dtype);
  message.set_device(device);
  message.set_request_type(MPIRequest::ALLGATHER);
  for (int i = 0; i < tensor.shape().dims(); i++) {
    message.add_tensor_shape(tensor.shape().dim_size(i));
  }

  TensorTableEntry e;
  e.context = context;
  e.tensor = tensor;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  std::lock_guard<std::mutex> guard(horovod_global.mutex);
  horovod_global.tensor_table.emplace(name, std::move(e));
  horovod_global.message_queue.push(message);
}

// MPI must be initialized and the background thread must be running before
// this function is called.
void EnqueueTensorBroadcast(OpKernelContext* context, const Tensor& tensor,
                            Tensor* output, int root_rank,
                            GPU_EVENT_IF_CUDA ready_event,
                            const std::string name, const int device,
                            StatusCallback callback) {
  MPIDataType dtype;
  Status status = DataTypeToMPIType(tensor.dtype(), &dtype);
  if (!status.ok()) {
    callback(status);
    return;
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPIRequest message;
  message.set_request_rank(rank);
  message.set_tensor_name(name);
  message.set_tensor_type(dtype);
  message.set_root_rank(root_rank);
  message.set_device(device);
  message.set_request_type(MPIRequest::BROADCAST);
  for (int i = 0; i < tensor.shape().dims(); i++) {
    message.add_tensor_shape(tensor.shape().dim_size(i));
  }

  TensorTableEntry e;
  e.context = context;
  e.tensor = tensor;
  e.output = output;
  e.root_rank = root_rank;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  std::lock_guard<std::mutex> guard(horovod_global.mutex);
  horovod_global.tensor_table.emplace(name, std::move(e));
  horovod_global.message_queue.push(message);
}

int GetDeviceID(OpKernelContext* context) {
  int device = CPU_DEVICE_ID;
  if (context->device() != nullptr &&
      context->device()->tensorflow_gpu_device_info() != nullptr) {
    device = context->device()->tensorflow_gpu_device_info()->gpu_id;
  }
  return device;
}

// On GPU this event will signal that data is ready, and tensors are
// allocated.
GPU_EVENT_IF_CUDA RecordReadyEvent(OpKernelContext* context) {
#if HAVE_CUDA
  auto device_context = context->op_device_context();
  if (device_context != nullptr) {
    auto executor = device_context->stream()->parent();
    GPU_EVENT_IF_CUDA ready_event = new perftools::gputools::Event(executor);
    ready_event->Init();
    device_context->stream()->ThenRecordEvent(ready_event);
    return ready_event;
  }
#endif
  return nullptr;
}

} // namespace tensorflow

class HorovodAllreduceOp : public AsyncOpKernel {
public:
  explicit HorovodAllreduceOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    OP_REQUIRES_OK(context, CheckInitialized());

    auto node_name = name();
    auto device = GetDeviceID(context);
    auto tensor = context->input(0);
    Tensor* output;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, tensor.shape(), &output));
    GPU_EVENT_IF_CUDA ready_event = RecordReadyEvent(context);
    EnqueueTensorAllreduce(context, tensor, output, ready_event, node_name,
                           device, [context, done](const Status& status) {
                             context->SetStatus(status);
                             done();
                           });
  }
};

REGISTER_KERNEL_BUILDER(Name("HorovodAllreduce").Device(DEVICE_CPU),
                        HorovodAllreduceOp);
#if HOROVOD_GPU_ALLREDUCE
REGISTER_KERNEL_BUILDER(Name("HorovodAllreduce").Device(DEVICE_GPU),
                        HorovodAllreduceOp);
#endif

REGISTER_OP("HorovodAllreduce")
    .Attr("T: {int32, int64, float32, float64}")
    .Input("tensor: T")
    .Output("sum: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Perform an MPI Allreduce on a tensor. All other processes that do a reduction
on a tensor with the same name must have the same dimension for that tensor.
Tensors are reduced with other tensors that have the same node name for the
allreduce.

Arguments
    tensor:     A tensor to reduce.

Output
    sum:    A tensor with the same shape as `tensor`, summed across all MPI processes.
)doc");

class HorovodAllgatherOp : public AsyncOpKernel {
public:
  explicit HorovodAllgatherOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    OP_REQUIRES_OK(context, CheckInitialized());

    auto node_name = name();
    auto device = GetDeviceID(context);
    auto tensor = context->input(0);
    // We cannot pre-allocate output for allgather, since shape of result
    // is only known after all ranks make a request.
    GPU_EVENT_IF_CUDA ready_event = RecordReadyEvent(context);
    EnqueueTensorAllgather(context, tensor, ready_event, node_name, device,
                           [context, done](const Status& status) {
                             context->SetStatus(status);
                             done();
                           });
  }
}; // namespace tensorflow

REGISTER_KERNEL_BUILDER(Name("HorovodAllgather").Device(DEVICE_CPU),
                        HorovodAllgatherOp);
#if HOROVOD_GPU_ALLGATHER
REGISTER_KERNEL_BUILDER(Name("HorovodAllgather").Device(DEVICE_GPU),
                        HorovodAllgatherOp);
#endif

REGISTER_OP("HorovodAllgather")
    .Attr("T: {uint8, int8, uint16, int16, int32, int64, float32, float64}")
    .Input("tensor: T")
    .Output("output: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle output;
      TF_RETURN_IF_ERROR(
          c->ReplaceDim(c->input(0), 0, c->UnknownDim(), &output));
      c->set_output(0, output);
      return Status::OK();
    })
    .Doc(R"doc(
Perform an MPI Allgather on a tensor. All other processes that do a gather on a
tensor with the same name must have the same rank for that tensor, and have the
same dimension on all but the first dimension.

Arguments
    tensor:     A tensor to gather.

Output
    gathered:    A tensor with the same shape as `tensor` except for the first dimension.
)doc");

class HorovodBroadcastOp : public AsyncOpKernel {
public:
  explicit HorovodBroadcastOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("root_rank", &root_rank_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    OP_REQUIRES_OK(context, CheckInitialized());

    auto node_name = name();
    auto device = GetDeviceID(context);
    auto tensor = context->input(0);
    Tensor* output = nullptr;
    if (horovod_global.rank == root_rank_) {
      context->set_output(0, tensor);
    } else {
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, tensor.shape(), &output));
    }
    GPU_EVENT_IF_CUDA ready_event = RecordReadyEvent(context);
    EnqueueTensorBroadcast(context, tensor, output, root_rank_, ready_event,
                           node_name, device,
                           [context, done](const Status& status) {
                             context->SetStatus(status);
                             done();
                           });
  }

private:
  int root_rank_;
};

REGISTER_KERNEL_BUILDER(Name("HorovodBroadcast").Device(DEVICE_CPU),
                        HorovodBroadcastOp);
#if HOROVOD_GPU_BROADCAST
REGISTER_KERNEL_BUILDER(Name("HorovodBroadcast").Device(DEVICE_GPU),
                        HorovodBroadcastOp);
#endif

REGISTER_OP("HorovodBroadcast")
    .Attr("T: {uint8, int8, uint16, int16, int32, int64, float32, float64}")
    .Attr("root_rank: int")
    .Input("tensor: T")
    .Output("output: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    })
    .Doc(R"doc(
Perform an MPI Broadcast on a tensor. All other processes that do a broadcast
on a tensor with the same name must have the same dimension for that tensor.

Arguments
    tensor:     A tensor to broadcast.
    root_rank:  Rank that will send data, other ranks will receive data.

Output
    output:    A tensor with the same shape as `tensor` and same value as
               `tensor` on root rank.
)doc");

} // namespace tensorflow
} // namespace horovod
