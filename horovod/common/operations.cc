// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
// Modifications copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#define OMPI_SKIP_MPICXX
#include "fusion_buffer_manager.h"
#include "half.h"
#include "hashes.h"
#include "global_state.h"
#include "fusion_buffer_manager.h"
#include "mpi.h"
#include "message.h"
#include "mpi_context.h"
#include "operations.h"
#include "ops/mpi_operations.h"
#include "ops/operation_manager.h"
#include "parameter_manager.h"
#include "timeline.h"
#include "logging.h"

#if HAVE_CUDA
#include "ops/cuda_operations.h"
#include "ops/mpi_cuda_operations.h"
#endif

#if HAVE_NCCL
#include "ops/nccl_operations.h"
#endif

#if HAVE_DDL
#include "ops/ddl_operations.h"
#endif

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

// All the Horovod state that must be stored globally per-process.
HorovodGlobalState horovod_global;

MPIContext mpi_context;

#if HAVE_CUDA
CUDAContext cuda_context;
#endif

#if HAVE_NCCL
NCCLContext nccl_context;
#endif

#if HAVE_DDL
DDLContext ddl_context;
#endif

std::unique_ptr<OperationManager> op_manager;

// For clarify in argument lists.
#define RANK_ZERO 0

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

OperationManager* CreateOperationManager(HorovodGlobalState& state) {
  // Order of these operations is very important. Operations will be checked sequentially from the first
  // to the last. The first 'Enabled' operation will be executed.
  std::vector<std::shared_ptr<AllreduceOp>> allreduce_ops;
  std::vector<std::shared_ptr<AllgatherOp>> allgather_ops;
  std::vector<std::shared_ptr<BroadcastOp>> broadcast_ops;

#if HAVE_CUDA
#if HOROVOD_GPU_ALLREDUCE == 'M'
  allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(new MPI_CUDAAllreduce(&mpi_context, &cuda_context, &state)));

#else
  #if HAVE_NCCL && HOROVOD_GPU_ALLREDUCE == 'N'
    allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(
        new NCCLHierarchicalAllreduce(&nccl_context, &mpi_context, &cuda_context, &state)));
    allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(
        new NCCLAllreduce(&nccl_context, &mpi_context, &cuda_context, &state)));

  #elif HAVE_DDL && HOROVOD_GPU_ALLREDUCE == 'D'
    allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(new DDLAllreduce(&ddl_context, &cuda_context, &state)));
  #endif

  allgather_ops.push_back(std::shared_ptr<AllgatherOp>(new MPIHierarchicalAllgather(&mpi_context, &state)));
#endif
#endif

  // Default operations, always enabled but last to be checked.
  allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(new MPIAllreduce(&mpi_context, &state)));
  allgather_ops.push_back(std::shared_ptr<AllgatherOp>(new MPIAllgather(&mpi_context, &state)));
  broadcast_ops.push_back(std::shared_ptr<BroadcastOp>(new MPIBroadcast(&mpi_context, &state)));
  std::shared_ptr<ErrorOp> error_op(new ErrorOp(&state));

  return new OperationManager(&state.param_manager, allreduce_ops, allgather_ops, broadcast_ops, error_op);
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

// Return the total byte size of the final allgathered output tensor
int64_t TotalByteSizeOfAllgatherOutput(const std::vector<int64_t> &tensor_sizes,
                                       const TensorTableEntry entry, MPIContext& ctx) {
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
  int element_size = ctx.GetMPITypeSize(entry.tensor->dtype());
  int64_t total_byte_size_of_output =
      total_count_of_output_entries * element_size;

  return total_byte_size_of_output;
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

// Populates provided ResponseList with responses from deque.
ResponseList FuseResponses(std::deque<Response>& responses,
                           HorovodGlobalState& state, MPIContext& ctx) {
  ResponseList response_list;
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
          auto& new_entry = state.tensor_table[new_response.tensor_names()[0]];
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
            // In general, don't try to fuse additional tensors since they are
            // usually computed in order of requests and skipping tensors may mean
            // that the batch will have to wait longer while skipped tensors
            // could be reduced at that time. However, mixed-precision training
            // may yield requests of various dtype in a mixed-up sequence causing
            // breakups in fusion. To counter this some look ahead is allowed.

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
            TotalByteSizeOfAllgatherOutput(response.tensor_sizes(), entry, ctx);

        std::deque<Response> skipped_responses;
        int64_t skipped_size = 0;
        while (!responses.empty()) {

          auto new_response = responses.front();
          assert(new_response.tensor_names().size() == 1);
          auto& new_entry = state.tensor_table[new_response.tensor_names()[0]];

          int64_t new_total_byte_size_of_output =
              TotalByteSizeOfAllgatherOutput(new_response.tensor_sizes(),
                                             new_entry, ctx);

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
            // In general, don't try to fuse additional tensors since they are
            // usually computed in order of requests and skipping tensors may mean
            // that the batch will have to wait longer while skipped tensors
            // could be reduced at that time. However, mixed-precision training
            // may yield requests of various dtype in a mixed-up sequence causing
            // breakups in fusion. To counter this some look ahead is allowed.

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

  return response_list;
}

// Helper function to get list of allreduced tensor names and total size for
// use with the autotuner.
int64_t GetTensorDataForAutotuner(const ResponseList& response_list,
                                  const TensorTable& tensor_table,
                                  std::vector<std::string>& tensor_names) {
  int64_t total_tensor_size = 0;
  for (auto& response : response_list.responses()) {
    if (response.response_type() == Response::ResponseType::ALLREDUCE) {
      for (auto& tensor_name : response.tensor_names()) {
        tensor_names.push_back(tensor_name);
        auto& entry = tensor_table.at(tensor_name);
        total_tensor_size += entry.tensor->size();
      }
    }
  }
  return total_tensor_size;
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
        TensorFusionThresholdBytes(),
        first_entry.device, first_entry.context,
        [&](){timeline.ActivityStartAll(entries, INIT_FUSION_BUFFER);},
        [&](){timeline.ActivityEndAll(entries);});
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
  try {
    status = op_manager->ExecuteOperation(entries, response);
  } catch (const std::exception& ex) {
    status = Status::UnknownError(ex.what());
  }

  if (!status.in_progress()) {
    for (auto& e : entries) {
      timeline.End(e.tensor_name, status.ok() ? e.output : nullptr);
      e.callback(status);
    }
  }

}

// Report Tensors that were submitted to be reduced, gathered or broadcasted by
// some ranks but not others and are waiting for long time to get processed.
bool CheckForStalledTensors(HorovodGlobalState& state) {
  bool should_shut_down = false;
  auto now = std::chrono::steady_clock::now();
  std::map<int32_t, std::set<std::string>> missing_ranks;
  std::unordered_set<int32_t> shutdown_ranks;
  std::chrono::seconds stall_warning_time(state.stall_warning_time_seconds);
  std::chrono::seconds stall_shutdown_time(state.stall_shutdown_time_seconds);

  if (stall_shutdown_time > std::chrono::seconds(0) &&
    stall_shutdown_time < stall_warning_time) {
    LOG(WARNING) << "HOROVOD_STALL_SHUTDOWN_TIME_SECONDS is less than HOROVOD_STALL_CHECK_TIME_SECONDS, will not shutdown.";
    stall_shutdown_time = std::chrono::seconds(0);
  }

  for (auto& m : *state.message_table) {
    auto tensor_name = m.first;
    std::vector<Request>& messages = std::get<0>(m.second);
    std::chrono::steady_clock::time_point start_at = std::get<1>(m.second);
    auto lag = now - start_at;

    if (lag > stall_warning_time) {
      std::unordered_set<int32_t> ready_ranks;
      for (auto msg_iter = messages.begin(); msg_iter != messages.end();
           ++msg_iter) {
        ready_ranks.insert(msg_iter->request_rank());
      }

      for (int32_t rank = 0; rank < state.size; ++rank) {
        if (ready_ranks.find(rank) == ready_ranks.end()) {
          missing_ranks[rank].insert(tensor_name);
          if (stall_shutdown_time > std::chrono::seconds(0) && lag > stall_shutdown_time) {
            shutdown_ranks.insert(rank);
            should_shut_down = true;
          }
        }
      }
    }
  }

  if (!missing_ranks.empty()) {
    std::stringstream message;
    message << "One or more tensors were submitted to be "
               "reduced, gathered or broadcasted by subset of ranks and "
               "are waiting for remainder of ranks for more than "
            << stall_warning_time.count() << " seconds. "
            << "This may indicate that different ranks are trying to "
               "submit different tensors or that only subset of ranks is "
               "submitting tensors, which will cause deadlock. "
            << std::endl << "Stalled ranks:";
    for (auto& kv: missing_ranks) {
      message << std::endl << kv.first;
      if (shutdown_ranks.find(kv.first) != shutdown_ranks.end()) {
        message << "!";
      }

      message << ": [";
      auto it = kv.second.begin();
      message << *it;
      int count = 0;
      while (++it != kv.second.end()) {
        message << ", " << *it;
        if (++count == 5) {
          message << " ...";
          break;
        }
      }

      message << "]";
    }

    if (should_shut_down) {
      message << std::endl
              << "One or more rank (marked by \"!\") is stalled for longer than "
              << stall_shutdown_time.count() << " seconds. Will shutdown.";
      LOG(ERROR) << message.str();
    } else {
      LOG(WARNING) << message.str();
    }
  }

  return should_shut_down;
}

// Invalidate cached tensors that have been pending for a long time.
void InvalidateStalledCachedTensors(HorovodGlobalState& state,
                                  CacheCoordinator& cache_coordinator) {
  auto now = std::chrono::steady_clock::now();
  std::chrono::seconds stall_warning_time(state.stall_warning_time_seconds);

  for (auto& entry : state.cache_tensor_start) {
    // If pending time for cached tensor exceeds stall_warning_time, mark entry
    // for global removal from cache to trigger stall messaging.
    if (now - entry.second > stall_warning_time) {
       uint32_t cache_bit = state.response_cache.peek_cache_bit(entry.first);
       cache_coordinator.record_invalid_bit(cache_bit);
       cache_coordinator.set_uncached_in_queue(true);
    }
  }
}

void set_bool_from_env(const char* env, bool& val, bool value_if_set) {
  auto env_value = std::getenv(env);
  if (env_value != nullptr &&
      std::strtol(env_value, nullptr, 10) > 0) {
    val = value_if_set;
  }
}

void set_int_from_env(const char* env, int& val) {
  auto env_value = std::getenv(env);
  if (env_value != nullptr) {
    val = std::strtol(env_value, nullptr, 10);
  }
}

// Routine to sync cache hit and invalid bit sets across workers.
// Also determines global shutdown state and whether uncached requests
// exist on any worker.
void CoordinateCacheAndState(CacheCoordinator& cache_coordinator,
                             HorovodGlobalState& state,
                             MPIContext& ctx) {

  // Sync cache and state information across workers.
  cache_coordinator.sync(ctx, state.timeline_enabled);

  // If invalid cache entries exist, erase associated entries.
  if (!cache_coordinator.invalid_bits().empty()) {
    for (auto bit : cache_coordinator.invalid_bits()) {
      state.response_cache.erase_response(bit);
    }
  }

  if (state.timeline_enabled) {
    // Start/continue negotiation phase on timeline bit entries.
    for (auto bit : cache_coordinator.timeline_bits()) {
      auto response = state.response_cache.peek_response(bit);
      state.timeline.NegotiateStart(response.tensor_names()[0],
          (Request::RequestType)response.response_type());
    }

    // End negotation phase for synced cache hit set entries.
    for (auto bit : cache_coordinator.cache_hits()) {
      auto response = state.response_cache.peek_response(bit);
      state.timeline.NegotiateEnd(response.tensor_names()[0]);
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
bool RunLoopOnce(HorovodGlobalState& state, MPIContext& ctx, bool is_coordinator);
void BackgroundThreadLoop(HorovodGlobalState& state, MPIContext& ctx) {
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
    MPI_Comm_create_group(MPI_COMM_WORLD, work_group, 0, &(ctx.mpi_comm));
    if (ctx.mpi_comm == MPI_COMM_NULL) {
      LOG(WARNING) << "Unable to create Horovod communicator, using "
                      "MPI_COMM_WORLD instead.";
      ctx.mpi_comm = MPI_COMM_WORLD;
    }
    MPI_Group_free(&world_group);
    MPI_Group_free(&work_group);
  } else if (!ctx.mpi_comm) {
    // No ranks were given and no communicator provided to horovod_init() so use
    // MPI_COMM_WORLD
    MPI_Comm_dup(MPI_COMM_WORLD, &(ctx.mpi_comm));
  }

  // Get MPI rank to determine if we are rank zero.
  int rank;
  MPI_Comm_rank(ctx.mpi_comm, &rank);
  bool is_coordinator = rank == 0;

  // Get MPI size to determine how many tensors to wait for before reducing.
  int size;
  MPI_Comm_size(ctx.mpi_comm, &size);
  if (is_coordinator) {
    LOG(INFO) << "Started Horovod with " << size << " processes";
  }

  // Determine local rank by querying the local communicator.
  MPI_Comm local_comm;
  MPI_Comm_split_type(ctx.mpi_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
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
                ctx.mpi_comm);

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
  MPI_Comm_split(ctx.mpi_comm, local_rank, rank, &cross_comm);
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
  ctx.local_comm = local_comm;
  ctx.cross_comm = cross_comm;
  ctx.mpi_float16_t = mpi_float16_t;
  ctx.mpi_float16_sum = mpi_float16_sum;
  state.mpi_threads_supported = (provided == MPI_THREAD_MULTIPLE);
  state.local_comm_ranks = local_comm_ranks;

  // Open the timeline file on coordinator.
  auto horovod_timeline = std::getenv(HOROVOD_TIMELINE);
  if (is_coordinator && horovod_timeline != nullptr) {
    state.timeline.Initialize(std::string(horovod_timeline),
                              static_cast<unsigned int>(size));
  }
  if (horovod_timeline != nullptr) {
    state.timeline_enabled = true;
  }

  set_bool_from_env(HOROVOD_TIMELINE_MARK_CYCLES, state.mark_cycles_in_timeline, true);

  set_bool_from_env(HOROVOD_STALL_CHECK_DISABLE, state.perform_stall_check, false);

  set_int_from_env(HOROVOD_STALL_CHECK_TIME_SECONDS, state.stall_warning_time_seconds);

  set_int_from_env(HOROVOD_STALL_SHUTDOWN_TIME_SECONDS, state.stall_shutdown_time_seconds);

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

  // Override response cache capacity, if it's set.
  state.param_manager.SetCacheEnabled(true);
  auto horovod_cache_capacity = std::getenv(HOROVOD_CACHE_CAPACITY);
  if (horovod_cache_capacity != nullptr) {
    uint32_t cache_capacity = std::strtol(horovod_cache_capacity, nullptr, 10);
    state.cache_capacity = cache_capacity;
    state.param_manager.SetCacheEnabled((cache_capacity > 0) ? true : false, true);
  }
  state.response_cache.set_capacity((int)state.param_manager.CacheEnabled() *
                                    state.cache_capacity);

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
    state.param_manager.Initialize(rank, RANK_ZERO, ctx.mpi_comm,
                                   horovod_autotune_log != nullptr
                                       ? std::string(horovod_autotune_log)
                                       : "");
    state.param_manager.SetAutoTuning(true);
  }

  // Initialize the tensor count table. No tensors are available yet.
  if (is_coordinator) {
    state.message_table = std::unique_ptr<MessageTable>(new MessageTable());
  }

  op_manager.reset(CreateOperationManager(state));

  // Signal that initialization is completed.
  state.initialization_done = true;

  LOG(INFO, rank) << "Horovod Initialized";

  // Iterate until shutdown.
  while (RunLoopOnce(state, ctx, is_coordinator))
    ;

  LOG(DEBUG, rank) << "Shutting down background thread";

  // Signal that shutdown has been requested.
  state.shut_down = true;

#if HAVE_NCCL
  nccl_context.ShutDown();
#endif

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
    MPI_Win_free(&ctx.window);
    horovod_global.shared_buffer = nullptr;
  }

  if (ctx.mpi_comm != MPI_COMM_NULL &&
      ctx.mpi_comm != MPI_COMM_WORLD) {
    MPI_Comm_free(&ctx.mpi_comm);
  }

  if (ctx.local_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&ctx.local_comm);
  }

  if (ctx.cross_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&ctx.cross_comm);
  }

  if (ctx.mpi_float16_t != MPI_DATATYPE_NULL) {
    MPI_Type_free(&ctx.mpi_float16_t);
  }

  if (ctx.mpi_float16_sum != MPI_OP_NULL) {
    MPI_Op_free(&ctx.mpi_float16_sum);
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

// If all messages in queue have responses in cache, use fast path with
// no additional MPI coordination.
void RunBypass(std::queue<Request>& message_queue, CacheCoordinator& cache_coordinator,
               HorovodGlobalState& state, MPIContext& ctx) {

  // Convert cache hits to responses. Populate so that least
  // recently used responses get priority. All workers call the code
  // here so we use the get method here to consistently update the cache
  // order.
  std::deque<Response> responses;
  for (auto bit : cache_coordinator.cache_hits()) {
    responses.push_back(state.response_cache.get_response(bit));
  }

  // Fuse responses as normal.
  auto response_list = FuseResponses(responses, state, ctx);

  if (!response_list.responses().empty()) {
    std::string tensors_ready;
    for (auto r : response_list.responses()) {
      tensors_ready += r.tensor_names_string() + "; " ;
    }
    LOG(TRACE) << "Sending ready responses as " << tensors_ready;
  }

  // Get tensor name and size data for autotuning.
  int64_t total_tensor_size;
  std::vector<std::string> tensor_names;
  if (state.param_manager.IsAutoTuning()) {
    std::lock_guard<std::mutex> guard(state.mutex);
    total_tensor_size = GetTensorDataForAutotuner(response_list, state.tensor_table,
                                                  tensor_names);
  }

  // Perform the collective operation. All nodes should end up performing
  // the same operation.
  for (auto& response : response_list.responses()) {
    LOG(TRACE, state.rank) << "Performing " << response.tensor_names_string();
    LOG(DEBUG, state.rank) << "Processing " << response.tensor_names().size() << " tensors";
    PerformOperation(state.tensor_table, response);
    LOG(TRACE, state.rank) << "Finished performing " << response.tensor_names_string();
  }

  // Reassign cache bits based on current cache order.
  state.response_cache.update_cache_bits();

  if (state.param_manager.IsAutoTuning()) {
    state.param_manager.Update(tensor_names, total_tensor_size);
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
bool RunLoopOnce(HorovodGlobalState& state, MPIContext& ctx, bool is_coordinator) {
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

  // Update cache capacity if autotuning is active.
  if (state.param_manager.IsAutoTuning()) {
    state.response_cache.set_capacity((int)state.param_manager.CacheEnabled() *
                                      state.cache_capacity);
  }

  // Copy the data structures from global state under this lock.
  // However, don't keep the lock for the rest of the loop, so that
  // enqueued stream callbacks can continue.

  CacheCoordinator cache_coordinator(state.response_cache.num_active_bits());

  std::queue<Request> message_queue;
  {
    std::lock_guard<std::mutex> guard(state.mutex);
    while (!state.message_queue.empty()) {
      Request message = state.message_queue.front();
      state.message_queue.pop();
      message_queue.push(message);

      // Keep track of cache hits
      if (state.response_cache.capacity() > 0) {
        auto cache_state = state.response_cache.cached(message);
        if (cache_state == ResponseCache::CacheState::HIT) {
          uint32_t cache_bit = state.response_cache.peek_cache_bit(message);
          cache_coordinator.record_hit(cache_bit);

          // Record initial time cached tensor is encountered in queue.
          if (state.perform_stall_check &&
              state.cache_tensor_start.find(message.tensor_name()) == state.cache_tensor_start.end()) {
            state.cache_tensor_start[message.tensor_name()] = std::chrono::steady_clock::now();
          }

        } else {
          if (cache_state == ResponseCache::CacheState::INVALID) {
            uint32_t cache_bit = state.response_cache.peek_cache_bit(message);
            cache_coordinator.record_invalid_bit(cache_bit);
          }
          cache_coordinator.set_uncached_in_queue(true);

          // Remove timing entry if uncached or marked invalid.
          if (state.perform_stall_check) {
            state.cache_tensor_start.erase(message.tensor_name());
          }

        }
      }
    }
  }

  // Flag indicating that the background thread should shut down.
  bool should_shut_down = state.shut_down;

  // Check for stalled tensors.
  if (state.perform_stall_check &&
       std::chrono::steady_clock::now() - state.last_stall_check >
           std::chrono::seconds(state.stall_warning_time_seconds)) {
    if (is_coordinator) {
      should_shut_down |= CheckForStalledTensors(state);
    }

    if (state.response_cache.capacity() > 0) {
      InvalidateStalledCachedTensors(state, cache_coordinator);
    }
    state.last_stall_check = std::chrono::steady_clock::now();
  }

  cache_coordinator.set_should_shut_down(should_shut_down);

  if (state.response_cache.capacity() > 0) {
    // Obtain common cache hits and cache invalidations across workers. Also,
    // determine if any worker has uncached messages in queue or requests
    // a shutdown. This function removes any invalid cache entries, if they
    // exist.
    CoordinateCacheAndState(cache_coordinator, state, ctx);

    {
      // Get lock in order to safely replace messages to global queue
      std::lock_guard<std::mutex> guard(state.mutex);

      // Remove uncommon cached tensors from queue and replace to state
      // queue for next cycle. Skip adding common cached tensors to
      // queue as they are handled separately.
      size_t num_messages = message_queue.size();
      for (size_t i = 0; i < num_messages; ++i) {
        auto message = message_queue.front();
        if (state.response_cache.cached(message) == ResponseCache::CacheState::HIT) {
          uint32_t cache_bit = state.response_cache.peek_cache_bit(message);
          if (cache_coordinator.cache_hits().find(cache_bit) ==
              cache_coordinator.cache_hits().end()) {
            // Try to process again in next cycle.
            state.message_queue.push(std::move(message));
          } else if (state.perform_stall_check) {
            // Remove timing entry for messages being handled this cycle.
            state.cache_tensor_start.erase(message.tensor_name());
          }
        } else {
          // Remove timing entry for messages being handled this cycle.
          if (state.perform_stall_check) {
            state.cache_tensor_start.erase(message.tensor_name());
          }
          message_queue.push(std::move(message));
        }
        message_queue.pop();
      }
    }
  }

  if (!message_queue.empty()) {
    LOG(DEBUG, state.rank) << "Sent " << message_queue.size() << " messages";
  }

  if (state.response_cache.capacity() > 0 && !cache_coordinator.uncached_in_queue()) {
    // If only cached messages in queue, use fast coordination path.
    if (!cache_coordinator.cache_hits().empty()) {
      RunBypass(message_queue, cache_coordinator, state, ctx);
    }
    return !cache_coordinator.should_shut_down();
  }

  // Collect all tensors that are ready to be reduced. Record them in the
  // tensor count table (rank zero) or send them to rank zero to be
  // recorded (everyone else).
  std::vector<std::string> ready_to_reduce;
  ResponseList response_list;
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
               ctx.mpi_comm);

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
                RANK_ZERO, ctx.mpi_comm);

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

    if (state.response_cache.capacity() > 0) {
      // Prepopulate response list with cached responses. Populate so that
      // least recently used responses get priority. Since only the coordinator
      // rank calls this code, use peek instead of get here to preserve cache
      // order across workers.
      for (auto bit : cache_coordinator.cache_hits()) {
        responses.push_back(state.response_cache.peek_response(bit));
      }
    }

    for (auto& tensor_name : ready_to_reduce) {
      Response response =
          ConstructResponse(state.message_table, tensor_name);
      responses.push_back(std::move(response));
    }

    response_list = FuseResponses(responses, state, ctx);
    response_list.set_shutdown(should_shut_down);

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
    MPI_Bcast(&encoded_response_length, 1, MPI_INT, RANK_ZERO, ctx.mpi_comm);
    MPI_Bcast((void*)encoded_response.c_str(), encoded_response_length,
              MPI_BYTE, RANK_ZERO, ctx.mpi_comm);

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
               RANK_ZERO, ctx.mpi_comm);
    MPI_Gatherv((void*)encoded_message.c_str(), encoded_message_length,
                MPI_BYTE, nullptr, nullptr, nullptr, MPI_BYTE, RANK_ZERO,
                ctx.mpi_comm);

    int msg_length;
    MPI_Bcast(&msg_length, 1, MPI_INT, RANK_ZERO, ctx.mpi_comm);
    auto buffer = new uint8_t[msg_length];
    MPI_Bcast(buffer, msg_length, MPI_BYTE, RANK_ZERO, ctx.mpi_comm);
    ResponseList::ParseFromBytes(response_list, buffer);
    delete[] buffer;
  }

  // Get tensor name and size data for autotuning.
  int64_t total_tensor_size;
  std::vector<std::string> tensor_names;
  if (state.param_manager.IsAutoTuning()) {
    std::lock_guard<std::mutex> guard(state.mutex);
    total_tensor_size = GetTensorDataForAutotuner(response_list, state.tensor_table,
                                                  tensor_names);
  }

  if (state.response_cache.capacity() > 0) {
    std::lock_guard<std::mutex> guard(horovod_global.mutex);
    // All workers add supported responses to cache. This updates the cache order
    // consistently across workers.
    for (auto& response : response_list.responses()) {
      if (response.response_type() == Response::ResponseType::ALLREDUCE &&
          (int)response.devices().size() == state.size) {
        state.response_cache.put(response, state.tensor_table);
      }
    }

    // Reassign cache bits based on current cache order.
    state.response_cache.update_cache_bits();
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
        std::thread(BackgroundThreadLoop, std::ref(horovod_global), std::ref(mpi_context));
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
  MPI_Comm_dup(comm, &(mpi_context.mpi_comm));
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
