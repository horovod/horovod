// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
// Modifications copyright (C) 2019 Intel Corporation
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

#ifndef HOROVOD_COMMON_H
#define HOROVOD_COMMON_H

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "message.h"

namespace horovod {
namespace common {

// Activity names, see Horovod Timeline for more details.
#define INIT_FUSION_BUFFER "INIT_FUSION_BUFFER"
#define WAIT_FOR_DATA "WAIT_FOR_DATA"
#define WAIT_FOR_OTHER_TENSOR_DATA "WAIT_FOR_OTHER_TENSOR_DATA"
#define ALLOCATE_OUTPUT "ALLOCATE_OUTPUT"
#define MPI_CROSS_ALLGATHER "MPI_CROSS_ALLGATHER"
#define MPI_ALLGATHER "MPI_ALLGATHER"
#define INIT_NCCL "INIT_NCCL"
#define QUEUE "QUEUE"
#define MEMCPY_IN_FUSION_BUFFER "MEMCPY_IN_FUSION_BUFFER"
#define MEMCPY_IN_HOST_BUFFER "MEMCPY_IN_HOST_BUFFER"
#define MEMCPY_IN_SHARED_BUFFER "MEMCPY_IN_SHARED_BUFFER"
#define MPI_ALLREDUCE "MPI_ALLREDUCE"
#define MPI_ADASUM_ALLREDUCE "MPI_ADASUM_ALLREDUCE"
#define MEMCPY_OUT_HOST_BUFFER "MEMCPY_OUT_HOST_BUFFER"
#define NCCL_ALLREDUCE "NCCL_ALLREDUCE"
#define MEMCPY_OUT_FUSION_BUFFER "MEMCPY_OUT_FUSION_BUFFER"
#define MPI_BCAST "MPI_BCAST"
#define NCCL_REDUCESCATTER "NCCL_REDUCESCATTER"
#define NCCL_ALLGATHER "NCCL_ALLGATHER"
#define NCCL_REDUCE "NCCL_REDUCE"
#define NCCL_BCAST "NCCL_BCAST"
#define COPY_ALLGATHER_OUTPUT "COPY_ALLGATHER_OUTPUT"
#define ALLOCATE_SHARED_BUFFER "ALLOCATE_SHARED_BUFFER"
#define CCL_ALLREDUCE "CCL_ALLREDUCE"
#define CCL_ALLGATHER "CCL_ALLGATHER"
#define CCL_BCAST "CCL_BCAST"
#define GLOO_ALLREDUCE "GLOO_ALLREDUCE"
#define GLOO_ALLGATHER "GLOO_ALLGATHER"
#define GLOO_BCAST "GLOO_BCAST"

// Horovod knobs.
#define HOROVOD_MPI_THREADS_DISABLE "HOROVOD_MPI_THREADS_DISABLE"
#define HOROVOD_TIMELINE "HOROVOD_TIMELINE"
#define HOROVOD_TIMELINE_MARK_CYCLES "HOROVOD_TIMELINE_MARK_CYCLES"
#define HOROVOD_AUTOTUNE "HOROVOD_AUTOTUNE"
#define HOROVOD_AUTOTUNE_LOG "HOROVOD_AUTOTUNE_LOG"
#define HOROVOD_AUTOTUNE_WARMUP_SAMPLES "HOROVOD_AUTOTUNE_WARMUP_SAMPLES"
#define HOROVOD_AUTOTUNE_STEPS_PER_SAMPLE "HOROVOD_AUTOTUNE_STEPS_PER_SAMPLE"
#define HOROVOD_AUTOTUNE_BAYES_OPT_MAX_SAMPLES "HOROVOD_AUTOTUNE_BAYES_OPT_MAX_SAMPLES"
#define HOROVOD_AUTOTUNE_GAUSSIAN_PROCESS_NOISE "HOROVOD_AUTOTUNE_GAUSSIAN_PROCESS_NOISE"
#define HOROVOD_FUSION_THRESHOLD "HOROVOD_FUSION_THRESHOLD"
#define HOROVOD_CYCLE_TIME "HOROVOD_CYCLE_TIME"
#define HOROVOD_STALL_CHECK_DISABLE "HOROVOD_STALL_CHECK_DISABLE"
#define HOROVOD_STALL_CHECK_TIME_SECONDS "HOROVOD_STALL_CHECK_TIME_SECONDS"
#define HOROVOD_STALL_SHUTDOWN_TIME_SECONDS "HOROVOD_STALL_SHUTDOWN_TIME_SECONDS"
#define HOROVOD_HIERARCHICAL_ALLREDUCE "HOROVOD_HIERARCHICAL_ALLREDUCE"
#define HOROVOD_HIERARCHICAL_ALLGATHER "HOROVOD_HIERARCHICAL_ALLGATHER"
#define HOROVOD_CACHE_CAPACITY "HOROVOD_CACHE_CAPACITY"
#define HOROVOD_CCL_BGT_AFFINITY "HOROVOD_CCL_BGT_AFFINITY"
#define HOROVOD_NUM_NCCL_STREAMS "HOROVOD_NUM_NCCL_STREAMS"
#define HOROVOD_CPU_OPERATIONS "HOROVOD_CPU_OPERATIONS"
#define HOROVOD_CONTROLLER "HOROVOD_CONTROLLER"
#define HOROVOD_GLOO_IFACE "HOROVOD_GLOO_IFACE"
#define HOROVOD_MPI "MPI"
#define HOROVOD_CCL "CCL"
#define HOROVOD_GLOO "GLOO"
#define HOROVOD_ADASUM_MPI_CHUNK_SIZE "HOROVOD_ADASUM_MPI_CHUNK_SIZE"
#define HOROVOD_THREAD_AFFINITY "HOROVOD_THREAD_AFFINITY"

// String constant for gloo interface.
#define GLOO_DEFAULT_IFACE ""

// The number of elements held by fusion buffer and hierarchical
// allreduce size is always a multiple of FUSION_BUFFER_ATOMIC_UNIT
#define FUSION_BUFFER_ATOMIC_UNIT 64
#define RANK_ZERO 0

// Device ID used for CPU.
#define CPU_DEVICE_ID (-1)

// Temporary tensor name for ranks that did Join().
#define JOIN_TENSOR_NAME "join.noname"

// List of supported frameworks.
enum Framework { TENSORFLOW, PYTORCH, MXNET };

enum StatusType { OK, UNKNOWN_ERROR, PRECONDITION_ERROR, ABORTED, INVALID_ARGUMENT, IN_PROGRESS };

enum DeviceType { CPU, GPU };

enum Communicator {
  GLOBAL = 0,
  LOCAL = 1,
  CROSS = 2
};

inline std::string CommunicatorName(Communicator comm) {
  switch (comm) {
    case GLOBAL:
      return "global";
    case LOCAL:
      return "local";
    case CROSS:
      return "cross";
    default:
      return "<unknown>";
  }
}

class Status {
public:
  Status();
  static Status OK();
  static Status UnknownError(std::string message);
  static Status PreconditionError(std::string message);
  static Status Aborted(std::string message);
  static Status InvalidArgument(std::string message);
  static Status InProgress();
  bool ok() const;
  bool in_progress() const;
  StatusType type() const;
  const std::string& reason() const;

private:
  StatusType type_ = StatusType::OK;
  std::string reason_ = "";
  Status(StatusType type, std::string reason);
};

// Common error status
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

class TensorShape {
public:
  void AddDim(int64_t dim);
  void AppendShape(TensorShape& other);

  const std::string DebugString() const;
  int dims() const;
  int64_t dim_size(int idx) const;
  int64_t num_elements() const;
  const std::vector<int64_t>& to_vector() const;

  inline bool operator==(const TensorShape& rhs) const {
    return shape_ == rhs.shape_;
  }

  inline bool operator!=(const TensorShape& rhs) const {
    return shape_ != rhs.shape_;
  }

private:
  std::vector<int64_t> shape_;
};

class ReadyEvent {
public:
  virtual bool Ready() const = 0;
  virtual ~ReadyEvent() = default;
};

class OpContext;

class PersistentBuffer {
public:
  virtual const void* AccessData(std::shared_ptr<OpContext> context) const = 0;
  virtual ~PersistentBuffer() = default;
};

class Tensor {
public:
  virtual const DataType dtype() const = 0;
  virtual const TensorShape shape() const = 0;
  virtual const void* data() const = 0;
  virtual int64_t size() const = 0;
  virtual ~Tensor() = default;
};

class OpContext {
public:
  // These allocators are fully synchronous, unlike TensorFlow counterparts.
  virtual Status
  AllocatePersistent(int64_t size,
                     std::shared_ptr<PersistentBuffer>* tensor) = 0;
  virtual Status AllocateOutput(TensorShape shape,
                                std::shared_ptr<Tensor>* tensor) = 0;
  virtual Status AllocateZeros(int64_t num_elements, DataType dtype,
                                std::shared_ptr<Tensor>* tensor) = 0;
  virtual Framework framework() const = 0;
  virtual ~OpContext() = default;
};

// A callback to call after the communication completes. Since the
// allreduce and allgather ops are asynchronous, this callback is what resumes
// computation after the reduction is completed.
using StatusCallback = std::function<void(const Status&)>;

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

// Set affinity function
void server_affinity_set(int affinity);

} // namespace common
} // namespace horovod

#endif // HOROVOD_COMMON_H
