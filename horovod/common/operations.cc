// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
// Modifications copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.
// Modifications copyright (C) 2019 Intel Corporation
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

#include "operations.h"

#include <atomic>
#include <cassert>
#include <cstring>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "common.h"
#include "fusion_buffer_manager.h"
#include "global_state.h"
#include "hashes.h"
#include "logging.h"
#include "message.h"
#include "nvtx_op_range.h"
#include "ops/operation_manager.h"
#include "parameter_manager.h"
#include "timeline.h"
#include "utils/env_parser.h"

#if HAVE_MPI
#define OMPI_SKIP_MPICXX
#include "mpi.h"
#include "mpi/mpi_context.h"
#include "mpi/mpi_controller.h"
#include "ops/adasum_mpi_operations.h"
#include "ops/mpi_operations.h"
#endif

#if HAVE_GPU
#include "ops/gpu_operations.h"
#if HAVE_MPI
#include "ops/mpi_gpu_operations.h"
#endif
#endif

#if HAVE_NCCL
#include "ops/nccl_operations.h"
#if HAVE_MPI
#include "ops/adasum_gpu_operations.h"
#endif
#endif

#if HAVE_DDL && HAVE_MPI
#include "mpi/ddl_mpi_context_manager.h"
#include "ops/ddl_operations.h"
#endif

#if HAVE_CCL
#include "ops/ccl_operations.h"
#endif

#if HAVE_GLOO
#include "gloo/gloo_controller.h"
#include "ops/gloo_operations.h"
#endif

/*
 * Allreduce, Allgather and Broadcast Ops.
 *
 * This module implements ops for allgather, allreduce and broadcast, which
 * do optimized gathers, reductions and broadcasts and can take advantage of
 * whichever hardware-optimized communication libraries are enabled.
 *
 * The primary logic of the allreduce, allgather and broadcast currently
 * support in MPI, NCCL, CUDA/ROCm, Gloo, oneCCL, DDL. The background thread
 * which facilitates controller operations is run in BackgroundThreadLoop().
 * The provided ops are:
 *      - HorovodAllreduce:
 *          Perform an allreduce on a Tensor, returning the sum
 *          across all processes in the global communicator.
 *      - HorovodAllgather:
 *          Perform an allgather on a Tensor, returning the concatenation of
 *          the tensor on the first dimension across all processes in the
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

const Status NOT_INITIALIZED_ERROR = Status::PreconditionError(
    "Horovod has not been initialized; use hvd.init().");

const Status SHUT_DOWN_ERROR = Status::UnknownError(
    "Horovod has been shut down. This was caused by an exception on one of the "
    "ranks or an attempt to allreduce, allgather or broadcast a tensor after "
    "one of the ranks finished execution. If the shutdown was caused by an "
    "exception, you should see the exception in the log before the first "
    "shutdown message.");

// All the Horovod state that must be stored globally per-process.
HorovodGlobalState horovod_global;

#if HAVE_MPI
MPIContext global_mpi_context;
#endif

#if HAVE_GLOO
GlooContext global_gloo_context;
#endif

#if HAVE_GPU
GPUContext gpu_context;
#endif

#if HAVE_NCCL
NCCLContext nccl_context;
NCCLContext local_nccl_context;
NCCLContext cross_nccl_context;
#endif

#if HAVE_DDL
DDLContext ddl_context;
#endif

#if HAVE_CCL
CCLContext ccl_context;
#endif

std::unique_ptr<OperationManager> op_manager;

OperationManager* CreateOperationManager(HorovodGlobalState& state) {
  // Order of these operations is very important. Operations will be checked
  // sequentially from the first to the last. The first 'Enabled' operation will
  // be executed.
  std::vector<std::shared_ptr<AllreduceOp>> allreduce_ops;
  std::vector<std::shared_ptr<AllgatherOp>> allgather_ops;
  std::vector<std::shared_ptr<BroadcastOp>> broadcast_ops;
  std::vector<std::shared_ptr<ReducescatterOp>> reducescatter_ops;
  std::vector<std::shared_ptr<AllreduceOp>> adasum_ops;
  std::vector<std::shared_ptr<AlltoallOp>> alltoall_ops;

#if HAVE_MPI && HAVE_GPU
  if (global_mpi_context.IsEnabled()) {
#if HOROVOD_GPU_ALLREDUCE == 'M'
    allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(
        new MPI_GPUAllreduce(&gpu_context, &state)));

#elif HAVE_NCCL && HOROVOD_GPU_ALLREDUCE == 'N'
    adasum_ops.push_back(std::shared_ptr<AllreduceOp>(new AdasumGpuAllreduceOp(
        &global_mpi_context, &nccl_context, &gpu_context, &state)));

    allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(
        new NCCLHierarchicalAllreduce(&nccl_context, &gpu_context, &state)));

#elif HAVE_DDL && HOROVOD_GPU_ALLREDUCE == 'D'
    allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(
        new DDLAllreduce(&ddl_context, &gpu_context, &state)));
#endif

#if HOROVOD_GPU_ALLGATHER == 'M'
    allgather_ops.push_back(std::shared_ptr<AllgatherOp>(
        new MPI_GPUAllgather(&gpu_context, &state)));
#endif
    allgather_ops.push_back(
        std::shared_ptr<AllgatherOp>(new MPIHierarchicalAllgather(&state)));

#if HOROVOD_GPU_ALLTOALL == 'M'
    alltoall_ops.push_back(
        std::shared_ptr<AlltoallOp>(new MPI_GPUAlltoall(&gpu_context, &state)));
#endif

#if HOROVOD_GPU_REDUCESCATTER == 'M'
    reducescatter_ops.push_back(std::shared_ptr<ReducescatterOp>(
        new MPI_GPUReducescatter(&gpu_context, &state)));
#endif
  }
#endif

#if HAVE_NCCL && HOROVOD_GPU_ALLREDUCE == 'N'
  allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(
    new NCCLTorusAllreduce(&local_nccl_context, &cross_nccl_context, &gpu_context, &state)));
  allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(
      new NCCLAllreduce(&nccl_context, &gpu_context, &state)));
#endif

#if HAVE_NCCL && HOROVOD_GPU_BROADCAST == 'N'
  broadcast_ops.push_back(std::shared_ptr<BroadcastOp>(
      new NCCLBroadcast(&nccl_context, &gpu_context, &state)));
#endif

#if HAVE_NCCL && HOROVOD_GPU_ALLGATHER == 'N'
  allgather_ops.push_back(std::shared_ptr<AllgatherOp>(
      new NCCLAllgather(&nccl_context, &gpu_context, &state)));
#endif

#if HAVE_NCCL && HOROVOD_GPU_REDUCESCATTER == 'N'
    reducescatter_ops.push_back(std::shared_ptr<ReducescatterOp>(
        new NCCLReducescatter(&nccl_context, &gpu_context, &state)));
#endif

#if HAVE_NCCL && HOROVOD_GPU_ALLTOALL == 'N'
  alltoall_ops.push_back(std::shared_ptr<AlltoallOp>(
      new NCCLAlltoall(&nccl_context, &gpu_context, &state)));
#endif

#if HAVE_GLOO
  if (global_gloo_context.IsEnabled()) {
    allreduce_ops.push_back(
        std::shared_ptr<AllreduceOp>(new GlooAllreduce(&state)));
    allgather_ops.push_back(
        std::shared_ptr<AllgatherOp>(new GlooAllgather(&state)));
    broadcast_ops.push_back(
        std::shared_ptr<BroadcastOp>(new GlooBroadcast(&state)));
    alltoall_ops.push_back(
        std::shared_ptr<AlltoallOp>(new GlooAlltoall(&state)));
    reducescatter_ops.push_back(
        std::shared_ptr<ReducescatterOp>(new GlooReducescatter(&state)));
  }
#endif

#if HAVE_CCL
  if (state.cpu_operation == LibType::CCL) {
    allreduce_ops.push_back(
        std::make_shared<CCLAllreduce>(&ccl_context, &state));
    allgather_ops.push_back(
        std::make_shared<CCLAllgather>(&ccl_context, &state));
    broadcast_ops.push_back(
        std::make_shared<CCLBroadcast>(&ccl_context, &state));
    alltoall_ops.push_back(std::make_shared<CCLAlltoall>(&ccl_context, &state));
  }
#endif

#if HAVE_MPI
  if (global_mpi_context.IsEnabled()) {
    adasum_ops.push_back(std::shared_ptr<AllreduceOp>(
        new AdasumMPIAllreduceOp(&global_mpi_context, &state)));
    allreduce_ops.push_back(
        std::shared_ptr<AllreduceOp>(new MPIAllreduce(&state)));
    allgather_ops.push_back(
        std::shared_ptr<AllgatherOp>(new MPIAllgather(&state)));
    broadcast_ops.push_back(
        std::shared_ptr<BroadcastOp>(new MPIBroadcast(&state)));
    alltoall_ops.push_back(
        std::shared_ptr<AlltoallOp>(new MPIAlltoall(&state)));
    reducescatter_ops.push_back(
        std::shared_ptr<ReducescatterOp>(new MPIReducescatter(&state)));
  }
#endif

  std::shared_ptr<JoinOp> join_op(new JoinOp(&state));
  std::shared_ptr<BarrierOp> barrier_op(new BarrierOp(&state));
  std::shared_ptr<ErrorOp> error_op(new ErrorOp(&state));

  return new OperationManager(&state.parameter_manager, allreduce_ops,
                              allgather_ops, broadcast_ops, alltoall_ops,
                              reducescatter_ops, join_op, adasum_ops,
                              barrier_op, error_op);
}

// Process a Response by doing a reduction, a gather, a broadcast, or
// raising an error.
void PerformOperation(Response response, ProcessSet& process_set) {
  std::vector<TensorTableEntry> entries;
  auto& timeline = horovod_global.timeline;
  process_set.tensor_queue.GetTensorEntriesFromResponse(response, entries,
                                                        process_set.joined);

  if (response.response_type() != Response::JOIN &&
      response.response_type() != Response::BARRIER) {
    for (auto& e : entries) {
      timeline.Start(e.tensor_name, response.response_type(), e.tensor->size());
    }

    if (entries.size() > 1) {
      auto first_entry = entries[0];
      // Note: it is OK for different entries to come from different frameworks
      // since buffer allocated here is guaranteed to survive at least till the
      // end of this operation.
      Status status = horovod_global.fusion_buffer.InitializeBuffer(
          process_set.controller->TensorFusionThresholdBytes(),
          first_entry.device, first_entry.context,
          horovod_global.current_nccl_stream,
          [&]() { timeline.ActivityStartAll(entries, INIT_FUSION_BUFFER); },
          [&]() { timeline.ActivityEndAll(entries); });
      if (!status.ok()) {
        LOG(DEBUG, horovod_global.global_controller->GetRank())
            << "InitializeBuffer Failed";
        for (auto& e : entries) {
          timeline.End(e.tensor_name, nullptr);
          e.FinishWithCallback(status);
        }
        return;
      }
    }
  }

  Status status;
  try {
    // process_set is passed here only for the case of Response::JOIN where
    // entries is empty. The other operations can infer process_set from
    // entries.
    status = op_manager->ExecuteOperation(entries, response, process_set);
  } catch (const std::exception& ex) {
    LOG(DEBUG, horovod_global.global_controller->GetRank())
        << "ExecuteOperation Failed";
    status = Status::UnknownError(ex.what());
  }

  if (!status.in_progress()) {
    for (auto& e : entries) {
      timeline.End(e.tensor_name, status.ok() ? e.output : nullptr);
      e.FinishWithCallback(status);
    }
  }
}

#if HAVE_MPI
void EnrichProcessSetWithMPIController(ProcessSet& process_set) {
  process_set.controller.reset(new MPIController(
      process_set.response_cache, process_set.tensor_queue,
      horovod_global.timeline, horovod_global.parameter_manager,
      process_set.group_table, horovod_global.timeline_controller,
      process_set.mpi_context));
}
#endif // HAVE_MPI

#if HAVE_GLOO
void EnrichProcessSetWithGlooController(ProcessSet& process_set) {
  process_set.controller.reset(new GlooController(
      process_set.response_cache, process_set.tensor_queue,
      horovod_global.timeline, horovod_global.parameter_manager,
      process_set.group_table, horovod_global.timeline_controller,
      process_set.gloo_context));
}
#endif // HAVE_GLOO

// If we already have a process set built from the same ranks (after sorting),
// return that and obtain its id. Otherwise register a new one, which will
// still need to be initialized, return it and obtain its id.
ProcessSet& GetProcessSetOrAddUnitialized(std::vector<int> ranks, int& id) {
  std::lock_guard<std::recursive_mutex> table_guard(
      horovod_global.process_set_table.mutex);
  std::sort(ranks.begin(), ranks.end());
  id = horovod_global.process_set_table.FindId(ranks);
  if (id >= 0) {
    return horovod_global.process_set_table.Get(id);
  }
  id = horovod_global.process_set_table.RegisterProcessSet(std::move(ranks));
  auto& process_set = horovod_global.process_set_table.Get(id);
#if HAVE_MPI
  if (horovod_global.control_operation == LibType::MPI) {
    EnrichProcessSetWithMPIController(process_set);
  }
#endif // HAVE_MPI
#if HAVE_GLOO
  if (horovod_global.control_operation == LibType::GLOO) {
    EnrichProcessSetWithGlooController(process_set);
  }
#endif // HAVE_GLOO
  assert(process_set.controller != nullptr);
  ParseStallInspectorFromEnv(process_set.controller->GetStallInspector());
  process_set.response_cache.set_capacity(
      (int)horovod_global.parameter_manager.CacheEnabled() *
      horovod_global.cache_capacity);
  return process_set;
}

// The background thread loop coordinates all the controller processes and the
// tensor reductions. The design of the communicator mechanism is limited by a
// few considerations:
//
//      1. Some MPI implementations require all MPI calls to happen from a
//      single thread. Since TensorFlow may use several threads for graph
//      processing, this means we must have our own dedicated thread for
//      dealing with MPI.
//      2. We want to gracefully handle errors, when all processes do not
//      properly agree upon what should happen (such as mismatched types or
//      shapes). To do so requires every process to know about the shapes
//      and types of the relevant tensors on the other processes.
//      3. The reductions and gathers should be able to happen in parallel
//      with other ongoing operations. This means that they cannot be blocking
//      ops, but rather must be async ops, the execution of which happens on a
//      separate thread.
//      4. We cannot guarantee that all the processes reduce their tensors
//      in the same order, so we cannot dispatch one thread per tensor,
//      otherwise we may end up dispatching many blocked threads and never
//      make progress if we have a thread pool limit.
bool RunLoopOnce(HorovodGlobalState& state);

void BackgroundThreadLoop(HorovodGlobalState& state) {
#if HAVE_CCL
  // Initialize ccl context
  if (state.cpu_operation == LibType::CCL) {
    ccl_context.Initialize();
  }
#endif

#if HAVE_MPI
  // Initialize mpi context
#if HAVE_DDL
  // If DDL is enabled, let DDL ops manage MPI environment.
  auto mpi_ctx_manager = DDL_MPIContextManager(ddl_context, gpu_context);
#else
  // Otherwise, let MPI ops be in charge.
  auto mpi_ctx_manager = MPIContextManager();
#endif
  if (global_mpi_context.IsEnabled()) {
    global_mpi_context.Initialize(mpi_ctx_manager);
    if (state.control_operation == LibType::MPI) {
      // Initializes global controller
      state.process_set_table.Initialize(global_mpi_context);
    }
  }
#endif

#if HAVE_GLOO
#if HAVE_MPI
  if (global_mpi_context.IsEnabled()) {
    // Initialize gloo context if mpi context is available
    global_gloo_context.InitializeFromMPI(
        state.process_set_table.Get(0).mpi_context, ParseGlooIface());
  } else
#endif // HAVE_MPI
  {
    global_gloo_context.Initialize(ParseGlooIface());
  }
  if (state.control_operation == LibType::GLOO) {
    // Initializes global controller
    state.process_set_table.Initialize(global_gloo_context);
  }
#endif // HAVE_GLOO

  assert(state.global_controller->IsInitialized());
  bool is_coordinator = state.global_controller->IsCoordinator();
  bool is_homogeneous = state.global_controller->IsHomogeneous();
  int size = state.global_controller->GetSize();
  int local_size = state.global_controller->GetLocalSize();
  int local_rank = state.global_controller->GetLocalRank();

  // Set background thread affinity
  parse_and_set_affinity(std::getenv(HOROVOD_THREAD_AFFINITY), local_size,
                         local_rank);

#if HAVE_GPU
  // Set number of GPU streams to use
  auto horovod_num_nccl_streams = std::getenv(HOROVOD_NUM_NCCL_STREAMS);
  if (horovod_num_nccl_streams != nullptr &&
      std::stol(horovod_num_nccl_streams, nullptr, 10) > 0) {
    state.num_nccl_streams = std::atoi(horovod_num_nccl_streams);
  }

#if HAVE_NCCL
  nccl_context.nccl_comms.resize(state.num_nccl_streams);
  local_nccl_context.nccl_comms.resize(state.num_nccl_streams);
  cross_nccl_context.nccl_comms.resize(state.num_nccl_streams);
  SetBoolFromEnv(HOROVOD_ELASTIC, nccl_context.elastic, true);
  SetBoolFromEnv(HOROVOD_ELASTIC, local_nccl_context.elastic, true);
  SetBoolFromEnv(HOROVOD_ELASTIC, cross_nccl_context.elastic, true);
#endif
  gpu_context.streams.resize(state.num_nccl_streams);

  // Create finalizer thread pool (one thread per stream)
  gpu_context.finalizer_thread_pool.create(state.num_nccl_streams);
#endif

#if HAVE_NVTX
  if (GetBoolEnvOrDefault(HOROVOD_DISABLE_NVTX_RANGES, false)) {
    NvtxOpRange::nvtx_ops_handle.Disable();
    horovod_global.timeline.DisableNvtx();
  }
#endif // HAVE_NVTX

  // Open the timeline file on coordinator.
  bool should_enable_timeline = false;
  auto timeline_env = std::getenv(HOROVOD_TIMELINE);
  if (timeline_env) {
    if (is_coordinator) {
      auto horovod_timeline = std::string(timeline_env);
      if (horovod_timeline != "DYNAMIC") {
        state.timeline.Initialize(horovod_timeline,
                                  static_cast<unsigned int>(size));
      } else {
        state.timeline.Initialize("", static_cast<unsigned int>(size));
      }
    }
    should_enable_timeline = true;
  }
  state.timeline_controller.SetTimelineEnabled(should_enable_timeline);

  SetBoolFromEnv(HOROVOD_ELASTIC, state.elastic_enabled, true);

  ParseStallInspectorFromEnv(
      state.process_set_table.Get(0).controller->GetStallInspector());
  bool mark_cycles = false;
  SetBoolFromEnv(HOROVOD_TIMELINE_MARK_CYCLES, mark_cycles, true);
  state.timeline_controller.SetMarkCyclesInTimelinePending(mark_cycles);
  state.mark_cycles_in_timeline = mark_cycles;

  // Override Tensor Fusion threshold, if it's set.
  state.parameter_manager.SetTensorFusionThresholdBytes(128 * 1024 * 1024);
  auto horovod_fusion_threshold = std::getenv(HOROVOD_FUSION_THRESHOLD);
  if (horovod_fusion_threshold != nullptr) {
    int64_t threshold = std::strtol(horovod_fusion_threshold, nullptr, 10);
    state.parameter_manager.SetTensorFusionThresholdBytes(threshold, true);
  }

  // Override the cycle time.
  state.parameter_manager.SetCycleTimeMs(1);
  bool enable_xla_ops = false;
  common::SetBoolFromEnv(HOROVOD_ENABLE_XLA_OPS, enable_xla_ops, true);
  if (enable_xla_ops) {
    // Setting the default Cycle Time to 0 because the XLA runtime is sensitive
    // to latencies.
    state.parameter_manager.SetCycleTimeMs(0);
  }

  auto horovod_cycle_time = std::getenv(HOROVOD_CYCLE_TIME);
  if (horovod_cycle_time != nullptr) {
    state.parameter_manager.SetCycleTimeMs(
        std::strtof(horovod_cycle_time, nullptr), true);
  }

  // Override response cache capacity, if it's set.
  state.parameter_manager.SetCacheEnabled(true);
  auto horovod_cache_capacity = std::getenv(HOROVOD_CACHE_CAPACITY);
  if (horovod_cache_capacity != nullptr) {
    uint32_t cache_capacity = std::strtol(horovod_cache_capacity, nullptr, 10);
    state.cache_capacity = cache_capacity;
    state.parameter_manager.SetCacheEnabled(cache_capacity > 0, true);
  }
  state.process_set_table.Get(0).response_cache.set_capacity(
      (int)state.parameter_manager.CacheEnabled() * state.cache_capacity);

  // Set flag for hierarchical allgather. Ignore if Horovod is running on a
  // single node.
  auto horovod_hierarchical_allgather =
      std::getenv(HOROVOD_HIERARCHICAL_ALLGATHER);
  state.parameter_manager.SetHierarchicalAllgather(false);
  if (horovod_hierarchical_allgather != nullptr) {
    bool value = std::strtol(horovod_hierarchical_allgather, nullptr, 10) > 0 &&
                 (size != local_size);
    state.parameter_manager.SetHierarchicalAllgather(value, true);
  }
  // Set flag for hierarchical allreduce. Ignore if Horovod is running on a
  // single node.
  auto horovod_hierarchical_allreduce =
      std::getenv(HOROVOD_HIERARCHICAL_ALLREDUCE);
  state.parameter_manager.SetHierarchicalAllreduce(false);
  if (horovod_hierarchical_allreduce != nullptr) {
    bool value = std::strtol(horovod_hierarchical_allreduce, nullptr, 10) > 0 &&
                 (size != local_size);
    state.parameter_manager.SetHierarchicalAllreduce(value, true);
  }

#if HOROVOD_GPU_ALLREDUCE != 'N' && HOROVOD_GPU_ALLREDUCE != 'D'
  // Hierarchical allreduce is not supported without NCCL or DDL
  state.parameter_manager.SetHierarchicalAllreduce(false, true);
#endif

  // Issue warning if hierarchical allreduce is enabled in heterogeneous cluster
  if (is_coordinator &&
      (state.parameter_manager.HierarchicalAllreduce() ||
       state.parameter_manager.HierarchicalAllgather()) &&
      !is_homogeneous) {
    std::cerr
        << "WARNING: Using different number of ranks per node might cause "
           "performance loss in hierarchical allgather and "
           "hierarchical allreduce. Consider assigning the same "
           "number of ranks to each node, or disabling hierarchical "
           "allgather and hierarchical allreduce.";
  }

  // Set flag for torus allreduce. Ignore if Horovod is running on a
  // single node.
  auto horovod_torus_allreduce =
      std::getenv(HOROVOD_TORUS_ALLREDUCE);
  state.parameter_manager.SetTorusAllreduce(false);
  if (horovod_torus_allreduce != nullptr) {
    bool value = std::strtol(horovod_torus_allreduce, nullptr, 10) > 0 &&
                 (size != local_size);
    state.parameter_manager.SetTorusAllreduce(value, true);
  }
#if HOROVOD_GPU_ALLREDUCE != 'N' && HOROVOD_GPU_ALLREDUCE != 'D'
  // Torus allreduce is not supported without NCCL or DDL
  state.parameter_manager.SetTorusAllreduce(false, true);
#endif

  // Set flag to control use of batched memcopy kernel on GPU
  auto horovod_batch_d2d_memcopies = std::getenv(HOROVOD_BATCH_D2D_MEMCOPIES);
  if (horovod_batch_d2d_memcopies != nullptr &&
      std::strtol(horovod_batch_d2d_memcopies, nullptr, 10) == 0) {
    state.batch_d2d_memcopies = false;
  }

  // Check if group fusion should be disabled
  SetBoolFromEnv(HOROVOD_DISABLE_GROUP_FUSION, state.disable_group_fusion,
                 true);

  // Check if async completion should be enabled
  SetBoolFromEnv(HOROVOD_ENABLE_ASYNC_COMPLETION, state.enable_async_completion,
                 true);
  if (enable_xla_ops) {
    // Enable async completion when XLA ops are enabled. Sine the XLA runtime is
    // single-threaded, async completion is essential to reduce host overhead.
    state.enable_async_completion = true;
  }

  // Enable auto-tuning.
  auto horovod_autotune = std::getenv(HOROVOD_AUTOTUNE);
  if (horovod_autotune != nullptr &&
      std::strtol(horovod_autotune, nullptr, 10) > 0) {
    auto horovod_autotune_log = std::getenv(HOROVOD_AUTOTUNE_LOG);
    state.parameter_manager.Initialize(
        state.global_controller->GetRank(), RANK_ZERO,
        horovod_autotune_log != nullptr ? std::string(horovod_autotune_log)
                                        : "");
    state.parameter_manager.SetAutoTuning(true);
  }

  // Set chunk size for MPI based Adasum allreduce algorithms
  auto horovod_adasum_mpi_chunk_size =
      std::getenv(HOROVOD_ADASUM_MPI_CHUNK_SIZE);
  if (horovod_adasum_mpi_chunk_size != nullptr) {
    state.adasum_mpi_chunk_size =
        std::strtol(horovod_adasum_mpi_chunk_size, nullptr, 10);
  }

  op_manager.reset(CreateOperationManager(state));

  state.dynamic_process_sets =
      GetBoolEnvOrDefault(HOROVOD_DYNAMIC_PROCESS_SETS, false);

  // Register and initialize any non-global process set requested during Horovod
  // initialization.
  try {
    for (const auto& process_set_ranks : state.process_set_ranks_to_register) {
      int id;
      GetProcessSetOrAddUnitialized(process_set_ranks, id);
    }
    int32_t initialized_count = 0;
#if HAVE_MPI
    if (state.control_operation == LibType::MPI) {
      initialized_count =
          state.process_set_table.InitializeRegisteredAndRemoveMarkedIfReady(
              global_mpi_context); // will only initialize, not remove
    }
#endif // HAVE_MPI
#if HAVE_GLOO
    if (state.control_operation == LibType::GLOO) {
      initialized_count =
          state.process_set_table.InitializeRegisteredAndRemoveMarkedIfReady(
              global_gloo_context); // will only initialize, not remove
    }
#endif // HAVE_GLOO
    if (state.process_set_ranks_to_register.size() > 0 &&
        initialized_count == 0) {
      throw std::logic_error("Different ranks tried to set up mismatching "
                             "numbers of process sets");
    }
    state.process_set_ranks_to_register.clear();
  } catch (const std::exception& ex) {
    LOG(ERROR, horovod_global.global_controller->GetRank())
        << "Horovod could not be initialized: " << ex.what();
    state.initialization_failed = true;
    goto shutdown;
  }

  // Signal that initialization is completed.
  state.initialization_done = true;
  LOG(INFO, horovod_global.global_controller->GetRank())
      << "Horovod initialized";

  // Iterate until shutdown.
  try {
    while (RunLoopOnce(state))
      ;
  } catch (const std::exception& ex) {
    LOG(ERROR, horovod_global.global_controller->GetRank())
        << "Horovod background loop uncaught exception: " << ex.what();
  }

shutdown:
  // Finalize all contexts
#if HAVE_NCCL
  nccl_context.ShutDown();
  local_nccl_context.ShutDown();
  cross_nccl_context.ShutDown();
#endif

  LOG(DEBUG, horovod_global.global_controller->GetRank())
      << "Shutting down background thread";

  // Signal that shutdown has been requested.
  state.shut_down = true;

  // For each process set: Notify all outstanding operations that Horovod has
  // been shut down, finalize tensor queue and communicators.
  // If there are multiple process sets, this blocks until all processes are
  // ready for shutdown and finalizes all process sets.
#if HAVE_MPI
  if (state.control_operation == LibType::MPI) {
    horovod_global.process_set_table.Finalize(global_mpi_context,
                                              SHUT_DOWN_ERROR);
  }
#endif // HAVE_MPI
#if HAVE_GLOO
  if (state.control_operation == LibType::GLOO) {
    horovod_global.process_set_table.Finalize(global_gloo_context,
                                              SHUT_DOWN_ERROR);
  }
#endif // HAVE_GLOO

#if HAVE_GPU
  gpu_context.Finalize();
#endif

#if HAVE_GLOO
  global_gloo_context.Finalize();
#endif

#if HAVE_MPI
  global_mpi_context.Finalize(mpi_ctx_manager);
#endif

#if HAVE_CCL
  if (state.cpu_operation == LibType::CCL) {
    ccl_context.Finalize();
  }
#endif
}

bool RunLoopOnce(HorovodGlobalState& state) {
  // This delay determines thread frequency and communication message latency
  auto start_time = std::chrono::steady_clock::now();
  auto sleep_duration = state.last_cycle_start +
                        std::chrono::microseconds(long(
                            state.parameter_manager.CycleTimeMs() * 1000.)) -
                        start_time;
  if (sleep_duration > std::chrono::steady_clock::duration::zero()) {
    std::this_thread::sleep_for(sleep_duration);
  }
  state.last_cycle_start = std::chrono::steady_clock::now();

  if (state.mark_cycles_in_timeline) {
    // Mark start of the new cycle.
    state.timeline.MarkCycleStart();
  }

  bool this_process_requested_shutdown = state.shut_down;

  if (state.dynamic_process_sets) {
    // Initialize any newly added process set that has been registered by all
    // Horovod processes and remove a process set that has been marked for
    // removal by all Horovod processes.
#if HAVE_MPI
    if (state.control_operation == LibType::MPI) {
      state.process_set_table.InitializeRegisteredAndRemoveMarkedIfReady(
          global_mpi_context);
    }
#endif // HAVE_MPI
#if HAVE_GLOO
    if (state.control_operation == LibType::GLOO) {
      state.process_set_table.InitializeRegisteredAndRemoveMarkedIfReady(
          global_gloo_context);
    }
#endif // HAVE_GLOO
  }

  bool should_shutdown = false;
  for (auto process_set_id : state.process_set_table.Ids()) {
    if (should_shutdown) {
      break;
    }
    auto& process_set = state.process_set_table.Get(process_set_id);
    if (!process_set.initialization_done) {
      continue;
    }
    auto response_list =
        process_set.IsCurrentProcessIncluded()
            ? process_set.controller->ComputeResponseList(
                  this_process_requested_shutdown, state, process_set)
            : ResponseList();

    if (process_set_id == 0) {
      state.mark_cycles_in_timeline =
          state.timeline_controller.MarkCyclesInTimelinePending();
    }

    // Get tensor name and size data for autotuning. // TODO: extend for all
    // process sets?
    int64_t total_tensor_size = 0;
    std::vector<std::string> tensor_names;
    if (process_set_id == 0 && state.parameter_manager.IsAutoTuning()) {
      total_tensor_size = process_set.tensor_queue.GetTensorDataForAutotuner(
          response_list, tensor_names);
    }

    // Perform the collective operation. All nodes in the process set should end
    // up performing the same operation.
    if (process_set.IsCurrentProcessIncluded()) {
      int global_rank = state.global_controller->GetRank();
      for (auto& response : response_list.responses()) {
        if (!process_set.group_table.empty()) {
          // Deregister any completed groups
          process_set.group_table.DeregisterGroups(response.tensor_names());
        }

        LOG(TRACE, global_rank) << "Process set id " << process_set_id;
        LOG(TRACE, global_rank)
            << "Performing " << response.tensor_names_string();
        LOG(TRACE, global_rank)
            << "Processing " << response.tensor_names().size() << " tensors";
        PerformOperation(response, process_set);
        LOG(TRACE, global_rank)
            << "Finished performing " << response.tensor_names_string();
      }
    }

    if (process_set_id == 0 && state.parameter_manager.IsAutoTuning()) {
      bool should_sync =
          state.parameter_manager.Update(tensor_names, total_tensor_size);

      if (should_sync) {
        process_set.controller->SynchronizeParameters();
      }
    }

    should_shutdown |= response_list.shutdown();
  }

  return !should_shutdown;
}

// Start Horovod background thread. Ensure that this is
// only done once no matter how many times this function is called.
// Returns false if initialization failed, otherwise true.
bool InitializeHorovodOnce(
    const std::vector<int>& ranks,
    const std::vector<std::vector<int>>& process_set_ranks) {
  // Ensure background thread is only started once.
  if (!horovod_global.initialize_flag.test_and_set()) {
    horovod_global.control_operation = ParseControllerOpsFromEnv();
    horovod_global.cpu_operation = ParseCPUOpsFromEnv();
#if HAVE_MPI
    // Enable mpi if it's used either in cpu data transfer or controller
    if (horovod_global.cpu_operation == LibType::MPI ||
        horovod_global.control_operation == LibType::MPI) {
      global_mpi_context.Enable(ranks);
    }

    if (horovod_global.control_operation == LibType::MPI) {
      auto& global_process_set = horovod_global.process_set_table.Get(0);
      EnrichProcessSetWithMPIController(global_process_set);
      horovod_global.global_controller = global_process_set.controller;
      horovod_global.process_set_ranks_to_register = process_set_ranks;
    }
#endif

#if HAVE_GLOO
    // Enable gloo if it's used either in cpu data transfer or controller
    if (horovod_global.cpu_operation == LibType::GLOO ||
        horovod_global.control_operation == LibType::GLOO) {
      global_gloo_context.Enable();
    }

    if (horovod_global.control_operation == LibType::GLOO) {
      auto& global_process_set = horovod_global.process_set_table.Get(0);
      EnrichProcessSetWithGlooController(global_process_set);
      horovod_global.global_controller = global_process_set.controller;
      horovod_global.process_set_ranks_to_register = process_set_ranks;
    }
#endif
    // Reset initialization flag
    horovod_global.initialization_done = false;
    horovod_global.background_thread =
        std::thread(BackgroundThreadLoop, std::ref(horovod_global));
  }

  // Wait to ensure that the background thread has finished initializing MPI.
  while (!horovod_global.initialization_done &&
         !horovod_global.initialization_failed) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (horovod_global.initialization_failed) {
    return false;
  }
  LOG(DEBUG) << "Background thread init done";
  return true;
}

std::vector<std::vector<int>>
BuildProcessSetRanksVectors(const int* process_set_ranks,
                            const int* process_set_sizes,
                            int num_process_sets) {
  std::vector<std::vector<int>> result;
  for (int p = 0; p < num_process_sets; ++p) {
    result.emplace_back(process_set_ranks,
                        process_set_ranks + process_set_sizes[p]);
    process_set_ranks += process_set_sizes[p];
  }
  return result;
};

} // namespace

Status CheckInitialized() {
  if (!horovod_global.initialization_done) {
    return NOT_INITIALIZED_ERROR;
  }
  return Status::OK();
}

extern "C" {

bool horovod_init(const int* ranks, int nranks, const int* process_set_ranks,
                  const int* process_set_sizes, int num_process_sets) {
  std::vector<int> ranks_vec;
  if (ranks && nranks > 0) {
    ranks_vec.assign(ranks, ranks + nranks);
  }
  return InitializeHorovodOnce(
      ranks_vec, BuildProcessSetRanksVectors(
                     process_set_ranks, process_set_sizes, num_process_sets));
}

#if HAVE_MPI
bool horovod_init_multi_comm(MPI_Comm* comm, int ncomms,
                             const int* process_set_ranks_via_ranks,
                             const int* process_set_sizes_via_ranks,
                             int num_process_sets_via_ranks) {
  assert(ncomms > 0);
  MPI_Comm_dup(comm[0], &global_mpi_context.global_comm);

  std::vector<std::vector<int>> process_set_ranks_vecs;

  int global_rank;
  MPI_Comm_rank(global_mpi_context.global_comm, &global_rank);
  int global_size;
  MPI_Comm_size(global_mpi_context.global_comm, &global_size);
  MPI_Group global_group;
  MPI_Comm_group(global_mpi_context.global_comm, &global_group);
  for (int i = 1; i < ncomms; ++i) {
    auto sub_comm = comm[i];

    MPI_Group sub_group;
    {
      MPI_Comm_group(sub_comm, &sub_group);
      MPI_Group diff_group;
      MPI_Group_difference(sub_group, global_group, &diff_group);
      if (diff_group != MPI_GROUP_EMPTY) {
        LOG(ERROR)
            << "Group of processes in horovod_init_multi_comm argument "
               "number " +
                   std::to_string(i) +
                   " is not a subset of the assumed global communicator.";
        return false;
      }
    }

    int rank;
    MPI_Comm_rank(sub_comm, &rank);
    int size;
    MPI_Comm_size(sub_comm, &size);

    auto global_ranks = std::vector<int>(size);
    {
      auto sub_ranks = std::vector<int>(size);
      std::iota(sub_ranks.begin(), sub_ranks.end(), 0);
      MPI_Group_translate_ranks(sub_group, size, sub_ranks.data(), global_group,
                                global_ranks.data());
    }

    std::set<std::vector<int>> collected_process_sets; // sorted
    {
      auto sub_sizes = std::vector<int>(global_size);
      MPI_Allgather(&size, 1, MPI_INT, sub_sizes.data(), 1, MPI_INT,
                    global_mpi_context.global_comm);

      auto displ = std::vector<int>(global_size);
      for (int j = 1; j < global_size; ++j) {
        displ[j] = displ[j - 1] + sub_sizes[j - 1];
      }

      auto process_sets_buf = std::vector<int>(
          std::accumulate(sub_sizes.begin(), sub_sizes.end(), 0));
      MPI_Allgatherv(global_ranks.data(), size, MPI_INT,
                     process_sets_buf.data(), sub_sizes.data(), displ.data(),
                     MPI_INT, global_mpi_context.global_comm);

      for (int j = 0; j < global_size; ++j) {
        collected_process_sets.insert(
            std::vector<int>(&process_sets_buf[displ[j]],
                             &process_sets_buf[displ[j] + sub_sizes[j]]));
      }
    }

    process_set_ranks_vecs.insert(process_set_ranks_vecs.end(),
                                  collected_process_sets.begin(),
                                  collected_process_sets.end());
  }

  // Add process sets defined via ranks:
  std::vector<std::vector<int>> process_set_ranks_via_ranks_vecs =
      BuildProcessSetRanksVectors(process_set_ranks_via_ranks,
                                  process_set_sizes_via_ranks,
                                  num_process_sets_via_ranks);
  process_set_ranks_vecs.insert(process_set_ranks_vecs.end(),
                                process_set_ranks_via_ranks_vecs.begin(),
                                process_set_ranks_via_ranks_vecs.end());

  return InitializeHorovodOnce(std::vector<int>(), process_set_ranks_vecs);
}

int horovod_comm_process_set(MPI_Comm comm) {
  if (!horovod_global.initialization_done or !horovod_mpi_enabled()) {
    return HOROVOD_PROCESS_SET_ERROR_INIT;
  }
  int size;
  MPI_Comm_size(comm, &size);
  auto global_ranks = std::vector<int>(size);
  {
    auto sub_ranks = std::vector<int>(size);
    std::iota(sub_ranks.begin(), sub_ranks.end(), 0);
    MPI_Group group;
    MPI_Comm_group(comm, &group);
    MPI_Group global_group;
    MPI_Comm_group(global_mpi_context.global_comm, &global_group);
    MPI_Group_translate_ranks(group, size, sub_ranks.data(), global_group,
                              global_ranks.data());
    // global_ranks is sorted ascendingly
  }
  int32_t id = horovod_global.process_set_table.FindId(global_ranks);
  if (id >= 0) {
    return id;
  }
  return HOROVOD_PROCESS_SET_ERROR_UNKNOWN_SET;
}

#endif

void horovod_shutdown() {
  if (horovod_global.background_thread.joinable()) {
    horovod_global.timeline.Shutdown();
    horovod_global.shut_down = true;
    horovod_global.background_thread.join();

    // Reset the initialization flag to allow restarting with horovod_init(...)
    horovod_global.initialize_flag.clear();
    horovod_global.shut_down = false;
    horovod_global.initialization_done = false;
  }
}

int horovod_is_initialized() {
  return int(horovod_global.initialization_done.load());
}

int horovod_start_timeline(const char* file_name, bool mark_cycles) {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  if (!horovod_global.timeline_controller.TimelineEnabled()) {
    return -2;
  }
  bool is_coordinator = horovod_global.global_controller->IsCoordinator();
  if (is_coordinator) {
    horovod_global.timeline.Initialize(
        std::string(file_name), horovod_global.global_controller->GetSize());
    horovod_global.timeline.SetPendingTimelineFile(std::string(file_name));
  }
  horovod_global.timeline_controller.SetMarkCyclesInTimelinePending(
      mark_cycles);
  return 1;
}

int horovod_stop_timeline() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  if (!horovod_global.timeline_controller.TimelineEnabledPending()) {
    LOG(INFO) << " Timeline is already stopped. Please start timeline before "
                 "stopping it.";
    return 1;
  }
  bool is_coordinator = horovod_global.global_controller->IsCoordinator();
  if (is_coordinator) {
    horovod_global.timeline.SetPendingTimelineFile(std::string(""));
  }
  return 1;
}

int horovod_rank() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.global_controller->GetRank();
}

int horovod_local_rank() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.global_controller->GetLocalRank();
}

int horovod_cross_rank() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.global_controller->GetCrossRank();
}

int horovod_size() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.global_controller->GetSize();
}

int horovod_local_size() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.global_controller->GetLocalSize();
}

int horovod_cross_size() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.global_controller->GetCrossSize();
}

bool horovod_is_homogeneous() {
  return horovod_global.global_controller->IsHomogeneous();
}

int horovod_mpi_threads_supported() {
  if (!horovod_global.initialization_done) {
    return -1;
  }

#if HAVE_MPI
  auto mpiController = std::dynamic_pointer_cast<MPIController>(
      horovod_global.global_controller);
  return mpiController->IsMpiThreadsSupported() ? 1 : 0;
#endif

  return -1;
}

bool horovod_mpi_enabled() {
#if HAVE_MPI
  return global_mpi_context.IsEnabled();
#else
  return false;
#endif
}

bool horovod_mpi_built() {
#if HAVE_MPI
  return true;
#else
  return false;
#endif
}

bool horovod_gloo_enabled() {
#if HAVE_GLOO
  return global_gloo_context.IsEnabled();
#else
  return false;
#endif
}

bool horovod_gloo_built() {
#if HAVE_GLOO
  return true;
#else
  return false;
#endif
}

int horovod_nccl_built() {
#if HAVE_NCCL
  return NCCL_VERSION_CODE;
#else
  return 0;
#endif
}

bool horovod_ddl_built() {
#if HAVE_DDL
  return true;
#else
  return false;
#endif
}

bool horovod_ccl_built() {
#if HAVE_CCL
  return true;
#else
  return false;
#endif
}

bool horovod_cuda_built() {
#if HAVE_CUDA
  return true;
#else
  return false;
#endif
}

bool horovod_rocm_built() {
#if HAVE_ROCM
  return true;
#else
  return false;
#endif
}

int horovod_reduce_op_average() { return ReduceOp::AVERAGE; }

int horovod_reduce_op_sum() { return ReduceOp::SUM; }

int horovod_reduce_op_adasum() { return ReduceOp::ADASUM; }

int horovod_reduce_op_min() { return ReduceOp::MIN; }

int horovod_reduce_op_max() { return ReduceOp::MAX; }

int horovod_reduce_op_product() { return ReduceOp::PRODUCT; }

const int HOROVOD_PROCESS_SET_ERROR_INIT = -1;
const int HOROVOD_PROCESS_SET_ERROR_DYNAMIC = -2;
const int HOROVOD_PROCESS_SET_ERROR_UNKNOWN_SET = -3;
const int HOROVOD_PROCESS_SET_ERROR_FOREIGN_SET = -4;
const int HOROVOD_PROCESS_SET_ERROR_SHUTDOWN = -5;
const int HOROVOD_PROCESS_SET_ERROR_EXISTING_SET = -6;

int horovod_add_process_set(const int* ranks, int nrank) {
  if (!horovod_global.initialization_done) {
    return HOROVOD_PROCESS_SET_ERROR_INIT;
  }
  if (!horovod_global.dynamic_process_sets) {
    return HOROVOD_PROCESS_SET_ERROR_DYNAMIC;
  }

  int id;
  ProcessSet* process_set = nullptr;
  {
    // Lock the table so the background thread will not initialize a newly added
    // process set before we leave this critical section.
    std::lock_guard<std::recursive_mutex> table_lock(
        horovod_global.process_set_table.mutex);
    process_set = &GetProcessSetOrAddUnitialized(
        ranks && nrank > 0 ? std::vector<int>(ranks, ranks + nrank)
                           : std::vector<int>(),
        id);
    if (process_set->initialization_done) {
      // A process set with these ranks existed before.
      return HOROVOD_PROCESS_SET_ERROR_EXISTING_SET;
    }
  }

  // Block until the background thread has initialized the process set.
  while (true) {
    if (process_set->initialization_done) {
      return id;
    }
    if (horovod_global.shut_down) {
      return HOROVOD_PROCESS_SET_ERROR_SHUTDOWN;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

int horovod_remove_process_set(int process_set_id) {
  if (!horovod_global.initialization_done) {
    return HOROVOD_PROCESS_SET_ERROR_INIT;
  }
  if (!horovod_global.dynamic_process_sets) {
    return HOROVOD_PROCESS_SET_ERROR_DYNAMIC;
  }

  {
    std::lock_guard<std::recursive_mutex> table_lock(
        horovod_global.process_set_table.mutex);

    if (!horovod_global.process_set_table.Contains(process_set_id)) {
      return HOROVOD_PROCESS_SET_ERROR_UNKNOWN_SET;
    }

    horovod_global.process_set_table.MarkProcessSetForRemoval(process_set_id);
  }

  // Block until the background thread has removed the process set.
  while (true) {
    if (horovod_global.process_set_table.ProcessSetHasJustBeenRemoved()) {
      return process_set_id;
    }
    if (horovod_global.shut_down) {
      return HOROVOD_PROCESS_SET_ERROR_SHUTDOWN;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

int horovod_process_set_rank(int process_set_id) {
  if (process_set_id == 0) {
    return horovod_rank();
  }
  if (!horovod_global.initialization_done) {
    return HOROVOD_PROCESS_SET_ERROR_INIT;
  }
  if (!horovod_global.process_set_table.Contains(process_set_id)) {
    return HOROVOD_PROCESS_SET_ERROR_UNKNOWN_SET;
  }
  auto& process_set = horovod_global.process_set_table.Get(process_set_id);
  if (process_set.IsCurrentProcessIncluded()) {
    return process_set.controller->GetRank();
  }
  return HOROVOD_PROCESS_SET_ERROR_FOREIGN_SET;
}

int horovod_process_set_size(int process_set_id) {
  if (process_set_id == 0) {
    return horovod_size();
  }
  if (!horovod_global.initialization_done) {
    return HOROVOD_PROCESS_SET_ERROR_INIT;
  }
  std::lock_guard<std::recursive_mutex> table_lock(
      horovod_global.process_set_table.mutex);
  if (!horovod_global.process_set_table.Contains(process_set_id)) {
    return HOROVOD_PROCESS_SET_ERROR_UNKNOWN_SET;
  }
  auto& process_set = horovod_global.process_set_table.Get(process_set_id);
  return static_cast<int>(process_set.registered_global_ranks.size());
}

int horovod_process_set_included(int process_set_id) {
  if (process_set_id == 0) {
    return 1;
  }
  if (!horovod_global.initialization_done) {
    return HOROVOD_PROCESS_SET_ERROR_INIT;
  }
  auto& process_set = horovod_global.process_set_table.Get(process_set_id);
  std::lock_guard<std::recursive_mutex> table_lock(
      horovod_global.process_set_table.mutex);
  if (!horovod_global.process_set_table.Contains(process_set_id)) {
    return HOROVOD_PROCESS_SET_ERROR_UNKNOWN_SET;
  }
  return static_cast<int>(process_set.IsCurrentProcessIncluded());
}

int horovod_number_of_process_sets() {
  return static_cast<int>(horovod_global.process_set_table.Ids().size());
}

void horovod_process_set_ids(int* ids_prealloc) {
  const auto ids_vec = horovod_global.process_set_table.Ids();
  std::copy(ids_vec.begin(), ids_vec.end(), ids_prealloc);
}

int horovod_process_set_ranks(int id, int* ranks_prealloc) {
  if (!horovod_global.initialization_done) {
    return HOROVOD_PROCESS_SET_ERROR_INIT;
  }
  try {
    const auto& process_set = horovod_global.process_set_table.Get(id);
    if (!process_set.initialization_done) {
      return HOROVOD_PROCESS_SET_ERROR_INIT;
    }
    std::copy(process_set.registered_global_ranks.begin(),
              process_set.registered_global_ranks.end(), ranks_prealloc);
  } catch (const std::out_of_range& ex) {
    return HOROVOD_PROCESS_SET_ERROR_UNKNOWN_SET;
  }
  return 0;
}
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueTensorAllreduce(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              ReadyEventList ready_event_list, std::string name,
                              const int device, StatusCallback callback,
                              ReduceOp reduce_op, double prescale_factor,
                              double postscale_factor, int32_t process_set_id) {
  // Wrap inputs in std::vector and pass onto multi tensor implementation
  std::vector<std::shared_ptr<OpContext>> contexts;
  std::vector<std::shared_ptr<Tensor>> tensors;
  std::vector<std::shared_ptr<Tensor>> outputs;
  std::vector<ReadyEventList> ready_event_lists;
  std::vector<std::string> names;
  std::vector<StatusCallback> callbacks;

  contexts.emplace_back(std::move(context));
  tensors.emplace_back(std::move(tensor));
  outputs.emplace_back(std::move(output));
  ready_event_lists.emplace_back(std::move(ready_event_list));
  names.emplace_back(std::move(name));
  callbacks.emplace_back(std::move(callback));

  return EnqueueTensorAllreduces(
      contexts, tensors, outputs, ready_event_lists, names, device, callbacks,
      reduce_op, prescale_factor, postscale_factor, process_set_id);
}

Status
EnqueueTensorAllreduces(std::vector<std::shared_ptr<OpContext>>& contexts,
                        std::vector<std::shared_ptr<Tensor>>& tensors,
                        std::vector<std::shared_ptr<Tensor>>& outputs,
                        std::vector<ReadyEventList>& ready_event_lists,
                        std::vector<std::string>& names, const int device,
                        std::vector<StatusCallback>& callbacks,
                        ReduceOp reduce_op, double prescale_factor,
                        double postscale_factor, int32_t process_set_id) {
  if (horovod_global.cpu_operation == LibType::CCL && process_set_id > 0 &&
      device == CPU_DEVICE_ID) {
    return Status::InvalidArgument(
        "Process sets are not supported yet with oneCCL operations.");
  }
  if (!horovod_global.process_set_table.Contains(process_set_id)) {
    return Status::InvalidArgument("Allreduce: Process set provided does not "
                                   "exist, or has not been registered.");
  }
  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  auto& process_set = horovod_global.process_set_table.Get(process_set_id);
  Status status;

  if (reduce_op == ReduceOp::ADASUM) {
#if HAVE_NCCL && !HAVE_ROCM
    if (device != CPU_DEVICE_ID) {
      // Averaging by local size happens via postscale_factor
      postscale_factor /= process_set.controller->GetLocalSize();
    }
#endif
  }

  std::vector<Request> messages;
  std::vector<TensorTableEntry> entries;
  messages.reserve(tensors.size());
  entries.reserve(tensors.size());

  for (int n = 0; n < (int)tensors.size(); ++n) {
    Request message;
    message.set_request_rank(process_set.controller->GetRank());
    message.set_tensor_name(names[n]);
    message.set_tensor_type(tensors[n]->dtype());
    message.set_device(device);
    message.set_prescale_factor(prescale_factor);
    message.set_postscale_factor(postscale_factor);

    if (reduce_op == ReduceOp::ADASUM) {
      message.set_request_type(Request::ADASUM);
    } else {
      message.set_request_type(Request::ALLREDUCE);
    }
    message.set_reduce_op(reduce_op);

    message.set_tensor_shape(tensors[n]->shape().to_vector());
    messages.push_back(std::move(message));

    TensorTableEntry e;
    e.tensor_name = names[n];
    e.context = std::move(contexts[n]);
    // input and output can be the same, only move when safe
    if (tensors[n] != outputs[n]) {
      e.tensor = std::move(tensors[n]);
      e.output = std::move(outputs[n]);
    } else {
      e.tensor = tensors[n];
      e.output = outputs[n];
    }
    e.process_set_id = process_set_id;
    e.ready_event_list = std::move(ready_event_lists[n]);
    e.device = device;
    e.callback = std::move(callbacks[n]);

    entries.push_back(std::move(e));
  }

  // Start appropriate NVTX range
  if (tensors.size() == 1) {
    auto& e = entries[0];
    e.nvtx_op_range.Start(RegisteredNvtxOp::HorovodAllreduce, e.tensor->size());
  } else {
    auto total_size =
        std::accumulate(entries.begin(), entries.end(), 0ll,
                        [](int64_t size_sum, const TensorTableEntry& e) {
                          return size_sum + e.tensor->size();
                        });
    SharedNvtxOpRange range;
    range.Start(RegisteredNvtxOp::HorovodGroupedAllreduce, total_size);
    for (auto& e : entries) {
      e.nvtx_op_range = range;
    }
  }

  if (!process_set.IsCurrentProcessIncluded()) {
    return Status::InvalidArgument(
        "Allreduce: Rank " +
        std::to_string(horovod_global.global_controller->GetRank()) +
        " is not a member of the provided process set.");
  }

  std::string tensors_enqueued;
  for (const auto& n : names) {
    tensors_enqueued += n + "; ";
  }
  LOG(TRACE, horovod_global.global_controller->GetRank())
      << "Enqueued " << tensors_enqueued;

  // Only create groups larger than 1 tensor, unless disable_group_fusion is
  // requested. In that case, even single tensor groups are created to enforce
  // disabling fusion.
  if (tensors.size() > 1 || horovod_global.disable_group_fusion) {
    auto group_id = process_set.group_table.RegisterGroup(std::move(names));
    for (auto& message : messages) {
      message.set_group_id(group_id);
    }
  }

  status = process_set.tensor_queue.AddToTensorQueueMulti(entries, messages);

  return status;
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueTensorAllgather(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              ReadyEventList ready_event_list,
                              const std::string& name, const int device,
                              StatusCallback callback, int32_t process_set_id) {
  // Wrap inputs in std::vector and pass onto multi tensor implementation
  std::vector<std::shared_ptr<OpContext>> contexts;
  std::vector<std::shared_ptr<Tensor>> tensors;
  std::vector<ReadyEventList> ready_event_lists;
  std::vector<std::string> names;
  std::vector<StatusCallback> callbacks;

  contexts.emplace_back(std::move(context));
  tensors.emplace_back(std::move(tensor));
  ready_event_lists.emplace_back(std::move(ready_event_list));
  names.emplace_back(std::move(name));
  callbacks.emplace_back(std::move(callback));

  return EnqueueTensorAllgathers(contexts, tensors, ready_event_lists, names,
                                 device, callbacks, process_set_id);
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status
EnqueueTensorAllgathers(std::vector<std::shared_ptr<OpContext>>& contexts,
                        std::vector<std::shared_ptr<Tensor>>& tensors,
                        std::vector<ReadyEventList>& ready_event_lists,
                        std::vector<std::string>& names, int device,
                        std::vector<StatusCallback>& callbacks,
                        int32_t process_set_id) {
  if (horovod_global.cpu_operation == LibType::CCL && process_set_id > 0 &&
      device == CPU_DEVICE_ID) {
    return Status::InvalidArgument(
        "Process sets are not supported yet with oneCCL operations.");
  }
  if (!horovod_global.process_set_table.Contains(process_set_id)) {
    return Status::InvalidArgument("Allgather: Process set provided does not "
                                   "exist, or has not been registered.");
  }
  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  auto& process_set = horovod_global.process_set_table.Get(process_set_id);

  if (!process_set.IsCurrentProcessIncluded()) {
    return Status::InvalidArgument(
        "Allgather: Rank " +
        std::to_string(horovod_global.global_controller->GetRank()) +
        " is not a member of the provided process set.");
  }

  std::vector<Request> messages;
  std::vector<TensorTableEntry> entries;
  messages.reserve(tensors.size());
  entries.reserve(tensors.size());

  for (int n = 0; n < (int)tensors.size(); ++n) {
    Request message;
    message.set_request_rank(process_set.controller->GetRank());
    message.set_tensor_name(names[n]);
    message.set_tensor_type(tensors[n]->dtype());
    message.set_device(device);
    message.set_request_type(Request::ALLGATHER);
    message.set_tensor_shape(tensors[n]->shape().to_vector());

    messages.push_back(std::move(message));

    TensorTableEntry e;
    e.tensor_name = names[n];
    e.context = contexts[n];
    e.tensor = tensors[n];
    e.output_index = n;
    e.process_set_id = process_set_id;
    e.ready_event_list = std::move(ready_event_lists[n]);
    e.device = device;
    e.callback = std::move(callbacks[n]);

    entries.push_back(std::move(e));
  }

  // Start appropriate NVTX range
  if (tensors.size() == 1) {
    auto& e = entries[0];
    e.nvtx_op_range.Start(RegisteredNvtxOp::HorovodAllgather, e.tensor->size());
  } else {
    auto total_size =
        std::accumulate(entries.begin(), entries.end(), 0ll,
                        [](int64_t size_sum, const TensorTableEntry& e) {
                          return size_sum + e.tensor->size();
                        });
    SharedNvtxOpRange range;
    range.Start(RegisteredNvtxOp::HorovodGroupedAllgather, total_size);
    for (auto& e : entries) {
      e.nvtx_op_range = range;
    }
  }

  std::string tensors_enqueued;
  for (const auto& n : names) {
    tensors_enqueued += n + "; ";
  }
  LOG(TRACE, horovod_global.global_controller->GetRank())
      << "Enqueued tensors for Allgather: " << tensors_enqueued;

  // Only create groups larger than 1 tensor, unless disable_group_fusion is
  // requested. In that case, even single tensor groups are created to enforce
  // disabling fusion.
  if (tensors.size() > 1 || horovod_global.disable_group_fusion) {
    auto group_id = process_set.group_table.RegisterGroup(std::move(names));
    for (auto& message : messages) {
      message.set_group_id(group_id);
    }
  }

  Status status =
      process_set.tensor_queue.AddToTensorQueueMulti(entries, messages);

  return status;
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueTensorBroadcast(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output, int root_rank,
                              ReadyEventList ready_event_list,
                              const std::string& name, const int device,
                              StatusCallback callback, int32_t process_set_id) {
  if (horovod_global.cpu_operation == LibType::CCL && process_set_id > 0 &&
      device == CPU_DEVICE_ID) {
    return Status::InvalidArgument(
        "Process sets are not supported yet with oneCCL operations.");
  }
  if (!horovod_global.process_set_table.Contains(process_set_id)) {
    return Status::InvalidArgument("Broadcast: Process set provided does not "
                                   "exist, or has not been registered.");
  }
  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  auto& process_set = horovod_global.process_set_table.Get(process_set_id);

  int root_rank_in_process_set;
  try {
    root_rank_in_process_set =
        process_set.controller->GetGlobalRankToControllerRank().at(root_rank);
  } catch (const std::out_of_range& e) {
    return Status::InvalidArgument("broadcast received invalid root rank " +
                                   std::to_string(root_rank) +
                                   " for provided process set");
  }

  Request message;
  message.set_request_rank(process_set.controller->GetRank());
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_root_rank(root_rank_in_process_set);
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
  e.process_set_id = process_set_id;
  e.root_rank = root_rank_in_process_set;
  e.ready_event_list = ready_event_list;
  e.device = device;
  e.callback = callback;
  e.nvtx_op_range.Start(RegisteredNvtxOp::HorovodBroadcast, e.tensor->size());

  if (!process_set.IsCurrentProcessIncluded()) {
    return Status::InvalidArgument(
        "Broadcast: Rank " +
        std::to_string(horovod_global.global_controller->GetRank()) +
        " is not a member of the provided process set.");
  }

  Status status = process_set.tensor_queue.AddToTensorQueue(e, message);
  if (status.ok()) {
    LOG(TRACE, horovod_global.global_controller->GetRank())
        << "Enqueued " << name;
  }
  return status;
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueTensorReducescatter(std::shared_ptr<OpContext> context,
                                  std::shared_ptr<Tensor> tensor,
                                  ReadyEventList ready_event_list,
                                  const std::string& name, const int device,
                                  StatusCallback callback, ReduceOp reduce_op,
                                  int32_t process_set_id) {
  // Wrap inputs in std::vector and pass onto multi tensor implementation
  std::vector<std::shared_ptr<OpContext>> contexts;
  std::vector<std::shared_ptr<Tensor>> tensors;
  std::vector<ReadyEventList> ready_event_lists;
  std::vector<std::string> names;
  std::vector<StatusCallback> callbacks;

  contexts.emplace_back(std::move(context));
  tensors.emplace_back(std::move(tensor));
  ready_event_lists.emplace_back(std::move(ready_event_list));
  names.emplace_back(std::move(name));
  callbacks.emplace_back(std::move(callback));

  return EnqueueTensorReducescatters(contexts, tensors, ready_event_lists,
                                     names, device, callbacks, reduce_op,
                                     process_set_id);
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status
EnqueueTensorReducescatters(std::vector<std::shared_ptr<OpContext>>& contexts,
                            std::vector<std::shared_ptr<Tensor>>& tensors,
                            std::vector<ReadyEventList>& ready_event_lists,
                            std::vector<std::string>& names, int device,
                            std::vector<StatusCallback>& callbacks,
                            ReduceOp reduce_op, int32_t process_set_id) {
  if (horovod_global.cpu_operation == LibType::CCL && device == CPU_DEVICE_ID) {
    return Status::InvalidArgument(
        "Reducescatter is not supported yet with oneCCL operations.");
  }
  if (!horovod_global.process_set_table.Contains(process_set_id)) {
    return Status::InvalidArgument(
        "Reducescatter: Process set provided does not "
        "exist, or has not been registered.");
  }

  if (reduce_op != ReduceOp::SUM) {
    // Note: AVERAGE is supported by enqueuing SUM and performing divide at the
    // framework level.
    LOG(ERROR, horovod_global.global_controller->GetRank())
        << "Reducescatter currently only supports SUM.";
    return Status::Aborted("Reducescatter currently only supports SUM.");
  }
  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  auto& process_set = horovod_global.process_set_table.Get(process_set_id);

  if (!process_set.IsCurrentProcessIncluded()) {
    return Status::InvalidArgument(
        "Reducescatter: Rank " +
        std::to_string(horovod_global.global_controller->GetRank()) +
        " is not a member of the provided process set.");
  }

  std::vector<Request> messages;
  std::vector<TensorTableEntry> entries;
  messages.reserve(tensors.size());
  entries.reserve(tensors.size());

  for (int n = 0; n < (int)tensors.size(); ++n) {
    Request message;
    message.set_request_rank(process_set.controller->GetRank());
    message.set_tensor_name(names[n]);
    message.set_tensor_type(tensors[n]->dtype());
    message.set_device(device);
    message.set_request_type(Request::REDUCESCATTER);
    message.set_tensor_shape(tensors[n]->shape().to_vector());
    messages.push_back(std::move(message));

    TensorTableEntry e;
    e.tensor_name = names[n];
    e.context = std::move(contexts[n]);
    e.tensor = tensors[n];
    e.output_index = n;
    e.process_set_id = process_set_id;
    e.ready_event_list = std::move(ready_event_lists[n]);
    e.device = device;
    e.callback = std::move(callbacks[n]);

    entries.push_back(std::move(e));
  }

  // Start appropriate NVTX range
  if (tensors.size() == 1) {
    auto& e = entries[0];
    e.nvtx_op_range.Start(RegisteredNvtxOp::HorovodReducescatter,
                          e.tensor->size());
  } else {
    auto total_size =
        std::accumulate(entries.begin(), entries.end(), 0ll,
                        [](int64_t size_sum, const TensorTableEntry& e) {
                          return size_sum + e.tensor->size();
                        });
    SharedNvtxOpRange range;
    range.Start(RegisteredNvtxOp::HorovodGroupedReducescatter, total_size);
    for (auto& e : entries) {
      e.nvtx_op_range = range;
    }
  }

  std::string tensors_enqueued;
  for (const auto& n : names) {
    tensors_enqueued += n + "; ";
  }
  LOG(TRACE, horovod_global.global_controller->GetRank())
      << "Enqueued tensors for Reducescatter: " << tensors_enqueued;

  // Only create groups larger than 1 tensor, unless disable_group_fusion is
  // requested. In that case, even single tensor groups are created to enforce
  // disabling fusion.
  if (tensors.size() > 1 || horovod_global.disable_group_fusion) {
    auto group_id = process_set.group_table.RegisterGroup(std::move(names));
    for (auto& message : messages) {
      message.set_group_id(group_id);
    }
  }

  Status status =
      process_set.tensor_queue.AddToTensorQueueMulti(entries, messages);

  return status;
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueTensorAlltoall(std::shared_ptr<OpContext> context,
                             std::shared_ptr<Tensor> tensor,
                             std::shared_ptr<Tensor> splits,
                             ReadyEventList ready_event_list,
                             const std::string& name, const int device,
                             StatusCallback callback, int32_t process_set_id) {
  if (horovod_global.cpu_operation == LibType::CCL && process_set_id > 0 &&
      device == CPU_DEVICE_ID) {
    return Status::InvalidArgument(
        "Process sets are not supported yet with oneCCL operations.");
  }
  if (!horovod_global.process_set_table.Contains(process_set_id)) {
    return Status::InvalidArgument("Alltoall: Process set provided does not "
                                   "exist, or has not been registered.");
  }
  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  auto& process_set = horovod_global.process_set_table.Get(process_set_id);

  // Check arguments
  if (splits->shape().dims() > 1) {
    return Status::InvalidArgument("alltoall expects a 1D splits tensor");
  }
  if (splits->dtype() != HOROVOD_INT32) {
    return Status::InvalidArgument(
        "alltoall expects splits to contain 32-bit integer elements.");
  }

  Request message;
  message.set_request_rank(process_set.controller->GetRank());
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  message.set_request_type(Request::ALLTOALL);
  for (int i = 0; i < tensor->shape().dims(); ++i) {
    message.add_tensor_shape((int64_t)tensor->shape().dim_size(i));
  }

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.tensor = tensor;
  e.process_set_id = process_set_id;
  e.ready_event_list = ready_event_list;
  e.device = device;
  e.callback = callback;
  e.nvtx_op_range.Start(RegisteredNvtxOp::HorovodAlltoall, e.tensor->size());

  int64_t splits_first_dim = splits->shape().dim_size(0);
  int64_t tensor_first_dim = tensor->shape().dim_size(0);
  int world_size = process_set.controller->GetSize();
  if (!process_set.IsCurrentProcessIncluded()) {
    return Status::InvalidArgument(
        "Alltoall: Rank " +
        std::to_string(horovod_global.global_controller->GetRank()) +
        " is not a member of the provided process set.");
  } else if (splits_first_dim == world_size) {
    auto splits_data = static_cast<const int32_t*>(splits->data());
    auto sum = std::accumulate(splits_data, splits_data + splits_first_dim, 0);
    if (sum > tensor_first_dim) {
      return Status::InvalidArgument("Sum of splits entries is greater than "
                                     "the first dimension of tensor.");
    }
    e.splits.assign(splits_data, splits_data + splits->shape().num_elements());
  } else if (splits_first_dim == 0) {
    if (tensor_first_dim % world_size != 0) {
      return Status::InvalidArgument(
          "splits not provided, but first dimension of tensor is not an even "
          "multiple of the number of workers.");
    }
    e.splits.resize(world_size, tensor_first_dim / world_size);
  } else {
    return Status::InvalidArgument(
        "Number of entries in splits does not equal number of workers.");
  }

  Status status = process_set.tensor_queue.AddToTensorQueue(e, message);
  if (status.ok()) {
    LOG(TRACE, horovod_global.global_controller->GetRank())
        << "Enqueued " << name;
  }
  return status;
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueJoin(std::shared_ptr<OpContext> context,
                   std::shared_ptr<Tensor> output_last_joined_rank,
                   ReadyEventList ready_event_list, const std::string& name,
                   const int device, StatusCallback callback,
                   int32_t process_set_id) {
  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  auto& process_set = horovod_global.process_set_table.Get(process_set_id);

  Request message;
  message.set_request_rank(process_set.controller->GetRank());
  message.set_device(device);
  message.set_request_type(Request::JOIN);

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.output = output_last_joined_rank;
  e.process_set_id = process_set_id;
  e.ready_event_list = ready_event_list;
  e.device = device;
  e.callback = callback;

  Status status = process_set.tensor_queue.AddToTensorQueue(e, message);
  if (status.ok()) {
    LOG(TRACE, horovod_global.global_controller->GetRank())
        << "Enqueued " << name;
  }
  return status;
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueBarrier(StatusCallback callback, int32_t process_set_id) {
  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  auto& process_set = horovod_global.process_set_table.Get(process_set_id);

  if (!process_set.IsCurrentProcessIncluded()) {
    return Status::InvalidArgument(
        "Barrier: Rank " +
        std::to_string(horovod_global.global_controller->GetRank()) +
        " is not a member of the provided process set.");
  }

  Request message;
  // Barrier doesn't need a tensor, we set an arbitrary name for tracing
  // purposes.
  message.set_tensor_name(BARRIER_TENSOR_NAME);
  message.set_request_rank(process_set.controller->GetRank());
  message.set_request_type(Request::BARRIER);

  TensorTableEntry e;
  e.tensor_name = BARRIER_TENSOR_NAME;
  e.process_set_id = process_set_id;
  e.callback = callback;

  Status status = process_set.tensor_queue.AddToTensorQueue(e, message);
  if (status.ok()) {
    LOG(TRACE, horovod_global.global_controller->GetRank())
        << "Enqueued barrier op";
  }

  return status;
}

} // namespace common
} // namespace horovod
