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
#include "ops/operation_manager.h"
#include "parameter_manager.h"
#include "timeline.h"
#include "utils/env_parser.h"

#if HAVE_MPI
#define OMPI_SKIP_MPICXX
#include "mpi.h"
#include "mpi/mpi_context.h"
#include "mpi/mpi_controller.h"
#include "ops/mpi_operations.h"
#include "ops/adasum_mpi_operations.h"
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

// All the Horovod state that must be stored globally per-process.
HorovodGlobalState horovod_global;

#if HAVE_MPI
MPIContext mpi_context;
#endif

#if HAVE_GLOO
GlooContext gloo_context;
#endif

#if HAVE_GPU
GPUContext gpu_context;
#endif

#if HAVE_NCCL
NCCLContext nccl_context;
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
  std::vector<std::shared_ptr<AllreduceOp>> adasum_ops;

#if HAVE_MPI && HAVE_GPU
  if (mpi_context.IsEnabled()) {
#if HOROVOD_GPU_ALLREDUCE == 'M'
    allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(
        new MPI_GPUAllreduce(&mpi_context, &gpu_context, &state)));

#elif HAVE_NCCL && HOROVOD_GPU_ALLREDUCE == 'N'
    adasum_ops.push_back(std::shared_ptr<AllreduceOp>(new AdasumGpuAllreduceOp(&mpi_context, &nccl_context, &gpu_context, &state)));

    allreduce_ops.push_back(
        std::shared_ptr<AllreduceOp>(new NCCLHierarchicalAllreduce(
            &nccl_context, &mpi_context, &gpu_context, &state)));

#elif HAVE_DDL && HOROVOD_GPU_ALLREDUCE == 'D'
    allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(
        new DDLAllreduce(&ddl_context, &gpu_context, &state)));
#endif

#if HOROVOD_GPU_ALLGATHER == 'M'
    allgather_ops.push_back(std::shared_ptr<AllgatherOp>(
        new MPI_GPUAllgather(&mpi_context, &gpu_context, &state)));
#endif
    allgather_ops.push_back(std::shared_ptr<AllgatherOp>(
        new MPIHierarchicalAllgather(&mpi_context, &state)));
  }
#endif

#if HAVE_NCCL && HOROVOD_GPU_ALLREDUCE == 'N'
  allreduce_ops.push_back(std::shared_ptr<AllreduceOp>(
      new NCCLAllreduce(&nccl_context, &gpu_context, &state)));
#endif

#if HAVE_NCCL && HOROVOD_GPU_BROADCAST == 'N'
    broadcast_ops.push_back(
        std::shared_ptr<BroadcastOp>(new NCCLBroadcast(&nccl_context, &gpu_context, &state)));
#endif

#if HAVE_GLOO
  if (gloo_context.IsEnabled()) {
    allreduce_ops.push_back(
        std::shared_ptr<AllreduceOp>(new GlooAllreduce(&gloo_context, &state)));
    allgather_ops.push_back(
        std::shared_ptr<AllgatherOp>(new GlooAllgather(&gloo_context, &state)));
    broadcast_ops.push_back(
        std::shared_ptr<BroadcastOp>(new GlooBroadcast(&gloo_context, &state)));
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
  }
#endif

#if HAVE_MPI
  if (mpi_context.IsEnabled()){
    adasum_ops.push_back(
        std::shared_ptr<AllreduceOp>(new AdasumMPIAllreduceOp(&mpi_context, &state)));
    allreduce_ops.push_back(
        std::shared_ptr<AllreduceOp>(new MPIAllreduce(&mpi_context,&state)));
    allgather_ops.push_back(
        std::shared_ptr<AllgatherOp>(new MPIAllgather(&mpi_context, &state)));
    broadcast_ops.push_back(
        std::shared_ptr<BroadcastOp>(new MPIBroadcast(&mpi_context, &state)));
  }
#endif

  std::shared_ptr<JoinOp> join_op(new JoinOp(&state));
  std::shared_ptr<ErrorOp> error_op(new ErrorOp(&state));

  return new OperationManager(&state.parameter_manager, allreduce_ops,
                              allgather_ops, broadcast_ops, join_op, adasum_ops, error_op);
}

// Process a Response by doing a reduction, a gather, a broadcast, or
// raising an error.
void PerformOperation(Response response, HorovodGlobalState& state) {
  std::vector<TensorTableEntry> entries;
  auto& timeline = horovod_global.timeline;
  if (response.response_type() != Response::JOIN) {
    horovod_global.tensor_queue.GetTensorEntriesFromResponse(response, entries,
                                                             state.joined);

    for (auto& e : entries) {
      timeline.Start(e.tensor_name, response.response_type());
    }

    if (entries.size() > 1) {
      auto first_entry = entries[0];
      // Note: it is OK for different entries to come from different frameworks
      // since buffer allocated here is guaranteed to survive at least till the
      // end of this operation.
      Status status = horovod_global.fusion_buffer.InitializeBuffer(
          horovod_global.controller->TensorFusionThresholdBytes(),
          first_entry.device, first_entry.context,
          horovod_global.current_nccl_stream,
          [&]() { timeline.ActivityStartAll(entries, INIT_FUSION_BUFFER); },
          [&]() { timeline.ActivityEndAll(entries); });
      if (!status.ok()) {
        for (auto& e : entries) {
          timeline.End(e.tensor_name, nullptr);
          // Callback can be null if the rank sent Join request.
          if (e.callback != nullptr) {
            e.callback(status);
          }
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
      // Callback can be null if the rank sent Join request.
      if (e.callback != nullptr) {
        e.callback(status);
      }
    }
  }
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
  // Set background thread affinity
  auto horovod_thread_affinity = std::getenv(HOROVOD_THREAD_AFFINITY);
#if HAVE_CCL
  if (horovod_thread_affinity != nullptr) {
    horovod_thread_affinity = std::getenv(HOROVOD_CCL_BGT_AFFINITY);
  }
#endif
  if (horovod_thread_affinity != nullptr) {
      int core = std::strtol(horovod_thread_affinity, nullptr, 10);
      server_affinity_set(core);
  }
#if HAVE_CCL
  // Initialize ccl context
  if (state.cpu_operation == LibType::CCL) {
    ccl_context.Init();
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
  mpi_context.Initialize(state.controller->GetRanks(), mpi_ctx_manager);
#endif

#if HAVE_GLOO
#if HAVE_MPI
    if (mpi_context.IsEnabled()) {
      // Initialize gloo context if mpi context is available
      gloo_context.InitializeFromMPI(mpi_context, ParseGlooIface());
    }
    else
#endif
    {
      gloo_context.Initialize(ParseGlooIface());
    }
#endif

  // Initialize controller
  state.controller->Initialize();

  bool is_coordinator = state.controller->IsCoordinator();
  bool is_homogeneous = state.controller->IsHomogeneous();
  int size = state.controller->GetSize();
  int local_size = state.controller->GetLocalSize();

#if HAVE_GPU
  // Set number of GPU streams to use
  auto horovod_num_nccl_streams =
      std::getenv(HOROVOD_NUM_NCCL_STREAMS);
  if (horovod_num_nccl_streams != nullptr &&
      std::stol(horovod_num_nccl_streams, nullptr, 10) > 0) {
    state.num_nccl_streams = std::atoi(horovod_num_nccl_streams);
  }

#if HAVE_NCCL
  nccl_context.nccl_comms.resize(state.num_nccl_streams);
#endif
  gpu_context.streams.resize(state.num_nccl_streams);

  // Create finalizer thread pool (one thread per stream)
  gpu_context.finalizer_thread_pool.create(state.num_nccl_streams);
#endif

  // Open the timeline file on coordinator.
  auto horovod_timeline = std::getenv(HOROVOD_TIMELINE);
  if (is_coordinator && horovod_timeline != nullptr) {
    state.timeline.Initialize(std::string(horovod_timeline),
                              static_cast<unsigned int>(size));
  }
  if (horovod_timeline != nullptr) {
    state.controller->SetTimelineEnabled(true);
  }

  ParseStallInspectorFromEnv(state.controller->GetStallInspector());

  SetBoolFromEnv(HOROVOD_TIMELINE_MARK_CYCLES, state.mark_cycles_in_timeline,
                 true);

  // Override Tensor Fusion threshold, if it's set.
  state.parameter_manager.SetTensorFusionThresholdBytes(64 * 1024 * 1024);
  auto horovod_fusion_threshold = std::getenv(HOROVOD_FUSION_THRESHOLD);
  if (horovod_fusion_threshold != nullptr) {
    int64_t threshold = std::strtol(horovod_fusion_threshold, nullptr, 10);
    state.parameter_manager.SetTensorFusionThresholdBytes(threshold, true);
  }

  // Override the cycle time.
  state.parameter_manager.SetCycleTimeMs(5);
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
  state.response_cache.set_capacity(
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

  // Enable auto-tuning.
  auto horovod_autotune = std::getenv(HOROVOD_AUTOTUNE);
  if (horovod_autotune != nullptr &&
      std::strtol(horovod_autotune, nullptr, 10) > 0) {
    auto horovod_autotune_log = std::getenv(HOROVOD_AUTOTUNE_LOG);
    state.parameter_manager.Initialize(state.controller->GetRank(), RANK_ZERO,
                                       horovod_autotune_log != nullptr
                                           ? std::string(horovod_autotune_log)
                                           : "");
    state.parameter_manager.SetAutoTuning(true);
  }

  // Set chunk size for MPI based Adasum allreduce algorithms
  auto horovod_adasum_mpi_chunk_size = std::getenv(HOROVOD_ADASUM_MPI_CHUNK_SIZE);
  if (horovod_adasum_mpi_chunk_size != nullptr) {
    state.adasum_mpi_chunk_size = std::strtol(horovod_adasum_mpi_chunk_size, nullptr, 10);
  }

  op_manager.reset(CreateOperationManager(state));

  // Signal that initialization is completed.
  state.initialization_done = true;
  LOG(INFO, horovod_global.controller->GetRank()) << "Horovod Initialized";

  // Iterate until shutdown.
  while (RunLoopOnce(state))
    ;

    // Finalize all contexts
#if HAVE_NCCL
  nccl_context.ShutDown();
#endif

#if HAVE_GLOO
  gloo_context.Finalize();
#endif

  LOG(DEBUG, horovod_global.controller->GetRank()) << "Shutting down background thread";

  // Signal that shutdown has been requested.
  state.shut_down = true;

  // Notify all outstanding operations that Horovod has been shut down
  // and finalize tensor queue.
  std::vector<StatusCallback> callbacks;
  horovod_global.tensor_queue.FinalizeTensorQueue(callbacks);
  for (auto& cb : callbacks) {
    cb(SHUT_DOWN_ERROR);
  }

#if HAVE_GPU
  gpu_context.Finalize();
#endif

#if HAVE_MPI
  mpi_context.Finalize(mpi_ctx_manager);
#endif

#if HAVE_CCL
  if (state.cpu_operation == LibType::CCL){
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

  auto response_list =
      state.controller->ComputeResponseList(horovod_global.shut_down, state);

  // Get tensor name and size data for autotuning.
  int64_t total_tensor_size = 0;
  std::vector<std::string> tensor_names;
  if (state.parameter_manager.IsAutoTuning()) {
    total_tensor_size = horovod_global.tensor_queue.GetTensorDataForAutotuner(
        response_list, tensor_names);
  }

  // Perform the collective operation. All nodes should end up performing
  // the same operation.
  int rank = state.controller->GetRank();
  for (auto& response : response_list.responses()) {
    LOG(TRACE, rank) << "Performing " << response.tensor_names_string();
    LOG(DEBUG, rank) << "Processing " << response.tensor_names().size()
                     << " tensors";
    PerformOperation(response, horovod_global);
    LOG(TRACE, rank) << "Finished performing "
                     << response.tensor_names_string();
  }

  if (state.parameter_manager.IsAutoTuning()) {
    bool should_sync =
        state.parameter_manager.Update(tensor_names, total_tensor_size);

    if (should_sync) {
      state.controller->SynchronizeParameters();
    }
  }

  return !response_list.shutdown();
}

// Start Horovod background thread. Ensure that this is
// only done once no matter how many times this function is called.
void InitializeHorovodOnce(const int* ranks, int nranks) {
  // Ensure background thread is only started once.
  if (!horovod_global.initialize_flag.test_and_set()) {
    horovod_global.control_operation = ParseControllerOpsFromEnv();
    horovod_global.cpu_operation = ParseCPUOpsFromEnv();
#if HAVE_MPI
    // Enable mpi is it's used either in cpu data transfer or controller
    if (horovod_global.cpu_operation == LibType::MPI ||
        horovod_global.control_operation == LibType::MPI) {
      mpi_context.Enable();
    }

    if (horovod_global.control_operation == LibType::MPI){
      horovod_global.controller.reset(new MPIController(
          horovod_global.response_cache,
          horovod_global.tensor_queue, horovod_global.timeline,
          horovod_global.parameter_manager, mpi_context));
      horovod_global.controller->SetRanks(ranks, nranks);
    }
#endif

#if HAVE_GLOO
    // Enable gloo is it's used either in cpu data transfer or controller
    if (horovod_global.cpu_operation == LibType::GLOO ||
        horovod_global.control_operation == LibType::GLOO) {
      gloo_context.Enable();
    }

    if (horovod_global.control_operation == LibType::GLOO) {
      horovod_global.controller.reset(new GlooController(
          horovod_global.response_cache,
          horovod_global.tensor_queue, horovod_global.timeline,
          horovod_global.parameter_manager, gloo_context));
    }
#endif
    // Reset initialization flag
    horovod_global.initialization_done = false;
    horovod_global.background_thread = std::thread(
        BackgroundThreadLoop, std::ref(horovod_global));
  }

  // Wait to ensure that the background thread has finished initializing MPI.
  while (!horovod_global.initialization_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  LOG(DEBUG) << "Background thread init done";
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

#if HAVE_MPI
void horovod_init_comm(MPI_Comm comm) {
  MPI_Comm_dup(comm, &mpi_context.mpi_comm);
  InitializeHorovodOnce(nullptr, 0);
}
#endif

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
  return horovod_global.controller->GetRank();
}

int horovod_local_rank() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.controller->GetLocalRank();
}

int horovod_size() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.controller->GetSize();
}

int horovod_local_size() {
  if (!horovod_global.initialization_done) {
    return -1;
  }
  return horovod_global.controller->GetLocalSize();
}

bool horovod_is_homogeneous() {
  return horovod_global.controller->IsHomogeneous();
}

int horovod_mpi_threads_supported() {
  if (!horovod_global.initialization_done) {
    return -1;
  }

#if HAVE_MPI
  auto mpiController =
      std::dynamic_pointer_cast<MPIController>(horovod_global.controller);
  return mpiController->IsMpiThreadsSupported() ? 1 : 0;
#endif

  return -1;
}

bool horovod_mpi_enabled() {
#if HAVE_MPI
  return mpi_context.IsEnabled();
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
  return gloo_context.IsEnabled();
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

bool horovod_nccl_built() {
#if HAVE_NCCL
  return true;
#else
  return false;
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

int horovod_reduce_op_average() {
  return ReduceOp::AVERAGE;
}

int horovod_reduce_op_sum() {
  return ReduceOp::SUM;
}

int horovod_reduce_op_adasum() {
  return ReduceOp::ADASUM;
}

}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueTensorAllreduce(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback,
                              ReduceOp reduce_op) {
  Status status;

  // AVERAGE should be taken care of in the framework layer. Equeuing it here directly is not allowed.
  // For example of how to deal with op=hvd.Average in framework layer, please refer to function
  // `def _allreduce_async(tensor, output, name, op)` in
  // horovod/horovod/torch/mpi_ops.py
  if (reduce_op == ReduceOp::AVERAGE) {
    LOG(ERROR, horovod_global.controller->GetRank()) << "Enqueuing AVERAGE allreduce is not allowed.";
    return status.Aborted("AVERAGE not allowed.");
  }
  Request message;
  message.set_request_rank(horovod_global.controller->GetRank());
  message.set_tensor_name(name);
  message.set_tensor_type(tensor->dtype());
  message.set_device(device);
  
  if (reduce_op == ReduceOp::ADASUM) {
    message.set_request_type(Request::ADASUM);
  } else {
    message.set_request_type(Request::ALLREDUCE);
  }
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

  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  status = horovod_global.tensor_queue.AddToTensorQueue(e, message);
  if (status.ok()) {
    LOG(TRACE, horovod_global.controller->GetRank()) << "Enqueued " << name;
  }
  return status;
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueTensorAllgather(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback) {
  Request message;
  message.set_request_rank(horovod_global.controller->GetRank());
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

  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = horovod_global.tensor_queue.AddToTensorQueue(e, message);
  if (status.ok()) {
    LOG(TRACE, horovod_global.controller->GetRank()) << "Enqueued " << name;
  }
  return status;
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueTensorBroadcast(std::shared_ptr<OpContext> context,
                              std::shared_ptr<Tensor> tensor,
                              std::shared_ptr<Tensor> output, int root_rank,
                              std::shared_ptr<ReadyEvent> ready_event,
                              const std::string name, const int device,
                              StatusCallback callback) {
  Request message;
  message.set_request_rank(horovod_global.controller->GetRank());
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

  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = horovod_global.tensor_queue.AddToTensorQueue(e, message);
  if (status.ok()) {
    LOG(TRACE, horovod_global.controller->GetRank()) << "Enqueued " << name;
  }
  return status;
}

// Contexts and controller must be initialized and the background thread
// must be running before this function is called.
Status EnqueueJoin(std::shared_ptr<OpContext> context,
                   std::shared_ptr<ReadyEvent> ready_event,
                   const std::string name, const int device,
                   StatusCallback callback) {
  Request message;
  message.set_request_rank(horovod_global.controller->GetRank());
  message.set_device(device);
  message.set_request_type(Request::JOIN);

  TensorTableEntry e;
  e.tensor_name = name;
  e.context = context;
  e.ready_event = ready_event;
  e.device = device;
  e.callback = callback;

  if (horovod_global.shut_down) {
    return SHUT_DOWN_ERROR;
  }
  Status status = horovod_global.tensor_queue.AddToTensorQueue(e, message);
  if (status.ok()) {
    LOG(TRACE, horovod_global.controller->GetRank()) << "Enqueued " << name;
  }
  return status;
}

} // namespace common
} // namespace horovod
