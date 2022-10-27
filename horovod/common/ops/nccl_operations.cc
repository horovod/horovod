// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "nccl_operations.h"

#if HAVE_MPI
#include "../mpi/mpi_context.h"
#endif

#if HAVE_CUDA
#include "cuda/cuda_kernels.h"
#endif
#if HAVE_ROCM
#include "rocm/hip_kernels.h"
#endif

namespace horovod {
namespace common {

ncclDataType_t GetNCCLDataType(const std::shared_ptr<Tensor> tensor) {
  switch (tensor->dtype()) {
  case HOROVOD_UINT8:
    return ncclUint8;
  case HOROVOD_INT8:
    return ncclInt8;
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

void commDestroyOrAbort(ncclComm_t& nccl_comm, bool elastic) {
  ncclResult_t nccl_async_err;
  auto nccl_err = ncclCommGetAsyncError(nccl_comm, &nccl_async_err);
  if (nccl_err != ncclSuccess) {
    return;
  }
  if (nccl_async_err == ncclSuccess && !elastic) {
    ncclCommDestroy(nccl_comm);
  } else {
    ncclCommAbort(nccl_comm);
  }
}

void NCCLContext::ErrorCheck(std::string op_name, ncclResult_t nccl_result,
                             ncclComm_t& nccl_comm) {
  if (nccl_result != ncclSuccess) {
    ncclCommAbort(nccl_comm);
    throw std::logic_error(std::string(op_name) +
                           " failed: " + ncclGetErrorString(nccl_result));
  }
}

void NCCLContext::ShutDown() {
  for (auto it = nccl_comms.begin(); it != nccl_comms.end(); ++it) {
    for (auto entry = it->begin(); entry != it->end(); ++entry) {
      commDestroyOrAbort(entry->second, elastic);
    }
  }
  nccl_comms.clear();
}

void NCCLOpContext::InitNCCLComm(const std::vector<TensorTableEntry>& entries,
                                 const std::vector<int32_t>& nccl_device_map) {
  assert(!entries.empty());
  auto process_set_id = entries[0].process_set_id;
  auto& process_set = global_state_->process_set_table.Get(process_set_id);
  // Ensure NCCL communicator is in the map before executing operation.
  // We need to maintain one map per process_set_id to avoid deadlocks
  // in situations where one process has already built nccl_comm, but
  // another has not done so yet.
  ncclComm_t& nccl_comm =
      nccl_context_
          ->nccl_comms[global_state_->current_nccl_stream]
                      [std::make_tuple(process_set_id, nccl_device_map)];
  if (nccl_comm == nullptr) {
    auto& timeline = global_state_->timeline;
    timeline.ActivityStartAll(entries, INIT_NCCL);

    int nccl_rank, nccl_size;
    Communicator nccl_id_bcast_comm;
    PopulateNCCLCommStrategy(nccl_rank, nccl_size, nccl_id_bcast_comm,
                             process_set);

    ncclUniqueId nccl_id;
    if (nccl_rank == 0) {
      nccl_context_->ErrorCheck("ncclGetUniqueId", ncclGetUniqueId(&nccl_id),
                                nccl_comm);
    }

    process_set.controller->Bcast((void*)&nccl_id, sizeof(nccl_id), 0,
                                  nccl_id_bcast_comm);

    ncclComm_t new_nccl_comm;
    auto nccl_result =
        ncclCommInitRank(&new_nccl_comm, nccl_size, nccl_id, nccl_rank);
    nccl_context_->ErrorCheck("ncclCommInitRank", nccl_result, nccl_comm);
    nccl_comm = new_nccl_comm;

    // Barrier helps NCCL to synchronize after initialization and avoid
    // deadlock that we've been seeing without it.
    process_set.controller->Barrier(Communicator::GLOBAL);
    timeline.ActivityEndAll(entries);
  }

  nccl_comm_ = &nccl_comm;
}

void NCCLOpContext::AsyncErrorCheck() {
  ncclResult_t nccl_async_err;
  auto nccl_err = ncclCommGetAsyncError(*nccl_comm_, &nccl_async_err);
  if (nccl_err != ncclSuccess) {
    throw std::logic_error(std::string("ncclGetAsyncError failed: ") +
                           ncclGetErrorString(nccl_err));
  }

  if (nccl_async_err != ncclSuccess) {
    // do not call ncclCommAbort(*nccl_comm_) from event polling thread to avoid
    // race condition
    throw std::logic_error(std::string("NCCL async error: ") +
                           ncclGetErrorString(nccl_async_err));
  }
}

void NCCLOpContext::PopulateNCCLCommStrategy(int& nccl_rank, int& nccl_size,
                                             Communicator& nccl_id_bcast_comm,
                                             const ProcessSet& process_set) {
  if (communicator_type_ == Communicator::GLOBAL) {
    nccl_rank = process_set.controller->GetRank();
    nccl_size = process_set.controller->GetSize();
  } else if (communicator_type_ == Communicator::LOCAL) {
    nccl_rank = process_set.controller->GetLocalRank();
    nccl_size = process_set.controller->GetLocalSize();
  } else if (communicator_type_ == Communicator::CROSS) {
    nccl_rank = process_set.controller->GetCrossRank();
    nccl_size = process_set.controller->GetCrossSize();
  } else {
    throw std::logic_error("Communicator type " +
                           std::to_string(communicator_type_) +
                           " is not supported in NCCL mode.");
  }
  nccl_id_bcast_comm = communicator_type_;
}

void NCCLAllreduce::WaitForData(std::vector<TensorTableEntry>& entries) {
  if (global_state_->timeline.Initialized()) {
    // If timeline is initialized, need to use normal CPU syncing path
    HorovodOp::WaitForData(entries);
  } else {
    // Push events to set to deduplicate entries
    std::unordered_set<gpuEvent_t> event_set;
    for (auto& e : entries) {
      e.ready_event_list.PushEventsToSet(event_set);
    }
    for (auto& ev : event_set) {
      HVD_GPU_CHECK(gpuStreamWaitEvent(*gpu_op_context_.stream, ev, 0));
    }
  }
}

Status NCCLAllreduce::Execute(std::vector<TensorTableEntry>& entries,
                              const Response& response) {
  auto& first_entry = entries[0];

  gpu_op_context_.InitGPU(entries);
  nccl_op_context_.InitNCCLComm(entries, response.devices());
  gpu_op_context_.InitGPUQueue(entries, response);

  WaitForData(entries);

  ncclRedOp_t ncclOp = ncclSum;
  double prescale_factor = response.prescale_factor();
  double postscale_factor = response.postscale_factor();

  if (response.reduce_op() == ReduceOp::AVERAGE) {
#if !HAVE_ROCM
#ifdef NCCL_AVG_SUPPORTED
    // Use NCCLAvg op in place of postscale_factor
    ncclOp = ncclAvg;
#else
    auto process_set_id = first_entry.process_set_id;
    auto& process_set = global_state_->process_set_table.Get(process_set_id);
    ncclOp = ncclSum;
    // Averaging happens via postscale_factor
    postscale_factor /= process_set.controller->GetSize();
#endif
#else
    throw std::logic_error("AVERAGE not allowed.");
#endif
  } else if (response.reduce_op() == ReduceOp::SUM) {
    ncclOp = ncclSum;
  } else if (response.reduce_op() == ReduceOp::MIN) {
    ncclOp = ncclMin;
  } else if (response.reduce_op() == ReduceOp::MAX) {
    ncclOp = ncclMax;
  } else if (response.reduce_op() == ReduceOp::PRODUCT) {
    ncclOp = ncclProd;
  } else {
    throw std::logic_error("Reduction op type not supported.");
  }

  const void* fused_input_data;
  void* buffer_data;
  size_t buffer_len;

  // Copy (and possibly scale) tensors into the fusion buffer.
  if (entries.size() > 1) {
    ScaleMemcpyInFusionBuffer(entries, fused_input_data, buffer_data,
                              buffer_len, prescale_factor);
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
                                MEMCPY_IN_FUSION_BUFFER,
                                *gpu_op_context_.stream);
    }
  } else {
    fused_input_data = first_entry.tensor->data();
    buffer_data = (void*)first_entry.output->data();
    buffer_len = (size_t)first_entry.output->size();
    int64_t num_elements =
        buffer_len / DataType_Size(first_entry.tensor->dtype());
    if (prescale_factor != 1.0) {
      // Execute prescaling op
      ScaleBuffer(prescale_factor, entries, fused_input_data,
                  buffer_data, num_elements);
      fused_input_data = buffer_data; // for unfused, scale is done out of place
    }
  }

  // Do allreduce.
  int64_t num_elements =
      buffer_len / DataType_Size(first_entry.tensor->dtype());
  auto nccl_result =
      ncclAllReduce(fused_input_data, buffer_data, (size_t)num_elements,
                    GetNCCLDataType(first_entry.tensor), ncclOp,
                    *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
  nccl_context_->ErrorCheck("ncclAllReduce", nccl_result,
                            *nccl_op_context_.nccl_comm_);
  if (global_state_->timeline.Initialized()) {
    gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_ALLREDUCE,
                              *gpu_op_context_.stream);
  }

  // Copy (and possible scale) tensors out of the fusion buffer.
  if (entries.size() > 1) {
    ScaleMemcpyOutFusionBuffer(buffer_data, buffer_len,
                               postscale_factor, entries);

    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
                                MEMCPY_OUT_FUSION_BUFFER,
                                *gpu_op_context_.stream);
    }
  } else {
    if (postscale_factor != 1.0) {
      // Execute postscaling op
      ScaleBuffer(postscale_factor, entries, buffer_data,
                  buffer_data, num_elements);
    }
  }

  return gpu_op_context_.FinalizeGPUQueue(
      entries, true, nccl_op_context_.error_check_callback_);
}

#if HAVE_MPI
void NCCLHierarchicalAllreduce::WaitForData(
    std::vector<TensorTableEntry>& entries) {
  if (global_state_->timeline.Initialized()) {
    // If timeline is initialized, need to use normal CPU syncing path
    HorovodOp::WaitForData(entries);
  } else {
    // Push events to set to deduplicate entries
    std::unordered_set<gpuEvent_t> event_set;
    for (auto& e : entries) {
      e.ready_event_list.PushEventsToSet(event_set);
    }
    for (auto& ev : event_set) {
      HVD_GPU_CHECK(gpuStreamWaitEvent(*gpu_op_context_.stream, ev, 0));
    }
  }
}

Status
NCCLHierarchicalAllreduce::Execute(std::vector<TensorTableEntry>& entries,
                                   const Response& response) {
  assert(!entries.empty());
  auto& first_entry = entries[0];
  auto& process_set =
      global_state_->process_set_table.Get(entries[0].process_set_id);
  const auto& mpi_context = process_set.mpi_context;

  // Determine GPU IDs of the devices participating in this communicator.
  std::vector<int32_t> nccl_device_map;
  nccl_device_map.reserve(process_set.controller->GetLocalCommRanks().size());
  for (int rank : process_set.controller->GetLocalCommRanks()) {
    nccl_device_map.push_back(response.devices()[rank]);
  }

  gpu_op_context_.InitGPU(entries);
  nccl_op_context_.InitNCCLComm(entries, nccl_device_map);
  gpu_op_context_.InitGPUQueue(entries, response);

  WaitForData(entries);

  ncclRedOp_t ncclOp = ncclSum;
  MPI_Op mpiOp = mpi_context.GetMPISumOp(first_entry.tensor->dtype());
  double prescale_factor = response.prescale_factor();
  double postscale_factor = response.postscale_factor();

  if (response.reduce_op() == ReduceOp::AVERAGE) {
#if !HAVE_ROCM
    ncclOp = ncclSum;
    mpiOp = mpi_context.GetMPISumOp(first_entry.tensor->dtype());
    // Averaging happens via postscale_factor
    postscale_factor /= process_set.controller->GetSize();
#else
    throw std::logic_error("AVERAGE not allowed.");
#endif
  } else if (response.reduce_op() == ReduceOp::SUM) {
    ncclOp = ncclSum;
    mpiOp = mpi_context.GetMPISumOp(first_entry.tensor->dtype());
  } else if (response.reduce_op() == ReduceOp::MIN) {
    ncclOp = ncclMin;
    mpiOp = mpi_context.GetMPIMinOp(first_entry.tensor->dtype());
  } else if (response.reduce_op() == ReduceOp::MAX) {
    ncclOp = ncclMax;
    mpiOp = mpi_context.GetMPIMaxOp(first_entry.tensor->dtype());
  } else if (response.reduce_op() == ReduceOp::PRODUCT) {
    ncclOp = ncclProd;
    mpiOp = mpi_context.GetMPIProdOp(first_entry.tensor->dtype());
  } else {
    throw std::logic_error("Reduction op type not supported.");
  }

  const void* fused_input_data;
  void* buffer_data;
  size_t buffer_len;

  // Copy memory into the fusion buffer.
  if (entries.size() > 1) {
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);

    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
                                MEMCPY_IN_FUSION_BUFFER,
                                *gpu_op_context_.stream);
    }
  } else {
    fused_input_data = first_entry.tensor->data();
    buffer_data = (void*)first_entry.output->data();
    buffer_len = (size_t)first_entry.output->size();
  }

  int64_t num_elements =
      buffer_len / DataType_Size(first_entry.tensor->dtype());

  if (prescale_factor != 1.0) {
    // Execute prescaling op
    ScaleBuffer(prescale_factor, entries, fused_input_data,
                buffer_data, num_elements);
    fused_input_data = buffer_data; // for unfused, scale is done out of place
  }

  // Do allreduce.
  int element_size = mpi_context.GetMPITypeSize(first_entry.tensor->dtype());
  int local_size = process_set.controller->GetLocalSize();
  int local_rank = process_set.controller->GetLocalRank();

  // If cluster is homogeneous and we are using fusion buffer, include
  // dummy elements from the buffer (if necessary) to make sure the data
  // is divisible by local_size. This is always possible since we
  // set the fusion buffer size divisible by local_size.
  if (process_set.controller->IsHomogeneous() && entries.size() > 1) {
    // Making sure the number of elements is divisible by
    // FUSION_BUFFER_ATOMIC_UNIT for improved performance
    int div = local_size * FUSION_BUFFER_ATOMIC_UNIT;
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
      process_set.controller->IsHomogeneous() ? num_elements / local_size : 0;

  size_t buffer_len_per_rank = element_size * num_elements_per_rank;

  void* buffer_data_at_rank_offset =
      (uint8_t*)buffer_data + buffer_len_per_rank * local_rank;

  int64_t num_elements_remaining = process_set.controller->IsHomogeneous()
                                       ? num_elements % local_size
                                       : num_elements;

  size_t buffer_len_remaining = element_size * num_elements_remaining;

  void* buffer_data_remainder =
      (uint8_t*)buffer_data + buffer_len_per_rank * local_size;

  void* fused_input_data_remainder =
      (uint8_t*)fused_input_data + buffer_len_per_rank * local_size;

  int root_rank = process_set.controller->IsHomogeneous() ? local_size - 1 : 0;
  bool is_root_rank = local_rank == root_rank;

  int64_t total_num_elements =
      is_root_rank ? num_elements_per_rank + num_elements_remaining
                   : num_elements_per_rank;
  int64_t total_buffer_len = is_root_rank
                                 ? buffer_len_per_rank + buffer_len_remaining
                                 : buffer_len_per_rank;

  auto& timeline = global_state_->timeline;
  if (num_elements_per_rank > 0) {
    auto nccl_result = ncclReduceScatter(
        fused_input_data, buffer_data_at_rank_offset,
        (size_t)num_elements_per_rank, GetNCCLDataType(first_entry.tensor),
        ncclOp, *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
    nccl_context_->ErrorCheck("ncclReduceScatter", nccl_result,
                              *nccl_op_context_.nccl_comm_);
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_REDUCESCATTER,
                                *gpu_op_context_.stream);
    }
  }

  if (num_elements_remaining > 0) {
    // Reduce the remaining data at local_size-1 to append to
    // existing buffer
    auto nccl_result =
        ncclReduce(fused_input_data_remainder, buffer_data_remainder,
                   (size_t)num_elements_remaining,
                   GetNCCLDataType(first_entry.tensor), ncclOp, root_rank,
                   *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
    nccl_context_->ErrorCheck("ncclReduce", nccl_result,
                              *nccl_op_context_.nccl_comm_);
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_REDUCE,
                                *gpu_op_context_.stream);
    }
  }

  if (process_set.controller->IsHomogeneous() || is_root_rank) {
    // cudaHostAlloc is significantly slower than malloc.  Pre-allocating
    // a buffer is not safe since the tensor can be arbitrarily large.
    gpu_op_context_.host_buffer = malloc(total_buffer_len);

    // Synchronize.
    if (global_state_->elastic_enabled) {
      gpu_context_->WaitForEventsElastic(
          gpu_op_context_.event_queue, entries, timeline,
          nccl_op_context_.error_check_callback_);
    } else {
      gpu_context_->WaitForEvents(gpu_op_context_.event_queue, entries,
                                  timeline,
                                  nccl_op_context_.error_check_callback_);
    }

    // According to https://docs.nvidia.com/cuda/cuda-runtime-api/
    // api-sync-behavior.html#api-sync-behavior__memcpy-async,
    // cudaMemcpyAsync is synchronous with respect to the host, so we
    // memcpy (effectively) synchronously to generate an accurate timeline
    timeline.ActivityStartAll(entries, MEMCPY_IN_HOST_BUFFER);
    gpu_context_->MemcpyAsyncD2H(gpu_op_context_.host_buffer,
                                 buffer_data_at_rank_offset, total_buffer_len,
                                 *gpu_op_context_.stream);
    timeline.ActivityEndAll(entries);

    timeline.ActivityStartAll(entries, MPI_ALLREDUCE);
    int op = MPI_Allreduce(MPI_IN_PLACE, gpu_op_context_.host_buffer,
                           (int)total_num_elements,
                           mpi_context.GetMPIDataType(first_entry.tensor),
                           mpiOp,
                           mpi_context.GetMPICommunicator(Communicator::CROSS));
    if (op != MPI_SUCCESS) {
      throw std::runtime_error(
          "MPI_Allreduce failed, see MPI output for details.");
    }
    timeline.ActivityEndAll(entries);

    timeline.ActivityStartAll(entries, MEMCPY_OUT_HOST_BUFFER);
    gpu_context_->MemcpyAsyncH2D(buffer_data_at_rank_offset,
                                 gpu_op_context_.host_buffer, total_buffer_len,
                                 *gpu_op_context_.stream);
    timeline.ActivityEndAll(entries);
  }

  if (num_elements_per_rank > 0) {
    nccl_context_->ErrorCheck(
        "ncclAllGather",
        ncclAllGather(buffer_data_at_rank_offset, buffer_data,
                      (size_t)num_elements_per_rank,
                      GetNCCLDataType(first_entry.tensor),
                      *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream),
        *nccl_op_context_.nccl_comm_);
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_ALLGATHER,
                                *gpu_op_context_.stream);
    }
  }
  if (num_elements_remaining > 0) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 2, 12)
    nccl_context_->ErrorCheck(
        "ncclBroadcast",
        ncclBroadcast(buffer_data_remainder, buffer_data_remainder,
                      (size_t)num_elements_remaining,
                      GetNCCLDataType(first_entry.tensor), root_rank,
                      *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream),
        *nccl_op_context_.nccl_comm_);
#else
    nccl_context_->ErrorCheck(
        "ncclBcast",
        ncclBcast(buffer_data_remainder, (size_t)num_elements_remaining,
                  GetNCCLDataType(first_entry.tensor), root_rank,
                  *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream),
        *nccl_op_context_.nccl_comm_);
#endif
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_BCAST,
                                *gpu_op_context_.stream);
    }
  }

  if (postscale_factor != 1.0) {
    // Execute postscaling op
    ScaleBuffer(postscale_factor, entries, buffer_data, buffer_data,
                num_elements);
  }

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    MemcpyOutFusionBuffer(buffer_data, entries);

    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
                                MEMCPY_OUT_FUSION_BUFFER,
                                *gpu_op_context_.stream);
    }
  }

  return gpu_op_context_.FinalizeGPUQueue(
      entries, true, nccl_op_context_.error_check_callback_);
}

bool NCCLHierarchicalAllreduce::Enabled(
    const ParameterManager& param_manager,
    const std::vector<TensorTableEntry>& entries,
    const Response& response) const {
  if (!NCCLAllreduce::Enabled(param_manager, entries, response)) {
    return false;
  }
  return param_manager.HierarchicalAllreduce();
}
#endif

void NCCLTorusAllreduce::WaitForData(std::vector<TensorTableEntry>& entries) {
  if (global_state_->timeline.Initialized()) {
    // If timeline is initialized, need to use normal CPU syncing path
    HorovodOp::WaitForData(entries);
  } else {
    // Push events to set to deduplicate entries
    std::unordered_set<gpuEvent_t> event_set;
    for (auto& e : entries) {
      e.ready_event_list.PushEventsToSet(event_set);
    }
    for (auto& ev : event_set) {
      HVD_GPU_CHECK(gpuStreamWaitEvent(*gpu_op_context_.stream, ev, 0));
    }
  }
}

Status NCCLTorusAllreduce::Execute(std::vector<TensorTableEntry>& entries,
                                   const Response& response) {
  auto& first_entry = entries.at(0);
  auto& process_set =
      global_state_->process_set_table.Get(first_entry.process_set_id);

  // Determine GPU IDs of the devices participating in this communicator.
  std::vector<int32_t> local_nccl_device_map;
  local_nccl_device_map.reserve(process_set.controller->GetLocalCommRanks().size());
  for (int rank : process_set.controller->GetLocalCommRanks()) {
    int32_t device = response.devices().at(rank);
    local_nccl_device_map.push_back(device);
  }
  gpu_op_context_.InitGPU(entries);
  local_nccl_op_context_.InitNCCLComm(entries, local_nccl_device_map);

  std::vector<int32_t> cross_nccl_device_map({response.devices()[process_set.controller->GetCrossRank()]});
  cross_nccl_op_context_.InitNCCLComm(entries, cross_nccl_device_map);
  gpu_op_context_.InitGPUQueue(entries, response);

  WaitForData(entries);

  ncclRedOp_t ncclOp = ncclSum;
  double prescale_factor = response.prescale_factor();
  double postscale_factor = response.postscale_factor();

  if (response.reduce_op() == ReduceOp::AVERAGE) {
#if !HAVE_ROCM
    ncclOp = ncclSum;
    // Averaging happens via postscale_factor
    postscale_factor /= process_set.controller->GetSize();
#else
    throw std::logic_error("AVERAGE not allowed.");
#endif
  } else if (response.reduce_op() == ReduceOp::SUM) {
    ncclOp = ncclSum;
  } else if (response.reduce_op() == ReduceOp::MIN) {
    ncclOp = ncclMin;
  } else if (response.reduce_op() == ReduceOp::MAX) {
    ncclOp = ncclMax;
  } else if (response.reduce_op() == ReduceOp::PRODUCT) {
    ncclOp = ncclProd;
  } else {
    throw std::logic_error("Reduction op type not supported.");
  }

  const void* fused_input_data;
  void* buffer_data;
  size_t buffer_len;

  // Copy (and possibly scale) tensors into the fusion buffer.
  if (entries.size() > 1) {
    ScaleMemcpyInFusionBuffer(entries, fused_input_data, buffer_data, 
                              buffer_len, prescale_factor);
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
                                MEMCPY_IN_FUSION_BUFFER,
                                *gpu_op_context_.stream);
    }
  } else {
    fused_input_data = first_entry.tensor->data();
    buffer_data = (void*)first_entry.output->data();
    buffer_len = (size_t)first_entry.output->size();
    int64_t num_elements =
        buffer_len / DataType_Size(first_entry.tensor->dtype());
    if (prescale_factor != 1.0) {
      // Execute prescaling op
      ScaleBuffer(prescale_factor, entries, fused_input_data,
                  buffer_data, num_elements);
      fused_input_data = buffer_data; // for unfused, scale is done out of place
    }
  }

  // Do allreduce.
  int64_t num_elements =
      buffer_len / DataType_Size(first_entry.tensor->dtype());
  int element_size = DataType_Size(first_entry.tensor->dtype());
  int local_size = process_set.controller->GetLocalSize();
  int local_rank = process_set.controller->GetLocalRank();

  // If cluster is homogeneous and we are using fusion buffer, include
  // dummy elements from the buffer (if necessary) to make sure the data
  // is divisible by local_size. This is always possible since we
  // set the fusion buffer size divisible by local_size.
  if (process_set.controller->IsHomogeneous() && entries.size() > 1) {
    // Making sure the number of elements is divisible by
    // FUSION_BUFFER_ATOMIC_UNIT for improved performance
    int div = local_size * FUSION_BUFFER_ATOMIC_UNIT;
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
  // ReduceScatter - Parallelized NCCL Allreduce - NCCL Allgather. For the
  // non-divisible part (if any), do NCCL Reduce (at rank local_size-1),
  // NCCL Allreduce (across rank (local_size-1)'s), and NCCL Bcast.

  int64_t num_elements_per_rank =
      process_set.controller->IsHomogeneous() ? num_elements / local_size : 0;

  size_t buffer_len_per_rank = element_size * num_elements_per_rank;

  void* buffer_data_at_rank_offset =
      (uint8_t*)buffer_data + buffer_len_per_rank * local_rank;

  int64_t num_elements_remaining = process_set.controller->IsHomogeneous()
                                       ? num_elements % local_size
                                       : num_elements;

  size_t buffer_len_remaining = element_size * num_elements_remaining;

  void* buffer_data_remainder =
      (uint8_t*)buffer_data + buffer_len_per_rank * local_size;

  void* fused_input_data_remainder =
      (uint8_t*)fused_input_data + buffer_len_per_rank * local_size;

  int root_rank = process_set.controller->IsHomogeneous() ? local_size - 1 : 0;
  bool is_root_rank = local_rank == root_rank;

  int64_t total_num_elements =
      is_root_rank ? num_elements_per_rank + num_elements_remaining
                   : num_elements_per_rank;
  int64_t total_buffer_len = is_root_rank
                                 ? buffer_len_per_rank + buffer_len_remaining
                                 : buffer_len_per_rank;

  auto& timeline = global_state_->timeline;
  if (num_elements_per_rank > 0) {
    auto nccl_result = ncclReduceScatter(
        fused_input_data, buffer_data_at_rank_offset,
        (size_t)num_elements_per_rank, GetNCCLDataType(first_entry.tensor),
        ncclOp, *local_nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
    local_nccl_context_->ErrorCheck("ncclReduceScatter", nccl_result,
                              *local_nccl_op_context_.nccl_comm_);
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_REDUCESCATTER,
                                *gpu_op_context_.stream);
    }
  }

  if (num_elements_remaining > 0) {
    // Reduce the remaining data at local_size-1 to append to
    // existing buffer
    auto nccl_result =
        ncclReduce(fused_input_data_remainder, buffer_data_remainder,
                   (size_t)num_elements_remaining,
                   GetNCCLDataType(first_entry.tensor), ncclOp, root_rank,
                   *local_nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
    local_nccl_context_->ErrorCheck("ncclReduce", nccl_result,
                              *local_nccl_op_context_.nccl_comm_);
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_REDUCE,
                                *gpu_op_context_.stream);
    }
  }

  if (process_set.controller->IsHomogeneous() || is_root_rank) {
  timeline.ActivityStartAll(entries, NCCL_ALLREDUCE);
    auto cross_nccl_result = ncclAllReduce(buffer_data_at_rank_offset, buffer_data_at_rank_offset,
                                           (size_t) total_num_elements, GetNCCLDataType(first_entry.tensor),
                                           ncclOp, *cross_nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
    cross_nccl_context_->ErrorCheck("ncclAllReduce", cross_nccl_result, *cross_nccl_op_context_.nccl_comm_);
    timeline.ActivityEndAll(entries);
  }
  if (num_elements_per_rank > 0) {
    local_nccl_context_->ErrorCheck(
        "ncclAllGather",
        ncclAllGather(buffer_data_at_rank_offset, buffer_data,
                      (size_t)num_elements_per_rank,
                      GetNCCLDataType(first_entry.tensor),
                      *local_nccl_op_context_.nccl_comm_, *gpu_op_context_.stream),
        *local_nccl_op_context_.nccl_comm_);
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_ALLGATHER,
                                *gpu_op_context_.stream);
    }
  }
  if (num_elements_remaining > 0) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 2, 12)
    local_nccl_context_->ErrorCheck(
        "ncclBroadcast",
        ncclBroadcast(buffer_data_remainder, buffer_data_remainder,
                      (size_t)num_elements_remaining,
                      GetNCCLDataType(first_entry.tensor), root_rank,
                      *local_nccl_op_context_.nccl_comm_, *gpu_op_context_.stream),
        *local_nccl_op_context_.nccl_comm_);
#else
    local_nccl_context_->ErrorCheck(
        "ncclBcast",
        ncclBcast(buffer_data_remainder, (size_t)num_elements_remaining,
                  GetNCCLDataType(first_entry.tensor), root_rank,
                  *local_nccl_op_context_.nccl_comm_, *gpu_op_context_.stream),
        *local_nccl_op_context_.nccl_comm_);
#endif
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_BCAST,
                                *gpu_op_context_.stream);
    }
  }

  // Copy (and possible scale) tensors out of the fusion buffer.
  if (entries.size() > 1) {
    ScaleMemcpyOutFusionBuffer(buffer_data, buffer_len,
                               postscale_factor, entries);

    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
                                MEMCPY_OUT_FUSION_BUFFER,
                                *gpu_op_context_.stream);
    }
  } else {
    if (postscale_factor != 1.0) {
      // Execute postscaling op
      ScaleBuffer(postscale_factor, entries, buffer_data,
                  buffer_data, num_elements);
    }
  }

  return gpu_op_context_.FinalizeGPUQueue(
      entries, true, local_nccl_op_context_.error_check_callback_);
}

bool NCCLTorusAllreduce::Enabled(const ParameterManager& param_manager,
                                 const std::vector<TensorTableEntry>& entries,
                                 const Response& response) const {
  if (!GPUAllreduce::Enabled(param_manager, entries, response)) {
    return false;
  }
  return param_manager.TorusAllreduce();
}

void NCCLBroadcast::WaitForData(std::vector<TensorTableEntry>& entries) {
  if (global_state_->timeline.Initialized()) {
    // If timeline is initialized, need to use normal CPU syncing path
    HorovodOp::WaitForData(entries);
  } else {
    // Push events to set to deduplicate entries
    std::unordered_set<gpuEvent_t> event_set;
    for (auto& e : entries) {
      e.ready_event_list.PushEventsToSet(event_set);
    }
    for (auto& ev : event_set) {
      HVD_GPU_CHECK(gpuStreamWaitEvent(*gpu_op_context_.stream, ev, 0));
    }
  }
}

Status NCCLBroadcast::Execute(std::vector<TensorTableEntry>& entries,
                              const Response& response) {
  assert(entries.size() == 1);
  auto e = entries[0];
  auto& process_set = global_state_->process_set_table.Get(e.process_set_id);

  gpu_op_context_.InitGPU(entries);
  nccl_op_context_.InitNCCLComm(entries, response.devices());
  gpu_op_context_.InitGPUQueue(entries, response);

  WaitForData(entries);

  // On root rank, ncclbcast sends data, on other ranks it receives data.
  void* data_ptr;
  if (process_set.controller->GetRank() == e.root_rank) {
    data_ptr = (void*)e.tensor->data();
  } else {
    data_ptr = (void*)e.output->data();
  }

  // We only use 'ncclChar' for this operation because the type format does not
  // matter for a broadcast, only the size of the data.
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 2, 12)
  nccl_context_->ErrorCheck("ncclBroadcast",
                            ncclBroadcast(data_ptr, data_ptr,
                                          e.tensor->shape().num_elements() *
                                              DataType_Size(e.tensor->dtype()),
                                          ncclChar, e.root_rank,
                                          *nccl_op_context_.nccl_comm_,
                                          *gpu_op_context_.stream),
                            *nccl_op_context_.nccl_comm_);
#else
  nccl_context_->ErrorCheck("ncclBcast",
                            ncclBcast(data_ptr,
                                      e.tensor->shape().num_elements() *
                                          DataType_Size(e.tensor->dtype()),
                                      ncclChar, e.root_rank,
                                      *nccl_op_context_.nccl_comm_,
                                      *gpu_op_context_.stream),
                            *nccl_op_context_.nccl_comm_);
#endif
  if (global_state_->timeline.Initialized()) {
    gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_BCAST,
                              *gpu_op_context_.stream);
  }

  return gpu_op_context_.FinalizeGPUQueue(
      entries, true, nccl_op_context_.error_check_callback_);
}

Status NCCLAllgather::AllocateOutput(std::vector<TensorTableEntry>& entries,
                                   const Response& response,
                                   int64_t**& entry_component_sizes) {
  for (size_t ec = 0; ec < entries.size(); ++ec) {
    auto& e = entries[ec];
    auto& process_set = global_state_->process_set_table.Get(e.process_set_id);
    int global_size = process_set.controller->GetSize();
    // Every tensor participating in Allgather operation may have different
    // first dimension size, but the rest of dimensions are same for all
    // tensors.  Here we get shape of tensor sliced by first dimension.
    TensorShape single_slice_shape;
    for (int i = 1; i < e.tensor->shape().dims(); ++i) {
      single_slice_shape.AddDim(e.tensor->shape().dim_size(i));
    }

    // Copy tensor sizes from the response into a vector of int64_t
    // and compute total size.  This is size of first dimension.
    int64_t total_entry_dimension_size = 0;
    const auto& tensor_sizes = response.tensor_sizes();
    for (int rc = 0; rc < global_size; ++rc) {
      auto component_size = tensor_sizes[ec * global_size + rc];
      total_entry_dimension_size += component_size;

      if (entry_component_sizes) {
        entry_component_sizes[ec][rc] =
                          component_size * single_slice_shape.num_elements();
      }
    }

    // Allgather output will have shape of:
    // (sum of first dimension of every tensor) x (tensor slice shape).
    TensorShape output_shape;
    output_shape.AddDim((int64_t)total_entry_dimension_size);
    output_shape.AppendShape(single_slice_shape);

    std::shared_ptr<ReadyEvent> event;
    Status status = e.context->AllocateOutput(e.output_index, output_shape,
                                              &e.output, &event);
    if (!status.ok()) {
      LOG(WARNING) << "NCCLAllgather::AllocateOutput failed: "
                   << status.reason();
      return status;
    }

    // Add event dependency for output allocation to stream
    if (event) {
      HVD_GPU_CHECK(gpuStreamWaitEvent(*gpu_op_context_.stream, event->event(), 0));
    }

  }

  return Status::OK();
}

void NCCLAllgather::WaitForData(std::vector<TensorTableEntry>& entries) {
  if (global_state_->timeline.Initialized()) {
    // If timeline is initialized, need to use normal CPU syncing path
    HorovodOp::WaitForData(entries);
  } else {
    // Push events to set to deduplicate entries
    std::unordered_set<gpuEvent_t> event_set;
    for (auto& e : entries) {
      e.ready_event_list.PushEventsToSet(event_set);
    }
    for (auto& ev : event_set) {
      HVD_GPU_CHECK(gpuStreamWaitEvent(*gpu_op_context_.stream, ev, 0));
    }
  }
}

Status NCCLAllgather::Execute(std::vector<TensorTableEntry>& entries,
                              const Response& response) {
  assert(!entries.empty());
  auto& first_entry = entries[0];
  auto& process_set =
      global_state_->process_set_table.Get(first_entry.process_set_id);

  gpu_op_context_.InitGPU(entries);
  nccl_op_context_.InitNCCLComm(entries, response.devices());
  gpu_op_context_.InitGPUQueue(entries, response);

  WaitForData(entries);

  // Sizes of subcomponents of each entry from all ranks
  auto** entry_component_sizes = new int64_t*[entries.size()];

  // Offset of each subcomponent of every entry in the final buffer after
  // allgatherv
  auto** entry_component_offsets = new int64_t*[entries.size()];

  int global_size = process_set.controller->GetSize();
  int global_rank = process_set.controller->GetRank();
  auto* recvcounts = new int[global_size]();
  auto* displcmnts = new int[global_size]();

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    entry_component_sizes[ec] = new int64_t[global_size]();
    entry_component_offsets[ec] = new int64_t[global_size]();
  }

  global_state_->timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);
  Status status =
      AllocateOutput(entries, response, entry_component_sizes);
  if (!status.ok()) {
    for (size_t ec = 0; ec < entries.size(); ++ec) {
      delete[] entry_component_sizes[ec];
      delete[] entry_component_offsets[ec];
    }
    delete[] entry_component_sizes;
    delete[] entry_component_offsets;
    delete[] recvcounts;
    delete[] displcmnts;
    return status;
  }
  global_state_->timeline.ActivityEndAll(entries);

  auto element_size = (int)DataType_Size(first_entry.tensor->dtype());
  int padding_elements = 1;
  if (entries.size() > 1) {
    assert(BATCHED_D2D_PADDING % element_size == 0);
    padding_elements = BATCHED_D2D_PADDING / element_size;
  }

  SetRecvcounts(entry_component_sizes, entries.size(), global_size, recvcounts,
                padding_elements);
  SetDisplacements(recvcounts, displcmnts, global_size);
  SetEntryComponentOffsets(entry_component_sizes, recvcounts, entries.size(),
                           global_size, entry_component_offsets);

  const void* fused_input_data;
  void* buffer_data;

  // Copy memory into the fusion buffer.
  if (entries.size() > 1) {
    MemcpyInFusionBuffer(entries, displcmnts, element_size, buffer_data);
    fused_input_data =
        (uint8_t*)buffer_data + displcmnts[global_rank] * element_size;

    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
                                MEMCPY_IN_FUSION_BUFFER,
                                *gpu_op_context_.stream);
    }
  } else {
    fused_input_data = first_entry.tensor->data();
    buffer_data = (void*)first_entry.output->data();
  }

  bool same_shape = true;
  const auto& tensor_sizes = response.tensor_sizes();
  for (size_t ec = 0; ec < entries.size(); ++ec) {
    for (int rc = 1; rc < global_size; ++rc) {
      if (tensor_sizes[ec * global_size + rc] !=
          tensor_sizes[ec * global_size]) {
        same_shape = false;
        break;
      }
    }
    if (!same_shape) {
      break;
    }
  }

  // Do allgather.
  if (same_shape) {
    auto nccl_result = ncclAllGather(
        fused_input_data, buffer_data, recvcounts[0] * element_size, ncclChar,
        *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);

    nccl_context_->ErrorCheck("ncclAllGather", nccl_result,
                              *nccl_op_context_.nccl_comm_);

    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_ALLGATHER,
                                *gpu_op_context_.stream);
    }
  } else {
    nccl_context_->ErrorCheck("ncclGroupStart", ncclGroupStart(),
                              *nccl_op_context_.nccl_comm_);
    for (int rc = 0; rc < global_size; ++rc) {
      void* new_buffer_data =
          (uint8_t*)buffer_data + displcmnts[rc] * element_size;
      auto nccl_result = ncclBroadcast(
          fused_input_data, new_buffer_data, recvcounts[rc] * element_size,
          ncclChar, rc, *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
      nccl_context_->ErrorCheck("ncclBroadcast", nccl_result,
                                *nccl_op_context_.nccl_comm_);
    }
    nccl_context_->ErrorCheck("ncclGroupEnd", ncclGroupEnd(),
                              *nccl_op_context_.nccl_comm_);

    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_BCAST,
                                *gpu_op_context_.stream);
    }
  }

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    MemcpyOutFusionBuffer(entry_component_offsets, entry_component_sizes,
                          buffer_data, element_size, entries);

    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
                                MEMCPY_OUT_FUSION_BUFFER,
                                *gpu_op_context_.stream);
    }
  }

  delete[] recvcounts;
  delete[] displcmnts;

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    delete[] entry_component_sizes[ec];
    delete[] entry_component_offsets[ec];
  }
  delete[] entry_component_sizes;
  delete[] entry_component_offsets;

  return gpu_op_context_.FinalizeGPUQueue(
      entries, true, nccl_op_context_.error_check_callback_);
}

bool NCCLAllgather::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}

void NCCLAlltoall::WaitForData(std::vector<TensorTableEntry>& entries) {
  if (global_state_->timeline.Initialized()) {
    // If timeline is initialized, need to use normal CPU syncing path
    HorovodOp::WaitForData(entries);
  } else {
    // Push events to set to deduplicate entries
    std::unordered_set<gpuEvent_t> event_set;
    for (auto& e : entries) {
      e.ready_event_list.PushEventsToSet(event_set);
    }
    for (auto& ev : event_set) {
      HVD_GPU_CHECK(gpuStreamWaitEvent(*gpu_op_context_.stream, ev, 0));
    }
  }
}

Status NCCLAlltoall::Execute(std::vector<TensorTableEntry>& entries,
                             const Response& response) {
#ifdef NCCL_P2P_SUPPORTED
  assert(entries.size() == 1);
  auto e = entries[0];
  auto& process_set = global_state_->process_set_table.Get(e.process_set_id);

  gpu_op_context_.InitGPU(entries);
  nccl_op_context_.InitNCCLComm(entries, response.devices());
  gpu_op_context_.InitGPUQueue(entries, response);

  WaitForData(entries);

  std::vector<int32_t> sdispls, rdispls;
  std::vector<int32_t> sendcounts, recvcounts;
  Status status =
      PrepareOutputAndParams(e, sdispls, rdispls, sendcounts, recvcounts);
  if (!status.ok()) {
    return status;
  }

  auto world_size = process_set.controller->GetSize();

  nccl_context_->ErrorCheck("ncclGroupStart", ncclGroupStart(),
                            *nccl_op_context_.nccl_comm_);

  for (int i = 0; i < world_size; ++i) {
    if (recvcounts[i] > 0) {
      auto nccl_result =
          ncclRecv((uint8_t*)e.output->data() +
                       rdispls[i] * DataType_Size(e.tensor->dtype()),
                   recvcounts[i] * DataType_Size(e.tensor->dtype()), ncclChar,
                   i, *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
      nccl_context_->ErrorCheck("ncclRecv", nccl_result,
                                *nccl_op_context_.nccl_comm_);
    }

    if (sendcounts[i] > 0) {
      auto nccl_result =
          ncclSend((uint8_t*)e.tensor->data() +
                       sdispls[i] * DataType_Size(e.tensor->dtype()),
                   sendcounts[i] * DataType_Size(e.tensor->dtype()), ncclChar,
                   i, *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
      nccl_context_->ErrorCheck("ncclSend", nccl_result,
                                *nccl_op_context_.nccl_comm_);
    }
  }
  nccl_context_->ErrorCheck("ncclGroupEnd", ncclGroupEnd(),
                            *nccl_op_context_.nccl_comm_);

  if (global_state_->timeline.Initialized()) {
    gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_ALLTOALL,
                              *gpu_op_context_.stream);
  }

  return gpu_op_context_.FinalizeGPUQueue(entries);
#else
  throw std::runtime_error(
      "NCCLAlltoall requires NCCL version >= 2.7.0. If your NCCL installation "
      "cannot be updated "
      "and you installed with HOROVOD_GPU_OPERATIONS=NCCL, reinstall with only "
      "supported "
      "operations individually specified (i.e. HOROVOD_GPU_ALLREDUCE=NCCL "
      "HOROVOD_GPU_BROADCAST=NCCL "
      "HOROVOD_GPU_ALLGATHER=NCCL). Otherwise, exclude "
      "HOROVOD_GPU_ALLTOALL=NCCL from your "
      "installation command.");
#endif
}

Status NCCLReducescatter::Execute(std::vector<TensorTableEntry>& entries,
                                  const Response& response) {
  assert(!entries.empty());
  auto& first_entry = entries[0];
  auto& process_set =
      global_state_->process_set_table.Get(first_entry.process_set_id);

  gpu_op_context_.InitGPU(entries);
  nccl_op_context_.InitNCCLComm(entries, response.devices());
  gpu_op_context_.InitGPUQueue(entries, response);

  WaitForData(entries);

  int process_set_rank = process_set.controller->GetRank();
  int process_set_size = process_set.controller->GetSize();
  auto output_shapes = ComputeOutputShapes(entries, process_set_size);
  std::vector<int> recvcounts = ComputeReceiveCounts(output_shapes);

  global_state_->timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);
  Status status = AllocateOutput(entries, output_shapes[process_set_rank]);
  if (!status.ok()) {
    return status;
  }
  global_state_->timeline.ActivityEndAll(entries);

  size_t element_size = DataType_Size(first_entry.tensor->dtype());
  void* buffer_data = nullptr;
  void* recv_pointer = nullptr;
  const void* fused_input_data = nullptr;

  // Copy memory into the fusion buffer.
  if (entries.size() > 1) {
    MemcpyInFusionBuffer(entries, output_shapes, element_size, buffer_data);
    fused_input_data = buffer_data;
    recv_pointer = reinterpret_cast<int8_t*>(buffer_data) +
                   process_set_rank * recvcounts[0] * element_size;
    // ncclReduceScatter() will be performed in place on the fusion buffer, cf.:
    // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/inplace.html
  } else {
    fused_input_data = first_entry.tensor->data();
    buffer_data = (void*)first_entry.output->data();
    recv_pointer = buffer_data;
  }

  // If a reduced tensor cannot be scattered evenly over the participating processes, the first ranks 0, ...,
  // (tensor_shape.dim_size(0) % process_set_size - 1) will receive more data than the last ranks.
  // [See ReducescatterOp::ComputeOutputShapeForRank()]
  bool same_output_shape = (recvcounts.front() == recvcounts.back());
  assert(!same_output_shape || recvcounts.size() < 2 ||
         std::all_of(recvcounts.begin() + 1, recvcounts.end(),
                     [rc0 = recvcounts[0]](int rc) { return rc == rc0; }));

  if (same_output_shape) {
    // Do reducescatter.
    auto nccl_result = ncclReduceScatter(
        fused_input_data, recv_pointer, recvcounts[0],
        GetNCCLDataType(first_entry.tensor), ncclSum,
        *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
    nccl_context_->ErrorCheck("ncclReduceScatter", nccl_result,
                              *nccl_op_context_.nccl_comm_);

    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_REDUCESCATTER,
                                *gpu_op_context_.stream);
    }
  } else {
    // Simulate "ReduceScatterV" by an equivalent group of reduces.
    nccl_context_->ErrorCheck("ncclGroupStart", ncclGroupStart(),
                              *nccl_op_context_.nccl_comm_);
    std::size_t offset_bytes = 0;
    for (int recv_rank = 0; recv_rank < process_set_size; ++recv_rank) {
      const void* send_pointer =
          reinterpret_cast<const int8_t*>(fused_input_data) + offset_bytes;
      auto nccl_result =
          ncclReduce(send_pointer, recv_pointer, recvcounts[recv_rank],
                     GetNCCLDataType(first_entry.tensor), ncclSum, recv_rank,
                     *nccl_op_context_.nccl_comm_, *gpu_op_context_.stream);
      nccl_context_->ErrorCheck("ncclReduce", nccl_result,
                                *nccl_op_context_.nccl_comm_);
      offset_bytes += recvcounts[recv_rank] * element_size;
    }
    nccl_context_->ErrorCheck("ncclGroupEnd", ncclGroupEnd(),
                              *nccl_op_context_.nccl_comm_);
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue, NCCL_REDUCE,
                                *gpu_op_context_.stream);
    }
  }

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    MemcpyOutFusionBuffer(recv_pointer, entries);

    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
                                MEMCPY_OUT_FUSION_BUFFER,
                                *gpu_op_context_.stream);
    }
  }

  return gpu_op_context_.FinalizeGPUQueue(
      entries, true, nccl_op_context_.error_check_callback_);
}

bool NCCLReducescatter::Enabled(const ParameterManager& param_manager,
                                const std::vector<TensorTableEntry>& entries,
                                const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}

Status NCCLReducescatter::AllocateOutput(
    std::vector<TensorTableEntry>& entries, const std::vector<TensorShape>& output_shapes) {
  for (size_t ec = 0; ec < entries.size(); ++ec) {
    auto& e = entries[ec];
    const auto& output_shape = output_shapes[ec];

    std::shared_ptr<ReadyEvent> event;
    Status status = e.context->AllocateOutput(e.output_index, output_shape,
                                              &e.output, &event);
    if (!status.ok()) {
      LOG(WARNING) << "NCCLReducescatter::AllocateOutput failed: "
                   << status.reason();
      return status;
    }

    // Add event dependency for output allocation to stream
    if (event) {
      HVD_GPU_CHECK(gpuStreamWaitEvent(*gpu_op_context_.stream, event->event(), 0));
    }

  }

  return Status::OK();
}

void NCCLReducescatter::WaitForData(std::vector<TensorTableEntry>& entries) {
  if (global_state_->timeline.Initialized()) {
    // If timeline is initialized, need to use normal CPU syncing path
    HorovodOp::WaitForData(entries);
  } else {
    // Push events to set to deduplicate entries
    std::unordered_set<gpuEvent_t> event_set;
    for (auto& e : entries) {
      e.ready_event_list.PushEventsToSet(event_set);
    }
    for (auto& ev : event_set) {
      HVD_GPU_CHECK(gpuStreamWaitEvent(*gpu_op_context_.stream, ev, 0));
    }
  }
}

} // namespace common
} // namespace horovod
