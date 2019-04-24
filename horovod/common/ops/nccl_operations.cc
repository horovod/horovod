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

#include "nccl_operations.h"

namespace horovod {
namespace common {

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

void NCCLContext::ErrorCheck(std::string op_name, ncclResult_t nccl_result) {
  if (nccl_result != ncclSuccess) {
    throw std::logic_error(std::string(op_name) + " failed: " + ncclGetErrorString(nccl_result));
  }
}

void NCCLContext::ShutDown(){
  for(auto it = nccl_comms.begin(); it != nccl_comms.end(); ++it){
    ncclCommDestroy(it->second);
  }
  nccl_comms.clear();
}

void ParallelNCCLContext::ShutDown(){
  NCCLContext::ShutDown();

  for (auto it = end_nccl_comms.begin(); it != end_nccl_comms.end(); ++it) {
    ncclCommDestroy(it->second);
  }
  end_nccl_comms.clear();
}

NCCLAllreduce::NCCLAllreduce(NCCLContext* nccl_context,
                             MPIContext* mpi_context,
                             CUDAContext* cuda_context,
                             HorovodGlobalState* global_state)
    : CUDAAllreduce(cuda_context, global_state),
      nccl_context_(nccl_context), mpi_context_(mpi_context) {}

Status NCCLAllreduce::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& first_entry = entries[0];

  InitCUDA(entries);
  InitNCCLComm(entries, response.devices());
  InitCUDAQueue(entries, response);

  const void* fused_input_data;
  void* buffer_data;
  size_t buffer_len;

  // Copy memory into the fusion buffer.
  if (entries.size() > 1) {
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);

    if (global_state_->timeline.Initialized()) {
      cuda_context_->RecordEvent(event_queue_, MEMCPY_IN_FUSION_BUFFER, *stream_);
    }
  } else {
    fused_input_data = first_entry.tensor->data();
    buffer_data = (void*) first_entry.output->data();
    buffer_len = (size_t) first_entry.output->size();
  }

  int64_t num_elements = 0;
  for (auto& e : entries) {
    num_elements += e.tensor->shape().num_elements();
  }

  // Do allreduce.
  auto nccl_result = ncclAllReduce(fused_input_data, buffer_data,
                                   (size_t) num_elements,
                                   GetNCCLDataType(first_entry.tensor), ncclSum,
                                   *nccl_comm_, *stream_);
  nccl_context_->ErrorCheck("ncclAllReduce", nccl_result);
  if (global_state_->timeline.Initialized()) {
    cuda_context_->RecordEvent(event_queue_, NCCL_ALLREDUCE, *stream_);
  }

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    MemcpyOutFusionBuffer(buffer_data, entries);

    if (global_state_->timeline.Initialized()) {
      cuda_context_->RecordEvent(event_queue_, MEMCPY_OUT_FUSION_BUFFER, *stream_);
    }
  }

  return FinalizeCUDAQueue(entries);
}

void NCCLAllreduce::InitNCCLComm(const std::vector<TensorTableEntry>& entries,
                                 const std::vector<int32_t>& nccl_device_map) {
  // Ensure NCCL communicator is in the map before executing reduction.
  ncclComm_t& nccl_comm = nccl_context_->nccl_comms[nccl_device_map];
  if (nccl_comm == nullptr) {
    auto& timeline = global_state_->timeline;
    timeline.ActivityStartAll(entries, INIT_NCCL);

    int nccl_rank, nccl_size;
    Communicator nccl_id_bcast_comm;
    PopulateNCCLCommStrategy(nccl_rank, nccl_size, nccl_id_bcast_comm);

    ncclUniqueId nccl_id;
    if (nccl_rank == 0) {
      nccl_context_->ErrorCheck("ncclGetUniqueId", ncclGetUniqueId(&nccl_id));
    }

    int bcast_op = MPI_Bcast((void*) &nccl_id,
                             sizeof(nccl_id),
                             mpi_context_->GetMPIDataType(HOROVOD_BYTE),
                             0,
                             mpi_context_->GetMPICommunicator(nccl_id_bcast_comm));
    if (bcast_op != MPI_SUCCESS) {
      throw std::logic_error("MPI_Broadcast failed, see MPI output for details.");
    }

    ncclComm_t new_nccl_comm;
    auto nccl_result = ncclCommInitRank(&new_nccl_comm, nccl_size, nccl_id, nccl_rank);
    nccl_context_->ErrorCheck("ncclCommInitRank", nccl_result);
    nccl_comm = new_nccl_comm;

    // Barrier helps NCCL to synchronize after initialization and avoid
    // deadlock that we've been seeing without it.
    int barrier_op = MPI_Barrier(mpi_context_->GetMPICommunicator(Communicator::GLOBAL));
    if (barrier_op != MPI_SUCCESS) {
      throw std::logic_error("MPI_Barrier failed, see MPI output for details.");
    }

    timeline.ActivityEndAll(entries);
  }

  nccl_comm_ = &nccl_comm;
}

void NCCLAllreduce::PopulateNCCLCommStrategy(int& nccl_rank, int& nccl_size,
                                             Communicator& nccl_id_bcast_comm) {
  nccl_rank = global_state_->rank;
  nccl_size = global_state_->size;
  nccl_id_bcast_comm = Communicator::GLOBAL;
}

NCCLHierarchicalAllreduce::NCCLHierarchicalAllreduce(NCCLContext* nccl_context, MPIContext* mpi_context,
                                                     CUDAContext* cuda_context, HorovodGlobalState* global_state)
    : NCCLAllreduce(nccl_context, mpi_context,
                    cuda_context, global_state) {}

Status NCCLHierarchicalAllreduce::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& first_entry = entries[0];

  // Determine GPU IDs of the devices participating in this communicator.
  std::vector<int32_t> nccl_device_map;
  nccl_device_map.reserve(global_state_->local_comm_ranks.size());
  for (int rank : global_state_->local_comm_ranks) {
    nccl_device_map.push_back(response.devices()[rank]);
  }

  InitCUDA(entries);
  InitNCCLComm(entries, nccl_device_map);
  InitCUDAQueue(entries, response);

  const void* fused_input_data;
  void* buffer_data;
  size_t buffer_len;

  // Copy memory into the fusion buffer.
  if (entries.size() > 1) {
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);

    if (global_state_->timeline.Initialized()) {
      cuda_context_->RecordEvent(event_queue_, MEMCPY_IN_FUSION_BUFFER, *stream_);
    }
  } else {
    fused_input_data = first_entry.tensor->data();
    buffer_data = (void*) first_entry.output->data();
    buffer_len = (size_t) first_entry.output->size();
  }

  int64_t num_elements = 0;
  for (auto& e : entries) {
    num_elements += e.tensor->shape().num_elements();
  }

  // Do allreduce.
  int element_size = mpi_context_->GetMPITypeSize(first_entry.tensor->dtype());

  // If cluster is homogeneous and we are using fusion buffer, include
  // dummy elements from the buffer (if necessary) to make sure the data
  // is divisible by local_size. This is always possible since we
  // set the fusion buffer size divisible by local_size.
  if (global_state_->is_homogeneous && entries.size() > 1) {
    // Making sure the number of elements is divisible by
    // FUSION_BUFFER_ATOMIC_UNIT for improved performance
    int div = global_state_->local_size * FUSION_BUFFER_ATOMIC_UNIT;
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
      global_state_->is_homogeneous
      ? num_elements / global_state_->local_size
      : 0;

  size_t buffer_len_per_rank = element_size * num_elements_per_rank;

  void* buffer_data_at_rank_offset =
      (uint8_t*) buffer_data +
      buffer_len_per_rank * global_state_->local_rank;

  int64_t num_elements_remaining =
      global_state_->is_homogeneous
      ? num_elements % global_state_->local_size
      : num_elements;

  size_t buffer_len_remaining = element_size * num_elements_remaining;

  void* buffer_data_remainder =
      (uint8_t*) buffer_data +
      buffer_len_per_rank * global_state_->local_size;

  void* fused_input_data_remainder =
      (uint8_t*) fused_input_data +
      buffer_len_per_rank * global_state_->local_size;

  int root_rank =
      global_state_->is_homogeneous ? global_state_->local_size - 1 : 0;
  bool is_root_rank = global_state_->local_rank == root_rank;

  int64_t total_num_elements =
      is_root_rank ? num_elements_per_rank + num_elements_remaining
                   : num_elements_per_rank;
  int64_t total_buffer_len =
      is_root_rank ? buffer_len_per_rank + buffer_len_remaining
                   : buffer_len_per_rank;

  auto& timeline = global_state_->timeline;
  if (num_elements_per_rank > 0) {
    auto nccl_result = ncclReduceScatter(fused_input_data,
                                         buffer_data_at_rank_offset,
                                         (size_t) num_elements_per_rank,
                                         GetNCCLDataType(first_entry.tensor),
                                         ncclSum, *nccl_comm_, *stream_);
    nccl_context_->ErrorCheck("ncclReduceScatter", nccl_result);
    if (global_state_->timeline.Initialized()) {
      cuda_context_->RecordEvent(event_queue_, NCCL_REDUCESCATTER, *stream_);
    }
  }

  if (num_elements_remaining > 0) {
    // Reduce the remaining data at local_size-1 to append to
    // existing buffer
    auto nccl_result = ncclReduce(fused_input_data_remainder,
                                  buffer_data_remainder,
                                  (size_t) num_elements_remaining,
                                  GetNCCLDataType(first_entry.tensor), ncclSum,
                                  root_rank, *nccl_comm_, *stream_);
    nccl_context_->ErrorCheck("ncclReduce", nccl_result);
    if (global_state_->timeline.Initialized()) {
      cuda_context_->RecordEvent(event_queue_, NCCL_REDUCE, *stream_);
    }
  }

  if (global_state_->is_homogeneous || is_root_rank) {
    // cudaHostAlloc is significantly slower than malloc.  Pre-allocating
    // a buffer is not safe since the tensor can be arbitrarily large.
    host_buffer_ = malloc(total_buffer_len);

    // Synchronize.
    cuda_context_->WaitForEvents(event_queue_, entries, timeline);

    // According to https://docs.nvidia.com/cuda/cuda-runtime-api/
    // api-sync-behavior.html#api-sync-behavior__memcpy-async,
    // cudaMemcpyAsync is synchronous with respect to the host, so we
    // memcpy (effectively) synchronously to generate an accurate timeline
    timeline.ActivityStartAll(entries, MEMCPY_IN_HOST_BUFFER);
    cuda_context_->ErrorCheck("cudaMemcpyAsync",
                              cudaMemcpyAsync(host_buffer_, buffer_data_at_rank_offset,
                                              total_buffer_len, cudaMemcpyDeviceToHost,
                                              *stream_));
    timeline.ActivityEndAll(entries);

    timeline.ActivityStartAll(entries, MPI_ALLREDUCE);
    int op = MPI_Allreduce(MPI_IN_PLACE, host_buffer_,
                           (int) total_num_elements,
                           mpi_context_->GetMPIDataType(first_entry.tensor),
                           mpi_context_->GetMPISumOp(first_entry.tensor->dtype()),
                           mpi_context_->GetMPICommunicator(Communicator::CROSS));
    if (op != MPI_SUCCESS) {
      throw std::logic_error("MPI_Allreduce failed, see MPI output for details.");
    }
    timeline.ActivityEndAll(entries);

    timeline.ActivityStartAll(entries, MEMCPY_OUT_HOST_BUFFER);
    cuda_context_->ErrorCheck("cudaMemcpyAsync",
                              cudaMemcpyAsync(buffer_data_at_rank_offset, host_buffer_,
                                              total_buffer_len, cudaMemcpyHostToDevice,
                                              *stream_));
    timeline.ActivityEndAll(entries);
  }

  if (num_elements_per_rank > 0) {
    nccl_context_->ErrorCheck("ncclAllGather",
                              ncclAllGather(buffer_data_at_rank_offset, buffer_data,
                                            (size_t) num_elements_per_rank,
                                            GetNCCLDataType(first_entry.tensor),
                                            *nccl_comm_, *stream_));
    if (global_state_->timeline.Initialized()) {
      cuda_context_->RecordEvent(event_queue_, NCCL_ALLGATHER, *stream_);
    }
  }
  if (num_elements_remaining > 0) {
    nccl_context_->ErrorCheck("ncclBcast",
                              ncclBcast(buffer_data_remainder,
                                        (size_t) num_elements_remaining,
                                        GetNCCLDataType(first_entry.tensor), root_rank,
                                        *nccl_comm_, *stream_));
    if (global_state_->timeline.Initialized()) {
      cuda_context_->RecordEvent(event_queue_, NCCL_BCAST, *stream_);
    }
  }

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    MemcpyOutFusionBuffer(buffer_data, entries);

    if (global_state_->timeline.Initialized()) {
      cuda_context_->RecordEvent(event_queue_, MEMCPY_OUT_FUSION_BUFFER, *stream_);
    }
  }

  return FinalizeCUDAQueue(entries);
}

bool NCCLHierarchicalAllreduce::Enabled(const ParameterManager& param_manager,
                                        const std::vector<TensorTableEntry>& entries,
                                        const Response& response) const {
  if (!NCCLAllreduce::Enabled(param_manager, entries, response)) {
    return false;
  }
  return param_manager.HierarchicalAllreduce();
}

void NCCLHierarchicalAllreduce::PopulateNCCLCommStrategy(int& nccl_rank, int& nccl_size,
                                                         Communicator& nccl_id_bcast_comm) {
  nccl_rank = global_state_->local_rank;
  nccl_size = global_state_->local_size;
  nccl_id_bcast_comm = Communicator::LOCAL;
}

// for ParallelNCCLHierarchicalAllreduce 
ParallelNCCLHierarchicalAllreduce::ParallelNCCLHierarchicalAllreduce(ParallelNCCLContext* nccl_context, 
                                                                    MPIContext* mpi_context,
                                                                    ParallelCUDAContext* cuda_context, 
                                                                    HorovodGlobalState* global_state): 
      NCCLAllreduce(nccl_context, mpi_context, cuda_context, global_state), 
      parallel_nccl_context_(nccl_context), 
      parallel_cuda_context_(cuda_context),
      parallel_mpi_context_(mpi_context), 
      mpi_queue_("mpi_queue"), 
      end_queue_("end_queue") {
}

bool ParallelNCCLHierarchicalAllreduce::Enabled(const ParameterManager& param_manager,
                                                const std::vector<TensorTableEntry>& entries,
                                                const Response& response) {
  if (!NCCLHierarchicalAllreduce::NCCLAllreduce(param_manager, entries, response)) {
    return false;
  }

  // for test
  return true;
}

// init the parallel nccl_commm
void ParallelNCCLHierarchicalAllreduce::InitParallelNCCLComm(const std::vector<TensorTableEntry>& entries, 
                                                            const std::vector<int32_t>& nccl_device_map) {
  // first should init nccl_comm and end_nccl_comm
  ncclComm_t& nccl_comm     = parallel_nccl_context_->nccl_comms[nccl_device_map];
  ncclComm_t& end_nccl_comm = parallel_nccl_context_->end_nccl_comms[nccl_device_map];

  if (nullptr == nccl_comm || nullptr == end_nccl_comm) {
    auto& timeline = global_state_->timeline;
    timeline.ActivityStartAll(entries, INIT_NCCL);

    int nccl_rank, nccl_size;
    Communicator nccl_id_bcast_comm;
    PopulateNCCLCommStrategy(nccl_rank, nccl_size, nccl_id_bcast_comm);

    if (nullptr == nccl_comm) {
      ncclUniqueId nccl_id;
      if (nccl_rank == 0) {
        parallel_nccl_context_->ErrorCheck("ncclGetUniqueId", ncclGetUniqueId(&nccl_id));
      }

      int bcast_op = MPI_Bcast((void*) &nccl_id,
                              sizeof(nccl_id),
                              mpi_context_->GetMPIDataType(HOROVOD_BYTE),
                              0,
                              mpi_context_->GetMPICommunicator(nccl_id_bcast_comm));
      if (bcast_op != MPI_SUCCESS) {
        throw std::logic_error("MPI_Broadcast failed, see MPI output for details.");
      }

      ncclComm_t new_nccl_comm;
      auto nccl_result = ncclCommInitRank(&new_nccl_comm, nccl_size, nccl_id, nccl_rank);
      parallel_nccl_context_->ErrorCheck("ncclCommInitRank", nccl_result);
      nccl_comm = new_nccl_comm;
    }

    if (nullptr == end_nccl_comm) {
      // same like nccl_comm
      ncclUniqueId end_nccl_id;
      if (nccl_rank == 0) {
        parallel_nccl_context_->ErrorCheck("ncclGetUniqueId", ncclGetUniqueId(&end_nccl_id));
      }

      int bcast_op = MPI_Bcast((void*) &end_nccl_id,
                              sizeof(end_nccl_id),
                              mpi_context_->GetMPIDataType(HOROVOD_BYTE),
                              0,
                              mpi_context_->GetMPICommunicator(nccl_id_bcast_comm));
      if (bcast_op != MPI_SUCCESS) {
        throw std::logic_error("MPI_Broadcast failed, see MPI output for details.");
      }

      ncclComm_t new_end_nccl_comm;
      auto nccl_result = ncclCommInitRank(&new_end_nccl_comm, nccl_size, end_nccl_id, nccl_rank);
      parallel_nccl_context_->ErrorCheck("ncclCommInitRank", nccl_result);
      end_nccl_comm = new_end_nccl_comm;
    }

    // Barrier helps NCCL to synchronize after initialization and avoid
    // deadlock that we've been seeing without it.
    int barrier_op = MPI_Barrier(mpi_context_->GetMPICommunicator(Communicator::GLOBAL));
    if (barrier_op != MPI_SUCCESS) {
      throw std::logic_error("MPI_Barrier failed, see MPI output for details.");
    }

    timeline.ActivityEndAll(entries);
  }
}

// init parallel cuda context
void ParallelNCCLHierarchicalAllreduce::InitParallelCUDA(const std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];

  parallel_cuda_context_->ErrorCheck("cudaSetDevice", cudaSetDevice(first_entry.device));

  // Ensure stream is in the map before executing reduction.
  cudaStream_t& stream = parallel_cuda_context_->streams[first_entry.device];
  cudaStream_t& end_stream = parallel_cuda_context_->streams[first_entry.device];

  if (nullptr == stream || nullptr == end_stream) {
    int greatest_priority;
    parallel_cuda_context_->ErrorCheck("cudaDeviceGetStreamPriorityRange",
                              cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));

    if (nullptr == stream) {
      parallel_cuda_context_->ErrorCheck("cudaStreamCreateWithPriority",
                              cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest_priority));
    }

    if (nullptr == end_stream) {
      parallel_cuda_context_->ErrorCheck("cudaStreamCreateWithPriority",
                              cudaStreamCreateWithPriority(&end_stream, cudaStreamNonBlocking, greatest_priority));
    }
  }
}

void ParallelNCCLHierarchicalAllreduce::PopulateNCCLCommStrategy(int& nccl_rank, int& nccl_size,
                                                                Communicator& nccl_id_bcast_comm) {
  nccl_rank = global_state_->local_rank;
  nccl_size = global_state_->local_size;
  nccl_id_bcast_comm = Communicator::LOCAL;
}

void ParallelNCCLHierarchicalAllreduce::MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                                                  const TensorTableEntry& e, 
                                                                  void* buffer_data_at_offset) {
  auto& first_entry = entries[0];
  auto cuda_result = cudaMemcpyAsync(buffer_data_at_offset, 
                                    e.tensor->data(),
                                    (size_t) e.tensor->size(), 
                                    cudaMemcpyDeviceToDevice,
                                    parallel_cuda_context_->streams[first_entry.device]);
  
  parallel_cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
}

void ParallelNCCLHierarchicalAllreduce::MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                                                  const void* buffer_data_at_offset, 
                                                                  TensorTableEntry& e) {
  auto& first_entry = entries[0];
  auto cuda_result = cudaMemcpyAsync((void*) e.output->data(), 
                                    buffer_data_at_offset,
                                    (size_t) e.tensor->size(), 
                                    cudaMemcpyDeviceToDevice,
                                    parallel_cuda_context_->end_streams[first_entry.device]);

  parallel_cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
}

// parallel execute is make the NCCLHierarchicalAllreduce to parallel
// the NCCLHierarchicalAllreduce can be devide to 3 step
// step1: copy data to fusion buffer, ncclReduceScatter/ncclReduce, copy data to CPU
// step2: use MPI do allreduce in CPU
// step3: copy data from CPU back to GPU, ncclAllGather/ncclBcast, copy data back to tensor
//
// set this 3 step is: a, b, c and suppose we need n allReduce and the step in NCCLHierarchicalAllreduce is serial,
// so we can think the whole process of NCCLHierarchicalAllreduce is below:
//
// a1, b1, c1, a2, b2, c2, ..., an, bn, cn
//
// So the total time is: sum[a1..n] + sum[b1..n] + sum[c1..n]
//
// in ParallelNCCLHierarchicalAllreduce use 3 thread to do the same thing (horovod background thread, MPI thread, End thread)
// than the whole process can be
//
// a1, a2, ..., an
//     b1, b2, ..., bn
//         c1, c2, ..., cn
// 
// So total time will be: max(sum[a1..n], sum[b1..n], sum[c1..n])
// 
// Memory usage:
// split allReduce to 3 thread will need more memory
// GPU memory: because task[a] and task[c] in it's own queue is running serial; So only need 2 fusion buffer for every thread (in NCCLHierarchicalAllreduce need 1)
// CPU memory: when task[a] finish, it will copy data to CPU and stay in CPU until task[c] running So it will need much more CPU memory to store the data,
// when global_state_->is_homogeneous is true: max need model_size / local_size in every rank
// whne global_state_->is_homogeneous is false: max need model_size in every rank
// model size is the size of training model, local_rank is the count of rank in current hosts
Status ParallelNCCLHierarchicalAllreduce::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& first_entry = entries[0];

  // like the NCCLHierarchicalAllreduce first step should get the resource
  std::vector<int32_t> nccl_device_map;
  nccl_device_map.reserve(global_state_->local_comm_ranks.size());
  for (int rank : global_state_->local_comm_ranks) {
    nccl_device_map.push_back(response.devices()[rank]);
  }

  // init resource
  InitParallelCUDA(entries);
  InitParallelNCCLComm(entries, nccl_device_map);

  // get timeline
  auto& timeline = global_state_->timeline;

  // get the cuda resource
  auto steam = parallel_cuda_context_->streams[first_entry.device];
  auto end_stream = parallel_cuda_context_->end_streams[first_entry.device];

  auto nccl_comm = parallel_nccl_context_->nccl_comms[nccl_device_map];
  auto end_nccl_comm = parallel_nccl_context_->end_nccl_comms[nccl_device_map];

  // init event queue
  auto event_queue = std::queue<std::pair<std::string, cudaEvent_t>>();

  if (global_state_->timeline.Initialized()) {
    parallel_cuda_context_->RecordEvent(event_queue, QUEUE, steam);
  }

  // step1: in this main thread will do ncclReduceScatter/ncclReduce, copy data to CPU
  // and will use stream and nccl_comm to do the task
  // do the same thing like NCCLHierarchicalAllreduce
  const void *fused_input_data;
  void *buffer_data;
  size_t buffer_len;

  if (entries.size() > 1) {
    // copy data to fusion buffer
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);

    if (global_state_->timeline.Initialized()) {
      parallel_cuda_context_->RecordEvent(event_queue, MEMCPY_IN_FUSION_BUFFER, steam);
    }
  } else {
    // reuse the tensor buffer
    fused_input_data = first_entry.tensor->data();
    buffer_data = (void*) first_entry.output->data();
    buffer_len = (size_t) first_entry.output->size();
  }

  int64_t num_elements = 0;
  for (auto& e : entries) {
    num_elements += e.tensor->shape().num_elements();
  }

  int element_size = parallel_mpi_context_->GetMPITypeSize(first_entry.tensor->dtype());

  if (global_state_->is_homogeneous && entries.size() > 1) {
    int div      = global_state_->local_size * FUSION_BUFFER_ATOMIC_UNIT;
    num_elements = ((num_elements + div - 1) / div) * div;
    buffer_len   = num_elements * element_size;
  }

  int64_t num_elements_per_rank = 
      global_state_->is_homogeneous
      ? num_elements / global_state_->local_size
      : 0;

  size_t buffer_len_per_rank = element_size * num_elements_per_rank;

  if (num_elements_per_rank > 0) {
    void *buffer_data_at_rank_offset = (uint8_t*)buffer_data + global_state_->local_rank * buffer_len_per_rank;

    auto nccl_result = ncclReduceScatter(fused_input_data,
                                         buffer_data_at_rank_offset,
                                         (size_t) num_elements_per_rank,
                                         GetNCCLDataType(first_entry.tensor),
                                         ncclSum, 
                                         nccl_comm, 
                                         stream);

    parallel_nccl_context_->ErrorCheck("ncclReduceScatter", nccl_result);

    if (global_state_->timeline.Initialized()) {
      parallel_cuda_context_->RecordEvent(event_queue, NCCL_REDUCESCATTER, stream);
    }
  }

  int64_t num_elements_remaining =
      global_state_->is_homogeneous
      ? num_elements % global_state_->local_size
      : num_elements;

  bool is_last_rank = (global_state_->local_rank == global_state_->local_size - 1);

  if (num_elements_remaining > 0) {
    // use laste rank to do reduce
    void* fused_input_data_remainder = (uint8_t*) fused_input_data 
                              + buffer_len_per_rank * global_state_->local_size;
                              
    void* buffer_data_remainder = (uint8_t*) buffer_data 
                              + buffer_len_per_rank * global_state_->local_size;

    // reduce to last rank
    auto nccl_result = ncclReduce(fused_input_data_remainder,
                                  buffer_data_remainder,
                                  (size_t) num_elements_remaining,
                                  GetNCCLDataType(first_entry.tensor), 
                                  ncclSum,
                                  last_rank, 
                                  nccl_comm, 
                                  stream);

    parallel_nccl_context_->ErrorCheck("ncclReduce", nccl_result);

    if (global_state_->timeline.Initialized()) {
      parallel_cuda_context_->RecordEvent(event_queue, NCCL_REDUCE, stream);
    }
  }

  // Synchronize
  parallel_cuda_context_->WaitForEvents(event_queue, entries, timeline);

  // malloc host buffer to store the mpi allreduce result
  void *host_buffer = nullptr;

  if (global_state_->is_homogeneous || is_last_rank) {
    int64_t host_buffer_len = 
                is_last_rank 
                ? ((num_elements_per_rank + num_elements_remaining) * element_size)
                : (num_elements_per_rank * element_size);

    // malloc host buffer
    host_buffer = malloc(host_buffer_len);

    void *buffer_data_at_rank_offset = (uint8_t*)buffer_data + global_state_->local_rank * buffer_len_per_rank;

    // copy data to CPU
    timeline.ActivityStartAll(entries, MEMCPY_IN_HOST_BUFFER);
    parallel_cuda_context_->ErrorCheck("cudaMemcpyAsync", 
                                                cudaMemcpyAsync(host_buffer,
                                                buffer_data_at_rank_offset,
                                                host_buffer_len, 
                                                cudaMemcpyDeviceToHost,
                                                stream));
    timeline.ActivityEndAll(entries);
  }

  // at here the data has been copy to CPU buffer, and will put the MPI allreduce to MPI thread
  // first get the end thread GPU buffer
  void *end_buffer_data = nullptr;

  if (entries.size() > 1) {
    auto &end_buffer = global_state_->fusion_buffer.GetEndBuffer(first_entry.device, first_entry.context->framework())
    
    end_buffer_data = const_cast<void*>(end_buffer->AccessData(first_entry.context));
  } else {
    end_buffer_data = (void*)first_entry.output->data();
  }

  mpi_queue_.enqueue([this, 
                      entries, 
                      host_buffer, 
                      end_buffer_data, 
                      end_stream,
                      end_nccl_comm,
                      num_elements_per_rank, 
                      num_elements_remain, 
                      element_size] {

    this->ExecuteMPIAllReduce(entries, 
                              host_buffer,
                              end_buffer_data,
                              end_stream,
                              end_nccl_comm,
                              num_elements_per_rank,
                              num_elements_remain,
                              element_size);
  });

  return Status::InProgress();
}

Status ParallelNCCLHierarchicalAllreduce::ExecuteMPIAllReduce(std::vector<TensorTableEntry> &entries, 
                                                          void *host_buffer, 
                                                          void *end_buffer_data, 
                                                          cudaStream_t end_stream,
                                                          ncclComm_t end_nccl_comm,
                                                          int64_t num_elements_per_rank,
                                                          int64_t num_elements_remain,
                                                          int element_size) {
  bool is_last_rank = (global_state_->local_size - 1 == global_state_->local_rank);

  if (global_state_->is_homogeneous || is_last_rank) {
    auto &timeline    = global_state_->timeline;
    auto &first_entry = entries[0];

    int64_t total_num_elements = 
                  is_last_rank 
                  ? (num_elements_per_rank + num_elements_remain) 
                  : num_elements_per_rank;

    timeline.ActivityStartAll(entries, MPI_ALLREDUCE);

    int op = MPI_Allreduce(MPI_IN_PLACE, 
                           host_buffer,
                           (int) total_num_elements,
                           parallel_mpi_context_->GetMPIDataType(first_entry.tensor),
                           parallel_mpi_context_->GetMPISumOp(first_entry.tensor->dtype()),
                           parallel_mpi_context_->GetMPICommunicator(Communicator::CROSS));

    if (op != MPI_SUCCESS) {
      throw std::logic_error("MPI_Allreduce failed, see MPI output for details.");
    }

    timeline.ActivityEndAll(entries);
  }

  // put left GPU task in end thread
  end_queue_.enqueue([this, 
                      entries, 
                      host_buffer, 
                      end_buffer_data, 
                      end_stream,
                      end_nccl_comm, 
                      num_elements_per_rank, 
                      num_elements_remain, 
                      element_size] {

    this->ExecuteEnd(entries, 
                    host_buffer,
                    end_buffer_data,
                    end_stream,
                    end_nccl_comm,
                    num_elements_per_rank,
                    num_elements_remain,
                    element_size);
  });

  return Status::InProgress();
}

Status ParallelNCCLHierarchicalAllreduce::ExecuteEnd(std::vector<TensorTableEntry> &entries, 
                                                  void *host_buffer, 
                                                  void *end_buffer_data, 
                                                  cudaStream_t end_stream,
                                                  ncclComm_t end_nccl_comm,
                                                  int64_t num_elements_per_rank,
                                                  int64_t num_elements_remain,
                                                  int element_size) {
  auto &timeline = global_state.timeline;
  auto &first_entry = entries[0];

  // set cuda device
  auto cuda_result = cudaSetDevice(first_entry.device);
  parallel_cuda_context->ErrorCheck("cudaSetDevice", cuda_result);

  int last_rank = global_state_->local_size - 1;
  bool is_last_rank = (global_state_->local_size - 1 == global_state_->local_rank);
  
  if (global_state_->is_homogeneous || is_last_rank) {
    int64_t host_buffer_len = 
                is_last_rank 
                ? ((num_elements_per_rank + num_elements_remaining) * element_size)
                : (num_elements_per_rank * element_size);

    void *end_buffer_data_at_offset = (uint8_t*)end_buffer_data 
                                      + global_state_->local_rank * num_elements_per_rank * element_size;

    // copy data back to GPU
    timeline.ActivityStartAll(entries, MEMCPY_OUT_HOST_BUFFER);
    cuda_context_->ErrorCheck("cudaMemcpyAsync", cudaMemcpyAsync(
                                                      end_buffer_data_at_offset, 
                                                      host_buffer,
                                                      host_buffer_len, 
                                                      cudaMemcpyHostToDevice,
                                                      end_stream));
    timeline.ActivityEndAll(entries);

    // free host memory
    free(host_buffer);
  }

  // create a end event queue to record timeline
  auto end_event_queue = std::queue<std::pair<std::string, cudaEvent_t>>();

  if (num_elements_per_rank > 0) {
    void *end_buffer_data_at_offset = (uint8_t*)end_buffer_data 
                                      + global_state_->local_rank * num_elements_per_rank * element_size;

    parallel_nccl_context_->ErrorCheck("ncclAllGather",
                              ncclAllGather(end_buffer_data_at_offset, 
                                            end_buffer_data,
                                            (size_t) num_elements_per_rank,
                                            GetNCCLDataType(first_entry.tensor),
                                            end_nccl_comm, 
                                            end_stream));
    if (global_state_->timeline.Initialized()) {
      parallel_cuda_context_->RecordEvent(end_event_queue, NCCL_ALLGATHER, end_stream);
    }
  }

  if (num_elements_remain > 0) {
    void *end_buffer_data_remainder = (uint8_t*)end_buffer_data 
                                + global_state.local_size * num_elements_per_rank * element_size;

    parallel_nccl_context_->ErrorCheck("ncclBcast",
                              ncclBcast(end_buffer_data_remainder,
                                        (size_t) num_elements_remaining,
                                        GetNCCLDataType(first_entry.tensor), 
                                        last_rank,
                                        end_nccl_comm, 
                                        end_stream));

    if (global_state_->timeline.Initialized()) {
      parallel_cuda_context_->RecordEvent(end_event_queue, NCCL_BCAST, end_stream);
    }
  }

  if (entries.size() > 1) {
    MemcpyOutFusionBuffer(end_buffer_data, entries);

    if (global_state_->timeline.Initialized()) {
      parallel_cuda_context_->RecordEvent(end_event_queue, MEMCPY_OUT_FUSION_BUFFER, end_stream);
    }
  }

  // wait all cuda event finish
  parallel_cuda_context_->RecordEvent(end_event_queue, "", end_stream);
  parallel_cuda_context_->WaitForEvents(end_event_queue, entries, timeline);

  // call tensor callback
  for (auto &e : entries) {
    timeline.End(e.tensor_name, e.output);
    e.callback(Status::OK());
  }

  return Status::InProgress();
}

} // namespace common
} // namespace horovod






















