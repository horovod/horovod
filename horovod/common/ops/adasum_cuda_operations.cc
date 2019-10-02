//TODO license
#include "adasum_cuda_operations.h"
#include <boost/asio/post.hpp>

namespace horovod {
namespace common {

std::unordered_map<std::thread::id, std::array<double*, 3>> AdasumCudaAllreduceOp::thread_to_device_variable_map;

#if HAVE_NCCL
AdasumCudaAllreduceOp::AdasumCudaAllreduceOp(MPIContext* mpi_context, NCCLContext* nccl_context, CUDAContext* cuda_context, HorovodGlobalState* global_state)
    : AdasumMPIOp(mpi_context, global_state), nccl_context_(nccl_context), cuda_context_(cuda_context) {
}
#endif

AdasumCudaAllreduceOp::AdasumCudaAllreduceOp(MPIContext* mpi_context, CUDAContext* cuda_context, HorovodGlobalState* global_state)
    : AdasumMPIOp(mpi_context, global_state), cuda_context_(cuda_context) {
}

AdasumCudaAllreduceOp::~AdasumCudaAllreduceOp() {
  FreeDeviceVariables();
}

void AdasumCudaAllreduceOp::InitCUDAStreams(const std::vector<TensorTableEntry> entries) {

  auto first_entry = entries[0];

  // Ensure streams are in the map before executing reduction.
  for(int i = 0; i < entries.size(); i++) {
    cuda_context_->ErrorCheck("cudaSetDevice", cudaSetDevice(entries[i].device));
    cudaStream_t& stream = cuda_context_->streams[global_state_->current_nccl_stream][i];
    if (stream == nullptr) {

      int greatest_priority;
      cuda_context_->ErrorCheck("cudaDeviceGetStreamPriorityRange",
                                cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
      cuda_context_->ErrorCheck("cudaStreamCreateWithPriority",
                                cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest_priority));
    }
  }
  cudaStream_t& device_stream = cuda_context_->streams[global_state_->current_nccl_stream][first_entry.device];
  if (device_stream == nullptr) {
    int greatest_priority;
    cuda_context_->ErrorCheck("cudaDeviceGetStreamPriorityRange",
                              cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
    cuda_context_->ErrorCheck("cudaStreamCreateWithPriority",
                              cudaStreamCreateWithPriority(&device_stream, cudaStreamNonBlocking, greatest_priority));
  }
}

void AdasumCudaAllreduceOp::InitDeviceVariables(TensorTableEntry entry) {
  auto thread_id = std::this_thread::get_id();
  cuda_context_->ErrorCheck("cudaSetDevice", cudaSetDevice(entry.device));
  if (thread_to_device_variable_map.find(thread_id) == thread_to_device_variable_map.end() &&
      thread_to_device_variable_map[thread_id][0] == nullptr)
  {
    double* device_normsq_memory_a;
    double* device_normsq_memory_b;
    double* device_dot_product_memory;
    cuda_context_->ErrorCheck("cudaMalloc",
                            cudaMalloc(&device_normsq_memory_a, sizeof(double)));
    cuda_context_->ErrorCheck("cudaMalloc",
                            cudaMalloc(&device_normsq_memory_b, sizeof(double)));
    cuda_context_->ErrorCheck("cudaMalloc",
                            cudaMalloc(&device_dot_product_memory, sizeof(double)));
    std::lock_guard<std::mutex> guard(buffer_lock_);
    thread_to_device_variable_map[thread_id][0] = device_normsq_memory_a;
    thread_to_device_variable_map[thread_id][1] = device_normsq_memory_b;
    thread_to_device_variable_map[thread_id][2] = device_dot_product_memory;
  }
}

void AdasumCudaAllreduceOp::FreeDeviceVariables() {
  if (!thread_to_device_variable_map.empty()){
    for (auto it = thread_to_device_variable_map.begin(); it != thread_to_device_variable_map.end(); ++it)
    {
      if(it->second[0] != nullptr) {
        cudaFree(it->second[0]);
        cudaFree(it->second[1]);
        cudaFree(it->second[2]);
      }
    }
  }
}

Status AdasumCudaAllreduceOp::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  if(entries.empty()) {
    return Status::OK();
  }
  InitCUDAStreams(entries);
  if(global_state_->adasum_algorithm == AdasumAlgorithm::GPU__TREE) {
    LOG(TRACE) << "Reducing with Adasum algorithm GPU_TREE.";
    return TreeHierarchical(entries, response);
  }
  else if(global_state_->adasum_algorithm == AdasumAlgorithm::GPU__RING) {
    return RingHierarchical(entries, response);
  }
  else if(global_state_->adasum_algorithm == AdasumAlgorithm::GPU__NCCL__LOCAL__AVG) {
#if HAVE_NCCL
    return NcclHierarchical(entries, response);
#else
    throw std::logic_error("GPU_NCCL_LOCAL_AVG needs NCCL to be available. PLease re-build Horovod with HOROVOD_GPU_ALLREDUCE=NCCL.");
#endif
  }
  else {
    throw std::logic_error("Unsupported adasum reduction algorithm");
  }  
}

#if HAVE_NCCL
void AdasumCudaAllreduceOp::InitNCCLComm(const std::vector<TensorTableEntry>& entries,
                                         const std::vector<int32_t>& nccl_device_map) {
  // Ensure NCCL communicator is in the map before executing reduction.
  ncclComm_t& nccl_comm = nccl_context_->nccl_comms[global_state_->current_nccl_stream][nccl_device_map];
  if (nccl_comm == nullptr) {
    auto& timeline = global_state_->timeline;
    timeline.ActivityStartAll(entries, INIT_NCCL);

    int nccl_rank = global_state_->controller->GetLocalRank();
    int nccl_size = global_state_->controller->GetLocalSize();
    Communicator nccl_id_bcast_comm = Communicator::LOCAL;

    ncclUniqueId nccl_id;
    if (nccl_rank == 0) {
      nccl_context_->ErrorCheck("ncclGetUniqueId", ncclGetUniqueId(&nccl_id));
    }

    global_state_->controller->Bcast((void*)&nccl_id, sizeof(nccl_id), 0,
                                     nccl_id_bcast_comm);

    ncclComm_t new_nccl_comm;
    auto nccl_result = ncclCommInitRank(&new_nccl_comm, nccl_size, nccl_id, nccl_rank);
    nccl_context_->ErrorCheck("ncclCommInitRank", nccl_result);
    nccl_comm = new_nccl_comm;

    // Barrier helps NCCL to synchronize after initialization and avoid
    // deadlock that we've been seeing without it.
    global_state_->controller->Barrier(Communicator::GLOBAL);

    timeline.ActivityEndAll(entries);
  }

  nccl_comm_ = &nccl_comm;
}

void AdasumCudaAllreduceOp::InitCUDAQueue(const std::vector<TensorTableEntry>& entries, const Response& response) {
  event_queue_ = std::queue<std::pair<std::string, cudaEvent_t>>();
  stream_ = &cuda_context_->streams[global_state_->current_nccl_stream][entries[0].device];
  host_buffer_ = nullptr;

  if (global_state_->timeline.Initialized()) {
    cuda_context_->RecordEvent(event_queue_, QUEUE, *stream_);
  }
}

Status AdasumCudaAllreduceOp::FinalizeCUDAQueue(const std::vector<TensorTableEntry>& entries) {
  // Use completion marker via event because it's faster than
  // blocking cudaStreamSynchronize() in this thread.
  cuda_context_->RecordEvent(event_queue_, "", *stream_);

  auto& first_entry = entries[0];
  void* host_buffer = host_buffer_;
  auto& event_queue = event_queue_;
  auto& timeline = global_state_->timeline;
  auto& cuda_context = cuda_context_;

  // Claim a std::shared_ptr to the fusion buffer to prevent its memory from being reclaimed
  // during finalization.
  auto fusion_buffer = global_state_->fusion_buffer.GetBuffer(
      first_entry.device, first_entry.context->framework(), global_state_->current_nccl_stream);

  // TODO: use thread pool or single thread for callbacks
  std::thread finalizer_thread([entries, first_entry, host_buffer, fusion_buffer,
                                event_queue, &timeline, &cuda_context]() mutable {
    auto cuda_result = cudaSetDevice(first_entry.device);
    cuda_context->ErrorCheck("cudaSetDevice", cuda_result);

    cuda_context->WaitForEvents(event_queue, entries, timeline);
    if (host_buffer != nullptr) {
      free(host_buffer);
    }

    for (auto& e : entries) {
      timeline.End(e.tensor_name, e.output);
      e.callback(Status::OK());
    }
  });

  finalizer_thread.detach();

  // Update current stream
  global_state_->current_nccl_stream = (global_state_->current_nccl_stream + 1) %
                                  global_state_->num_nccl_streams;

  return Status::InProgress();
}

void AdasumCudaAllreduceOp::MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                              const TensorTableEntry& e, void* buffer_data_at_offset) {
  auto& first_entry = entries[0];
  auto cuda_result = cudaMemcpyAsync(buffer_data_at_offset, e.tensor->data(),
                                     (size_t) e.tensor->size(), cudaMemcpyDeviceToDevice,
                                     cuda_context_->streams[global_state_->current_nccl_stream][first_entry.device]);
  cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
}

void AdasumCudaAllreduceOp::MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                               const void* buffer_data_at_offset, TensorTableEntry& e) {
  auto& first_entry = entries[0];
  auto cuda_result = cudaMemcpyAsync((void*) e.output->data(), buffer_data_at_offset,
                                     (size_t) e.tensor->size(), cudaMemcpyDeviceToDevice,
                                     cuda_context_->streams[global_state_->current_nccl_stream][first_entry.device]);
  cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
}

Status AdasumCudaAllreduceOp::NcclHierarchical(std::vector<TensorTableEntry>& entries,
                                               const Response& response) {
  auto& first_entry = entries[0];

  // Determine GPU IDs of the devices participating in this communicator.
  std::vector<int32_t> nccl_device_map;
  nccl_device_map.reserve(
      global_state_->controller->GetLocalCommRanks().size());
  for (int rank : global_state_->controller->GetLocalCommRanks()) {
    nccl_device_map.push_back(response.devices()[rank]);
  }

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
  int local_size = global_state_->controller->GetLocalSize();
  int local_rank = global_state_->controller->GetLocalRank();

  // If cluster is homogeneous and we are using fusion buffer, include
  // dummy elements from the buffer (if necessary) to make sure the data
  // is divisible by local_size. This is always possible since we
  // set the fusion buffer size divisible by local_size.
  if (global_state_->controller->IsHomogeneous() && entries.size() > 1) {
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

  int64_t num_elements_per_rank = global_state_->controller->IsHomogeneous()
                                      ? num_elements / local_size
                                      : 0;

  size_t buffer_len_per_rank = element_size * num_elements_per_rank;

  void* buffer_data_at_rank_offset =
      (uint8_t*)buffer_data + buffer_len_per_rank * local_rank;

  int64_t num_elements_remaining = global_state_->controller->IsHomogeneous()
                                       ? num_elements % local_size
                                       : num_elements;

  size_t buffer_len_remaining = element_size * num_elements_remaining;

  void* buffer_data_remainder =
      (uint8_t*)buffer_data + buffer_len_per_rank * local_size;

  void* fused_input_data_remainder =
      (uint8_t*)fused_input_data + buffer_len_per_rank * local_size;

  int root_rank =
      global_state_->controller->IsHomogeneous() ? local_size - 1 : 0;
  bool is_root_rank = local_rank == root_rank;

  int64_t total_num_elements =
      is_root_rank ? num_elements_per_rank + num_elements_remaining
                   : num_elements_per_rank;
  int64_t total_buffer_len = is_root_rank
                                 ? buffer_len_per_rank + buffer_len_remaining
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

  if (global_state_->controller->IsHomogeneous() || is_root_rank) {
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

    // Since Adasum is not a per-element operation, an allreduce for fused
    // tensors needs to know boundaries of tensors. Calculate here the count
    // of elements for each tensor owned by this rank.
		std::vector<int> tensor_counts(entries.size());
		if (global_state_->controller->IsHomogeneous()) {
      // For homogeneous clusters each rank owns a slice of the fused tensor.

			int64_t num_elements_sofar = 0;
			int i = 0;
			for (auto& e : entries) {
				int64_t e_num_elements = e.tensor->shape().num_elements();
				int64_t left_boundary  = std::max(num_elements_sofar, local_rank * num_elements_per_rank);
				int64_t right_boundary = std::min(num_elements_sofar + e_num_elements, (local_rank+1) * num_elements_per_rank);
				tensor_counts[i] = std::max(right_boundary - left_boundary, (int64_t)0);
				if (is_root_rank) {
					if (num_elements_sofar + e_num_elements >= local_size * num_elements_per_rank){
						left_boundary  = std::max(num_elements_sofar, local_size * num_elements_per_rank);
						right_boundary = num_elements_sofar + e_num_elements;
						tensor_counts[i] += std::max(right_boundary - left_boundary, (int64_t)0);
					}
				}

				num_elements_sofar += e_num_elements;
				i++;
			}
		} else {
      // For non-homogeneous clusters the root rank owns everything.

			if (is_root_rank) {
				int i = 0;
				for (auto& e : entries) {
					int e_num_elements = e.tensor->shape().num_elements();
					tensor_counts[i] = e_num_elements;
					i++;
				}
			}
		}

    auto recv_buffer = std::unique_ptr<char[]>(new char[total_buffer_len]);
    DispatchFusedAllreduce(host_buffer_, recv_buffer.get(), tensor_counts,
                      local_size, // start_level
                      global_state_->controller->IsHomogeneous() ?
                        MPI_COMM_WORLD :
                        mpi_context_->GetMPICommunicator(Communicator::CROSS),
                      0,
                      world_reduction_comms_,
                      first_entry.tensor->dtype());
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
#endif

Status AdasumCudaAllreduceOp::RingHierarchical(std::vector<TensorTableEntry>& entries,
                        const Response& response) {

int num_reductions = entries.size();
AllRings all_rings(global_state_->controller->GetLocalRank(), global_state_->controller->GetLocalSize());
std::deque<FusionBufferManager> used_buffer_managers;

int local_rank = 0;

bool need_pipeline = global_state_->controller->GetSize() > global_state_->controller->GetLocalSize();

MPI_Comm_rank(mpi_context_->local_comm, &local_rank);
bool do_cross_node = local_rank == 0 && rank_log_size_ != 0;

size_t unroll_size = need_pipeline ? 8 : num_reductions;
size_t layerid = 0;
size_t elements_left = num_reductions;
finished_parallel_reductions_ = 0;
while(elements_left > 0) {
  size_t increment_count = std::min(unroll_size, elements_left);
  size_t start_index = layerid;
  //enqueue messages
  for (; layerid < start_index + increment_count; ++layerid) {
    auto& entry = entries.at(layerid);
    void* buffer_data;
    int buffer_len;
    void* recv_buffer;
    buffer_data = (void*) entry.tensor->data();
    buffer_len = entry.output->size();
    if(entry.tensor->data() == entry.output->data()) {
      // Get the temp buffer to be used for the Op
      FusionBufferManager buffer_manager;
      if (!temp_buffers_.empty()) {
        buffer_manager = temp_buffers_.front();
        temp_buffers_.pop_front();
      }
      used_buffer_managers.push_back(buffer_manager);
      // TODO: Maybe add before and after callbacks to timeline?
      Status status = buffer_manager.InitializeBuffer(
          buffer_len,
          entry.device, entry.context,
          global_state_->current_nccl_stream,
          []() {},
          []() {},
          [](int64_t& size, int64_t& threshold) {return size >= threshold;});
      if (!status.ok()) {
          throw std::logic_error("RingHierarchical: Initialize buffer failed.");
      }
      auto buffer = buffer_manager.GetBuffer(entry.device, entry.context->framework(), global_state_->current_nccl_stream);
      recv_buffer = const_cast<void*>(buffer->AccessData(entry.context));
    }
    else {
      recv_buffer = (void*) entry.output->data();
    }
    cuda_context_->ErrorCheck("cudaSetDevice", cudaSetDevice(entry.device));

    all_rings.InitMessageInRing(new ReduceMessage(mpi_context_),
                      buffer_data,
                      recv_buffer,
                      buffer_len,
                      entry.tensor->dtype(),
                      mpi_context_->local_comm,
                      layerid,
                      global_state_->controller->GetLocalRank());
  }
  // wait for messages to finish
  all_rings.WaitAllMessages();
  // Return used buffer managers to the queue
  temp_buffers_.insert(temp_buffers_.end(), used_buffer_managers.begin(), used_buffer_managers.end());

  if (do_cross_node) {
    boost::asio::post(*global_state_->adasum_background_thread_pool,
    [this,&entries, start_index, increment_count]
    {
      std::vector<std::unique_ptr<char[]>> allreduce_buffers;
      // start device to host copies
      for (size_t index = start_index, i = 0; index < start_index + increment_count; ++index, ++i) {
        auto& entry = entries.at(index);
        int buffer_len = entry.output->size();
        allreduce_buffers.emplace_back(new char[buffer_len]);
        char* buffer_data = allreduce_buffers.at(i).get();
        
        auto cuda_result = cudaMemcpyAsync(
          buffer_data, (void*) entry.tensor->data(),
          buffer_len, 
          cudaMemcpyDeviceToHost,
          cuda_context_->streams[global_state_->current_nccl_stream][index]);
        cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
      }
      for (size_t index = start_index, i = 0; index < start_index + increment_count; ++index, ++i) {
        auto& entry = entries.at(index);
        int buffer_len = entry.output->size();
        char* buffer_data = allreduce_buffers.at(i).get();
        std::unique_ptr<char[]> recv_buffer(new char[buffer_len]);
        // wait for this layer to finish copying to host
        auto cuda_result = cudaStreamSynchronize(cuda_context_->streams[global_state_->current_nccl_stream][index]);
        cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);
        MPI_Comm* node_comm = &reduction_comms_[rank_log_size_-1];
        DispatchSyncAllreduce(buffer_data, recv_buffer.get(), node_comm, reduction_comms_, index, entry);
        // start the copy back to device
        cuda_result = cudaMemcpyAsync(
          (void*) entry.tensor->data(), buffer_data,
          buffer_len, 
          cudaMemcpyHostToDevice,
          cuda_context_->streams[global_state_->current_nccl_stream][index]);
        cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
      }
      // wait for all copies to device to finish
      for (size_t index = start_index, i = 0; index < start_index + increment_count; ++index, ++i) {
        auto cuda_result = cudaStreamSynchronize(cuda_context_->streams[global_state_->current_nccl_stream][index]);
        cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);
      }
      finished_parallel_reductions_.fetch_add(increment_count, std::memory_order_relaxed);
    });
  }
  // for ranks that are not doing vhdd, increment finished_parallel_reductions right away
  else {
    finished_parallel_reductions_.fetch_add(increment_count, std::memory_order_relaxed);
  }
  elements_left -= increment_count;
}
// wait for all vhdd to finish
while (finished_parallel_reductions_ < num_reductions) {
  std::this_thread::sleep_for(std::chrono::nanoseconds(25));
}
//ring broadcast
for (size_t layerid = 0; layerid < entries.size(); ++layerid) {
  auto& entry = entries.at(layerid);
  void* buffer_data;
  int buffer_len;
  buffer_data = (void*) entry.tensor->data();
  buffer_len = entry.output->size();

  // This will create a stream per layer.
  all_rings.InitMessageInRing(new BroadcastMessage(mpi_context_),
                    buffer_data,
                    nullptr,
                    buffer_len,
                    entry.output->dtype(),
                    mpi_context_->local_comm,
                    layerid,
                    global_state_->controller->GetLocalRank());
}

all_rings.WaitAllMessages();

for (size_t layerid = 0; layerid < entries.size(); ++layerid) {
  auto& entry = entries.at(layerid);
  if(entry.tensor->data() != entry.output->data()) {
    MemcpyUtil(entry, (void *) entry.output->data(), (void *) entry.tensor->data(), (size_t) entry.tensor->size(), layerid);
  }
}

return Status::OK();
}

void AdasumCudaAllreduceOp::MemcpyUtil(TensorTableEntry entry, void* dest, void* src, size_t buffer_len, int layerid) {
    assert(dest != nullptr);
    assert(src != nullptr);
    auto cuda_result = cudaMemcpyAsync(dest, src,
                                    buffer_len, 
                                    cudaMemcpyDeviceToDevice,
                                    cuda_context_->streams[global_state_->current_nccl_stream][entry.device]);
    cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
    auto cuda_sync_result = cudaStreamSynchronize(cuda_context_->streams[global_state_->current_nccl_stream][entry.device]);
    cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_sync_result);
}

bool AdasumCudaAllreduceOp::CheckPointerLocation(const void* ptr) {
  cudaPointerAttributes attributes;
  auto cuda_result = cudaPointerGetAttributes(&attributes, ptr);
  if(attributes.type == cudaMemoryType::cudaMemoryTypeHost || 
    attributes.type == cudaMemoryType::cudaMemoryTypeUnregistered || 
    cuda_result == cudaErrorInvalidValue) {
    LOG(INFO, global_state_->controller->GetRank())<<"Got a host pointer!";
    return true;
  }
  else {
    LOG(INFO, global_state_->controller->GetRank())<<"Got a device pointer!";
    return false;
  }
}

void AdasumCudaAllreduceOp::DispatchComputeDotAndNormSqrds(const void* __restrict__ a,
                                                          const void* __restrict__ b,
                                                          DataType horovod_datatype,
                                                          int count,
                                                          double& dotProduct,
                                                          double& anormsq,
                                                          double& bnormsq,
                                                          HorovodGlobalState *global_state,
                                                          int layerid) {
  bool use_cpu = CheckPointerLocation(a);

  // we pass through to the cpu dispatcher
  if(use_cpu) {
    AdasumMPIOp::DispatchComputeDotAndNormSqrds(a, b, horovod_datatype, count, dotProduct, anormsq, bnormsq, global_state_, layerid);
  }
  else {
    if (horovod_datatype == DataType::HOROVOD_FLOAT16) {
      DotProductImpl((uint16_t*)a, (uint16_t*)b, count, dotProduct, anormsq, bnormsq, global_state_, layerid);
    }
    else if (horovod_datatype == DataType::HOROVOD_FLOAT32) {
      DotProductImpl((float*)a, (float*)b, count, dotProduct, anormsq, bnormsq, global_state_, layerid);
    }
    else if (horovod_datatype == DataType::HOROVOD_FLOAT64) {
      DotProductImpl((double*)a, (double*)b, count, dotProduct, anormsq, bnormsq, global_state_, layerid);      
    }
    else {
        throw std::logic_error("Unsupported data type.");
    }
  }
}

void AdasumCudaAllreduceOp::DispatchScaledAdd(DataType horovod_datatype,
                                             int count,
                                             double acoeff,
                                             void* __restrict__ a,
                                             double bcoeff,
                                             void* __restrict__ b,
                                             HorovodGlobalState *global_state,
                                             int layerid) {
  bool use_cpu = CheckPointerLocation(a);
  // we pass through to the cpu dispatcher
  if(use_cpu) {
    AdasumMPIOp::DispatchScaledAdd(horovod_datatype, count, acoeff, a, bcoeff, b, global_state_, layerid);
  }
  else {
    if (horovod_datatype == DataType::HOROVOD_FLOAT16) {
      ScaleAddImpl(count, acoeff, (uint16_t*)a, bcoeff, (uint16_t*)b, global_state_, layerid);
    }
    else if (horovod_datatype == DataType::HOROVOD_FLOAT32) {
      ScaleAddImpl(count, acoeff, (float*)a, bcoeff, (float*)b, global_state_, layerid);
    }
    else if (horovod_datatype == DataType::HOROVOD_FLOAT64) {
      ScaleAddImpl(count, acoeff, (double*)a, bcoeff, (double*)b, global_state_, layerid);
    }
    else {
        throw std::logic_error("Unsupported data type.");
    }
  }
}

bool AdasumCudaAllreduceOp::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID 
         && global_state_->adasum_algorithm != AdasumAlgorithm::NONE 
         && global_state_->adasum_algorithm != AdasumAlgorithm::CPU__TREE;

}
}
}