//TODO license
#include "adasum_cuda_operations.h"
#include <boost/asio/post.hpp>

namespace horovod {
namespace common {

std::unordered_map<std::thread::id, std::array<double*, 3>> AdasumCudaAllreduceOp::thread_to_device_variable_map;

AdasumCudaAllreduceOp::AdasumCudaAllreduceOp(MPIContext* mpi_context, CUDAContext* cuda_context, HorovodGlobalState* global_state)
    : AdasumMPIOp(mpi_context, global_state), cuda_context_(cuda_context) {
}

AdasumCudaAllreduceOp::~AdasumCudaAllreduceOp() {
  FinalizeCUDA();
}

void AdasumCudaAllreduceOp::InitCUDA(const TensorTableEntry& entry, int layerid) {

  auto thread_id = std::this_thread::get_id();
  cuda_context_->ErrorCheck("cudaSetDevice", cudaSetDevice(entry.device));

  // Ensure stream is in the map before executing reduction.
  cudaStream_t& stream = cuda_context_->streams[global_state_->current_nccl_stream][layerid % global_state_->num_adasum_threads];
  if (stream == nullptr) {

    std::lock_guard<std::mutex> guard(global_state_->buffer_lock);
    if (stream == nullptr) {
      int greatest_priority;
      cuda_context_->ErrorCheck("cudaDeviceGetStreamPriorityRange",
                                cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
      cuda_context_->ErrorCheck("cudaStreamCreateWithPriority",
                                cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest_priority));
    }
  }
  cudaStream_t& device_stream = cuda_context_->streams[global_state_->current_nccl_stream][entry.device];
  if (device_stream == nullptr) {
    std::lock_guard<std::mutex> guard(global_state_->buffer_lock);
    if (stream == nullptr) {
      int greatest_priority;
      cuda_context_->ErrorCheck("cudaDeviceGetStreamPriorityRange",
                                cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
      cuda_context_->ErrorCheck("cudaStreamCreateWithPriority",
                                cudaStreamCreateWithPriority(&device_stream, cudaStreamNonBlocking, greatest_priority));
    }
  }
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
    std::lock_guard<std::mutex> guard(global_state_->buffer_lock);
    thread_to_device_variable_map[thread_id][0] = device_normsq_memory_a;
    thread_to_device_variable_map[thread_id][1] = device_normsq_memory_b;
    thread_to_device_variable_map[thread_id][2] = device_dot_product_memory;
  }
}

void AdasumCudaAllreduceOp::FinalizeCUDA() {
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

Status AdasumCudaAllreduceOp::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
      if(entries.size() < 1) {
      return Status::OK();
  }
  int layerid = 0;
  int num_reductions = entries.size();
  global_state_->finished_parallel_reductions = 0;
  for (auto& entry : entries) {
    boost::asio::post(*global_state_->background_thread_pool,
    [this, &entry, response, layerid, &entries]
    {
      void* buffer_data;
      int buffer_len;
      void* recv_buffer;

      buffer_data = (void*) entry.tensor->data();

      buffer_len = entry.output->size();

      FusionBufferManager buffer_manager;

      if(entry.tensor->data() == entry.output->data()) {

          // Get the temp buffer to be used for the Op
          global_state_->buffer_lock.lock();
          assert(!global_state_->temp_buffers.empty());
          buffer_manager = global_state_->temp_buffers.front();
          global_state_->temp_buffers.pop();
          global_state_->buffer_lock.unlock();

          // TODO: Maybe add before and after callbacks to timeline?
          Status status = buffer_manager.InitializeBuffer(
              buffer_len,
              entry.device, entry.context,
              global_state_->current_nccl_stream,
              [](){},
              [](){},
              [](int64_t& size, int64_t& threshold){return size >= threshold;});

          if (!status.ok()) {
              throw std::logic_error("AdaSumCudaAllreduceOp::Execute: Initialize buffer failed.");
              return;
          }
          auto buffer = buffer_manager.GetBuffer(entry.device, entry.context->framework(), global_state_->current_nccl_stream);
          recv_buffer = const_cast<void*>(buffer->AccessData(entry.context));
      }
      else {
          recv_buffer = (void*) entry.output->data();
      }
      
      MPI_Comm* node_comm = NULL;
      if (global_state_->rank_log_size != 0) {
          node_comm = &global_state_->reduction_comms[global_state_->rank_log_size-1];
      }
    
      // This will create a stream per layer.
      InitCUDA(entry, layerid);
      AdasumInternal(buffer_data,
                     recv_buffer,
                     node_comm,
                     global_state_->reduction_comms,
                     global_state_->local_comm,
                     layerid,
                     entry);  

      if(entry.tensor->data() == entry.output->data()) {
        // Return the buffer back into the pool of available buffers
        global_state_->buffer_lock.lock();
        global_state_->temp_buffers.push(buffer_manager);
        global_state_->buffer_lock.unlock();
      }
      else {
        MemcpyUtil(entry, (void *) entry.output->data(), (void *) entry.tensor->data(), (size_t) entry.tensor->size(), layerid);
      }
      global_state_->finished_parallel_reductions++;
    });
    layerid++;
  }
  while (global_state_->finished_parallel_reductions < num_reductions) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(50));
  }
  return Status::OK();
}

void AdasumCudaAllreduceOp::AdasumInternal(void* gradient_buffer,
                    void* recv_buffer,
                    MPI_Comm* node_comm,
                    MPI_Comm* reduction_comm_pool,
                    MPI_Comm local_comm,
                    int layerid,
                    TensorTableEntry entry) {
  int count = entry.tensor->size() / GetPerElementSize(entry);
  int local_rank = 0;
  local_rank = GetLocalRankWithComm(local_comm);
  // TODO factor ring local reduce in here and choose algorithms based on tensor size, topology and etc
  if(global_state_->adasum_algorithm == AdasumAlgorithm::GPU_TREE){
    SyncLocalReduce(gradient_buffer, recv_buffer, local_comm, layerid, entry);
  }
  else if(global_state_->adasum_algorithm == AdasumAlgorithm::GPU_RING)
  {
    throw std::logic_error("GPU Ring is not supportedf yet. Coming soon.");
  }
  else if(global_state_->adasum_algorithm == AdasumAlgorithm::GPU_RING)
  {
    throw std::logic_error("GPU Ring is not supportedf yet. Coming soon.");
  }
  else if(global_state_->adasum_algorithm == AdasumAlgorithm::GPU_NCCL_RING)
  {
    throw std::logic_error("GPU NCCL Ring is not supportedf yet. Coming soon.");
  }
  else
  {
    throw std::logic_error("Unsupported adasum reduction algorithm");
  }

  // TODO have a version of VHDD that performs asynchronously
  if (local_rank == 0 && node_comm != NULL) {
    switch(entry.tensor->dtype()) {
        case DataType::HOROVOD_FLOAT16:
          SyncAllreduce((uint16_t*)gradient_buffer, (uint16_t*)recv_buffer, *node_comm, reduction_comm_pool, layerid, entry);
          break;
        case DataType::HOROVOD_FLOAT32:
          SyncAllreduce((float*)gradient_buffer, (float*)recv_buffer, *node_comm, reduction_comm_pool, layerid, entry);
          break;
        case DataType::HOROVOD_FLOAT64:
          SyncAllreduce((double*)gradient_buffer, (double*)recv_buffer, *node_comm, reduction_comm_pool, layerid, entry);
          break;
        default:
          throw std::logic_error("Unsupported data type");
    }
  }
  SyncLocalBroadcast(gradient_buffer, local_comm, entry, layerid);

}

void AdasumCudaAllreduceOp::NcclHierarchical(std::vector<TensorTableEntry>& entries
                    MPI_Comm* node_comm,
                    MPI_Comm* reduction_comm_pool) {
  auto& first_entry = entries[0];

  // Determine GPU IDs of the devices participating in this communicator.
  std::vector<int32_t> nccl_device_map;
  nccl_device_map.reserve(
      global_state_->controller->GetLocalCommRanks().size());
  for (int rank : global_state_->controller->GetLocalCommRanks()) {
    nccl_device_map.push_back(response.devices()[rank]);
  }

  InitCUDA(entries);
  InitNCCLComm(entries, nccl_device_map);
  InitCUDAQueue(entries, response);

  bool is_root_rank = global_state_->controller->GetLocalRank() == 0;

  std::vector<std::unique<char[]>> host_buffers;
  std::vector<cudaEvent_t> events(entries.size());

  for (size_t i = 0; i < entries.size(); ++i) {
    auto& entry = entries.at(i);
    const void* input_data = entry.tensor->data();
    void* buffer_data = (void*) entry.output->data();
    size_t buffer_len = (size_t) entry.output->size();
    int num_elements = entry.tensor->shape().num_elements();

    auto nccl_result = ncclReduce(input_data,
                                  buffer_data,
                                  (size_t) num_elements,
                                  GetNCCLDataType(entry.tensor),
                                  ncclSum, 0, *nccl_comm_, *stream_);
    nccl_context_->ErrorCheck("ncclReduce", nccl_result);

    if (is_root_rank) {
      host_buffers.emplace_back(new char[buffer_len]);
      void* host_buffer = (void*)host_buffers[i].data();

      cuda_context_->ErrorCheck("cudaMemcpyAsync",
                                cudaMemcpyAsync(host_buffer, buffer_data,
                                                buffer_len, cudaMemcpyDeviceToHost,
                                                *stream_));

      auto& event = events[i];
      cuda_context_->ErrorCheck("GetCudaEvent", cuda_context_->GetCudaEvent(&event));
      cuda_context_->ErrorCheck("cudaEventRecord", cudaEventRecord(event, *stream_));
    }
  }

  if (is_root_rank) {
    std::vector<char> recv_buffer;
    for (size_t i = 0; i < entries.size(); ++i) {
      auto& entry = entries.at(i);
      void* buffer_data = (void*) entry.output->data();
      size_t buffer_len = (size_t) entry.output->size();
      int num_elements = entry.tensor->shape().num_elements();
      auto host_buffer = (void*)host_buffers[i].data();
      auto& event = events[i];

      cuda_context_->ErrorCheck("cudaEventSynchronize", cudaEventSynchronize(event));
      cuda_context_->ErrorCheck("ReleaseCudaEvent", cuda_context_->ReleaseCudaEvent(event));

      recv_buffer.resize(buffer_len);
      DispatchSyncAllreduce(host_buffer, recv_buffer.data(), node_comm, reduction_comm_pool, i, entry);

      cuda_context_->ErrorCheck("cudaMemcpyAsync",
                                cudaMemcpyAsync(buffer_data, host_buffer,
                                                buffer_len, cudaMemcpyHostToDevice,
                                                *stream_));
    }
  }

  for (size_t i = 0; i < entries.size(); ++i) {
    auto& entry = entries.at(i);
    void* buffer_data = (void*) entry.output->data();
    int num_elements = entry.tensor->shape().num_elements();

    nccl_context_->ErrorCheck("ncclBcast",
                              ncclBcast(buffer_data,
                                        (size_t) num_elements,
                                        GetNCCLDataType(entry.tensor),
                                        0,
                                        *nccl_comm_, *stream_));
  }

  FinalizeCUDAQueue(entries);
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

void AdasumCudaAllreduceOp::ComputeDotAndNormSqrdsWrapper(const void* __restrict__ a,
                                                          const void* __restrict__ b,
                                                          DataType horovod_datatype,
                                                          int count,
                                                          double& dotProduct,
                                                          double& anormsq,
                                                          double& bnormsq,
                                                          HorovodGlobalState *global_state,
                                                          int layerid) {
  if (horovod_datatype == DataType::HOROVOD_FLOAT16) {
    DotProductImpl((uint16_t*)a, (uint16_t*)b, count, dotProduct, anormsq, bnormsq, global_state_, layerid);
  }
  else if (horovod_datatype == DataType::HOROVOD_FLOAT32) {
    DotProductImpl((float*)a, (float*)b, count, dotProduct, anormsq, bnormsq, global_state_, layerid);
  }
  else if (horovod_datatype == DataType::HOROVOD_FLOAT32) {
    DotProductImpl((double*)a, (double*)b, count, dotProduct, anormsq, bnormsq, global_state_, layerid);      
  }
  else {
      throw std::logic_error("Unsupported data type.");
  }

}

void AdasumCudaAllreduceOp::ScaledAddWrapper(DataType horovod_datatype,
                                             int count,
                                             double acoeff,
                                             void* __restrict__ a,
                                             double bcoeff,
                                             void* __restrict__ b,
                                             HorovodGlobalState *global_state,
                                             int layerid) {
  if (horovod_datatype == DataType::HOROVOD_FLOAT16) {
    ScaleAddImpl(count, acoeff, (uint16_t*)a, bcoeff, (uint16_t*)b, global_state_, layerid);
  }
  else if (horovod_datatype == DataType::HOROVOD_FLOAT32) {
    ScaleAddImpl(count, acoeff, (float*)a, bcoeff, (float*)b, global_state_, layerid);
  }
  else if (horovod_datatype == DataType::HOROVOD_FLOAT32) {
    ScaleAddImpl(count, acoeff, (double*)a, bcoeff, (double*)b, global_state_, layerid);
  }
  else {
      throw std::logic_error("Unsupported data type.");
  }
  
}

bool AdasumCudaAllreduceOp::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}
}
}
