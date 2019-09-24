//TODO license
#include "adasum_cuda_ring_operations.h"

namespace horovod {
namespace common {

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	} 
}

AdasumCudaRingAllreduceOp::AdasumCudaRingAllreduceOp(MPIContext* mpi_context, CUDAContext* cuda_context, HorovodGlobalState* global_state)
    : AdasumCudaAllreduceOp(mpi_context, cuda_context, global_state) {
    }

void AdasumCudaRingAllreduceOp::InitCUDA(const TensorTableEntry& entry, int layerid) {
  cuda_context_->ErrorCheck("cudaSetDevice", cudaSetDevice(entry.device));

  // Ensure stream is in the map before executing reduction.
  cudaStream_t& stream = cuda_context_->streams[global_state_->current_nccl_stream][layerid];
  if (stream == nullptr) {

    std::lock_guard<std::mutex> guard(buffer_lock_);
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
    std::lock_guard<std::mutex> guard(buffer_lock_);
    if (device_stream == nullptr) {
      int greatest_priority;
      cuda_context_->ErrorCheck("cudaDeviceGetStreamPriorityRange",
                                cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
      cuda_context_->ErrorCheck("cudaStreamCreateWithPriority",
                                cudaStreamCreateWithPriority(&device_stream, cudaStreamNonBlocking, greatest_priority));

    }
  }
}

Status AdasumCudaRingAllreduceOp::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  if(entries.size() < 1) {
      return Status::OK();
  }
  //TODO how do we report statuses?
  std::map<int, Status> return_statuses;
  int num_reductions = entries.size();
	AllRings all_rings(global_state_->controller->GetLocalRank(), global_state_->controller->GetLocalSize());
  std::deque<FusionBufferManager> used_buffer_managers;
  for (size_t layerid = 0; layerid < entries.size(); ++layerid) {
    auto& entry = entries.at(layerid);
    void* buffer_data;
    int buffer_len;
    void* recv_buffer;

    buffer_data = (void*) entry.tensor->data();

    buffer_len = entry.output->size();

    if(entry.tensor->data() == entry.output->data()) {

        // Get the temp buffer to be used for the Op
        FusionBufferManager buffer_manager;
        if (!buffer_managers_.empty()) {
          buffer_manager = buffer_managers_.front();
          buffer_managers_.pop_front();
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
            throw std::logic_error("AdaSumOp::Execute_helper: Initialize buffer failed.");
        }
        auto buffer = buffer_manager.GetBuffer(entry.device, entry.context->framework(), global_state_->current_nccl_stream);
        recv_buffer = const_cast<void*>(buffer->AccessData(entry.context));
    }
    else {
        recv_buffer = (void*) entry.output->data();
    }
  
    // This will create a stream per layer.
    InitCUDA(entry, layerid);
    all_rings.InitMessageInRing(new ReduceMessage(mpi_context_),
                      buffer_data,
                      recv_buffer,
                      buffer_len,
                      entry.tensor->dtype(),
                      mpi_context_->local_comm,
                      layerid,
                      global_state_->controller->GetLocalRank());
  }
  all_rings.WaitAllMessages();
  // Return used buffer managers to the queue
  buffer_managers_.insert(buffer_managers_.end(), used_buffer_managers.begin(), used_buffer_managers.end());

  int local_rank = 0;
  MPI_Comm_rank(mpi_context_->local_comm, &local_rank);
  if (local_rank == 0 && rank_log_size_ != 0) {
    std::vector<std::unique_ptr<char[]>> allreduce_buffers;

    // start device to host copies
    for (size_t layerid = 0; layerid < entries.size(); ++layerid) {
      auto& entry = entries.at(layerid);
      int buffer_len = entry.output->size();
      allreduce_buffers.emplace_back(new char[buffer_len]);
      char* buffer_data = allreduce_buffers.at(layerid).get();
      
      auto cuda_result = cudaMemcpyAsync(
        buffer_data, (void*) entry.tensor->data(),
        buffer_len, 
        cudaMemcpyDeviceToHost,
        cuda_context_->streams[global_state_->current_nccl_stream][layerid]);
      cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
    }

    for (size_t layerid = 0; layerid < entries.size(); ++layerid) {
      auto& entry = entries.at(layerid);
      int buffer_len = entry.output->size();
      char* buffer_data = allreduce_buffers.at(layerid).get();
      std::unique_ptr<char[]> recv_buffer(new char[buffer_len]);

      // wait for this layer to finish copying to host
      auto cuda_result = cudaStreamSynchronize(cuda_context_->streams[global_state_->current_nccl_stream][layerid]);
      cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);

      MPI_Comm* node_comm = &reduction_comms_[rank_log_size_-1];
      switch(entry.tensor->dtype()) {
        case DataType::HOROVOD_FLOAT16:
          SyncAllreduce((uint16_t*)buffer_data, (uint16_t*)recv_buffer.get(), *node_comm, reduction_comms_, layerid, entry);
          break;
        case DataType::HOROVOD_FLOAT32:
          SyncAllreduce((float*)buffer_data, (float*)recv_buffer.get(), *node_comm, reduction_comms_, layerid, entry);
          break;
        case DataType::HOROVOD_FLOAT64:
          SyncAllreduce((double*)buffer_data, (double*)recv_buffer.get(), *node_comm, reduction_comms_, layerid, entry);
          break;
        default:
          throw std::logic_error("Unsupported data type");
      }
      // start the copy back to device
      cuda_result = cudaMemcpyAsync(
        (void*) entry.tensor->data(), buffer_data,
        buffer_len, 
        cudaMemcpyHostToDevice,
        cuda_context_->streams[global_state_->current_nccl_stream][layerid]);
      cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
    }

    // wait for all copies to device to finish
    for (size_t layerid = 0; layerid < entries.size(); ++layerid) {
      auto cuda_result = cudaStreamSynchronize(cuda_context_->streams[global_state_->current_nccl_stream][layerid]);
      cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);
    }
  }

  for (size_t layerid = 0; layerid < entries.size(); ++layerid) {
    auto& entry = entries.at(layerid);
    void* buffer_data;
    int buffer_len;

    buffer_data = (void*) entry.tensor->data();

    buffer_len = entry.output->size();

  
    // This will create a stream per layer.
    InitCUDA(entry, layerid);
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

void AdasumCudaRingAllreduceOp::MemcpyUtil(TensorTableEntry entry, void* dest, void* src, size_t buffer_len, int layerid) {
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

bool AdasumCudaRingAllreduceOp::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}

}
}
