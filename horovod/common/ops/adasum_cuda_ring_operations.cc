// Copyright 2019 Microsoft. All Rights Reserved.
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

#include "adasum_cuda_ring_operations.h"

namespace horovod {
namespace common {

AdasumCudaRingAllreduceOp::AdasumCudaRingAllreduceOp(MPIContext* mpi_context, GPUContext* gpu_context, HorovodGlobalState* global_state)
    : AdasumMPI(mpi_context, global_state), GPUAllreduce(gpu_context, global_state){
}

AdasumCudaRingAllreduceOp::~AdasumCudaRingAllreduceOp() {
}

Status AdasumCudaRingAllreduceOp::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  if(entries.empty()) {
    return Status::OK();
  }
  gpu_op_context_.InitGPU(entries);
  if (!reduction_comms_initialized) {
    InitializeVHDDReductionComms();
  }
  return RingHierarchical(entries, response);
}

Status AdasumCudaRingAllreduceOp::RingHierarchical(std::vector<TensorTableEntry>& entries,
                        const Response& response) {

  int num_reductions = entries.size();
  auto& first_entry = entries[0];

  // If needed, allocate a temporary GPU buffer that's needed for Adasum computation.
  // Currently, the max combined size of entries that can be passed in at once is equal to the tensor fusion buffer size.
  size_t new_gpu_size = (global_state_->parameter_manager.TensorFusionThresholdBytes() > first_entry.output->size()) ? 
                        global_state_->parameter_manager.TensorFusionThresholdBytes() : first_entry.output->size();
  if (new_gpu_size > gpu_buffer_size) {
    if (gpu_buffer_size > 0) {
      cudaFree(gpu_temp_recv_buffer);
    }
    cudaMalloc(&gpu_temp_recv_buffer, new_gpu_size);
    gpu_buffer_size = new_gpu_size;
  }
  int64_t offset = 0;
  size_t total_buffer_size = 0;

  AllRings all_rings(global_state_->controller->GetLocalRank(), global_state_->controller->GetLocalSize());

  int local_rank = 0;
  int local_size = global_state_->controller->GetLocalSize();

  bool cross_node = global_state_->controller->GetSize() > global_state_->controller->GetLocalSize();

  MPI_Comm_rank(mpi_context_->local_comm, &local_rank);

  size_t layerid = 0;
  //enqueue messages
  for (; layerid < num_reductions; ++layerid) {
    auto& entry = entries.at(layerid);
    void* buffer_data;
    int buffer_len;
    void* recv_buffer;
    buffer_data = (void*) entry.tensor->data();
    buffer_len = entry.output->size();
    if (layerid % 8 == local_rank) {
      total_buffer_size += buffer_len;
    }
    if (entry.tensor->data() == entry.output->data()) {
      // Get the temp buffer to be used for the Op
      recv_buffer = (uint8_t*)gpu_temp_recv_buffer + offset;
      offset += buffer_len;
    }
    else {
      recv_buffer = (void*) entry.output->data();
    }
    gpu_context_->ErrorCheck("cudaSetDevice", cudaSetDevice(entry.device));

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

  if (cross_node) {
      if (global_state_->parameter_manager.TensorFusionThresholdBytes() > total_buffer_size) {
        total_buffer_size = global_state_->parameter_manager.TensorFusionThresholdBytes();
      }
      if (total_buffer_size > cpu_buffer_size) {
        if (cpu_buffer_size > 0) {
          free(cpu_cross_node_buffer);
        }
        cpu_cross_node_buffer = malloc(total_buffer_size);
        cpu_buffer_size = total_buffer_size;
      }
      offset = 0;
      std::vector<int> tensor_counts(num_reductions);
      // start device to host copies
      for (layerid = 0; layerid < num_reductions; ++layerid) {
        if (layerid % 8 == local_rank) {
          auto& entry = entries.at(layerid);
          tensor_counts[layerid] = entry.tensor->shape().num_elements();
          int buffer_len = entry.output->size();
          void* buffer_data = (uint8_t*)cpu_cross_node_buffer + offset;
          offset += buffer_len;
          
          auto cuda_result = cudaMemcpyAsync(
            buffer_data, (void*) entry.tensor->data(),
            buffer_len, 
            cudaMemcpyDeviceToHost,
            gpu_context_->streams[global_state_->current_nccl_stream][layerid]);
          gpu_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
        } else {
          tensor_counts[layerid] = 0;
        }
      }
      auto recv_buffer = GetRecvBuffer(total_buffer_size);
      // wait for this layer to finish copying to host
      DispatchFusedAllreduce(entries, cpu_cross_node_buffer, recv_buffer, tensor_counts, local_size, 
                              mpi_context_->GetMPICommunicator(Communicator::GLOBAL), 0, reduction_comms_,
                              first_entry.tensor->dtype(), global_state_);
      offset = 0;
      for (layerid = 0; layerid < num_reductions; ++layerid) {
        if (layerid % 8 == local_rank) {
          auto& entry = entries.at(layerid);
          int buffer_len = entry.output->size();
          void* buffer_data = (uint8_t*)cpu_cross_node_buffer + offset;
          offset += buffer_len;
          // start the copy back to device
          auto cuda_result = cudaMemcpyAsync(
            (void*) entry.tensor->data(), buffer_data,
            buffer_len, 
            cudaMemcpyHostToDevice,
            gpu_context_->streams[global_state_->current_nccl_stream][layerid]);
          gpu_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
        }
      }
  }

  //ring broadcast
  for (layerid = 0; layerid < entries.size(); ++layerid) {
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

  for (layerid = 0; layerid < entries.size(); ++layerid) {
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
                                    gpu_context_->streams[global_state_->current_nccl_stream][entry.device]);
    gpu_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
    auto cuda_sync_result = cudaStreamSynchronize(gpu_context_->streams[global_state_->current_nccl_stream][entry.device]);
    gpu_context_->ErrorCheck("cudaStreamSynchronize", cuda_sync_result);
}

bool AdasumCudaRingAllreduceOp::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return (entries[0].device != CPU_DEVICE_ID) && (global_state_->controller->GetLocalSize() == 8);

}
}
}