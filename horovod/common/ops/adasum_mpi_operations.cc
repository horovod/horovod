//TODO license
#include "adasum_mpi_operations.h"
#include <boost/asio/post.hpp>

namespace horovod {
namespace common {
AdasumMPIOp::AdasumMPIOp(MPIContext* mpi_context, HorovodGlobalState* global_state)
    : AdasumOp(global_state), mpi_context_(mpi_context) {}

bool AdasumMPIOp::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

int AdasumMPIOp::GetLocalRankWithComm(MPI_Comm local_comm) {
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    return local_rank;
}

int AdasumMPIOp::GetSizeWithComm(MPI_Comm comm) {
    int size = 0;
    MPI_Comm_size(comm, &size);
    return size;
}

Status AdasumMPIOp::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
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
              throw std::logic_error("AdaSumOp::Execute_helper: Initialize buffer failed.");
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
  while (global_state_->finished_parallel_reductions.load() < num_reductions) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(50));
  }
  return Status::OK();
}

void AdasumMPIOp::PointToPointSend(void* input_data_buffer,
                                   int64_t buffer_length,
                                   DataType horovod_datatype,
                                   int dest_rank,
                                   int tag,
                                   MPI_Comm communicator) {
  int status;           
  int element_size = GetPerElementSize(horovod_datatype);            
  if (!global_state_->msg_chunk_enabled) {
      status = MPI_Send(input_data_buffer,
                        (int)buffer_length,
                        MPI_CHAR,
                        dest_rank,
                        tag,
                        communicator);
  }
  else {
        const int chunk_size = P2P_MESSAGE_CHUNK_SIZE / element_size;
        for (int buf_index = 0; buf_index < buffer_length; buf_index += chunk_size) {
          status = MPI_Send((uint8_t *)input_data_buffer + buf_index,
                            std::min((int)buffer_length - buf_index, chunk_size) * element_size,
                            MPI_CHAR,
                            dest_rank,
                            tag,
                            communicator);
          status &= status;
        }
  }
  if (status != MPI_SUCCESS) {
    throw std::logic_error("MPI_Send failed, see MPI output for details.");
  }
}

void AdasumMPIOp::PointToPointRecv(void* output_data_buffer,
                                   int64_t buffer_length,
                                   DataType horovod_datatype,
                                   int src_rank,
                                   int tag,
                                   MPI_Comm communicator)
{
  int status;
  int element_size = GetPerElementSize(horovod_datatype);
  if (!global_state_->msg_chunk_enabled) {
      status = MPI_Recv(output_data_buffer,
                        (int)buffer_length,
                        MPI_CHAR,
                        src_rank,
                        tag,
                        communicator,
                        MPI_STATUS_IGNORE);
  }
  else {
        const int chunk_size = P2P_MESSAGE_CHUNK_SIZE / element_size;
        for (int buf_index = 0; buf_index < buffer_length; buf_index += chunk_size) {
          status = MPI_Recv((uint8_t *)output_data_buffer + buf_index,
                            std::min((int)buffer_length - buf_index, chunk_size) * element_size,
                            MPI_CHAR,
                            src_rank,
                            tag,
                            communicator,
                            MPI_STATUS_IGNORE);
          status &= status;
        }
  }

  if (status != MPI_SUCCESS) {
    throw std::logic_error("MPI_Recv failed, see MPI output for details.");
  }
}
void AdasumMPIOp::PointToPointSendRecv(void* input_data_buffer,
                                       int64_t input_buffer_length,
                                       DataType input_horovod_datatype,
                                       int dst_rank,
                                       int send_tag,
                                       void* output_data_buffer,
                                       int64_t output_buffer_length,
                                       DataType output_horovod_datatype,
                                       int src_rank,
                                       int recv_tag,
                                       MPI_Comm communicator) {
  int status;       
  int element_size = GetPerElementSize(input_horovod_datatype);
                
  if (!global_state_->msg_chunk_enabled) {
      status = MPI_Sendrecv(input_data_buffer,
                            (int)input_buffer_length,
                            MPI_CHAR,
                            dst_rank,
                            send_tag,
                            output_data_buffer,
                            (int)output_buffer_length,
                            MPI_CHAR, 
                            src_rank,
                            recv_tag,
                            communicator,
                            MPI_STATUS_IGNORE);
  }
  else {
        const int chunk_size = P2P_MESSAGE_CHUNK_SIZE / element_size;
        for (int buf_index = 0; buf_index < input_buffer_length; buf_index += chunk_size) {
          status = MPI_Sendrecv((uint8_t *)input_data_buffer + buf_index,
                            std::min((int)input_buffer_length - buf_index, chunk_size) * element_size,
                            MPI_CHAR,
                            dst_rank,
                            send_tag,
                            (uint8_t *)output_data_buffer + buf_index,
                            std::min((int)output_buffer_length - buf_index, chunk_size) * element_size,
                            MPI_CHAR, 
                            src_rank,
                            recv_tag,
                            communicator,
                            MPI_STATUS_IGNORE);
          status &= status;
        }
  }
  if (status != MPI_SUCCESS) {
    throw std::logic_error("MPI_SendRecv failed, see MPI output for details.");
  }
}

void AdasumMPIOp::P2pAllreduce(void *grad_buffer,
                               void *recv_buffer,
                               int64_t buffer_length,
                               DataType horovod_datatype,
                               MPI_Comm communicator,
                               int message_tag) {
  int true_rank;
  int redn_rank;
  int size;
  int element_size = GetPerElementSize(horovod_datatype);
  MPI_Comm_rank(communicator, &true_rank);
  MPI_Comm_size(communicator, &size);
  static bool opt_permute_roots_on_allreduce = false;
  int root_node_rotation = opt_permute_roots_on_allreduce ? (message_tag % size) : 0;
  redn_rank = (true_rank ^ root_node_rotation);
  int count = buffer_length / element_size;
  assert(!(grad_buffer <= recv_buffer && recv_buffer < grad_buffer + count) || (recv_buffer <= grad_buffer && grad_buffer < recv_buffer + count));
  // Do a tree reduction
  // The reduction ranks used are a permutation of true ranks (permuted based on message_tag)
  // This spreads the load of tree reduction across different true ranks
  // at each level l, node X0[0..0] receives from X1[0...],
  // where [0..0] is l zeros in the bit representation of the rank of a node
  int level;
  for (level = 1; level < size; level *= 2) {
    int neighbor_redn_rank = redn_rank ^ level;
    int neighbor_true_rank = (neighbor_redn_rank ^ root_node_rotation);
    if (redn_rank % level != 0)
      continue; // stay idle at this level
    if (neighbor_redn_rank >= size)
      continue; // no neighbor and so stay idle at this level
    if ((redn_rank & level) == 0) {
    // recv buffer from neighbor
      PointToPointRecv(recv_buffer, buffer_length, horovod_datatype, neighbor_true_rank, message_tag, communicator);
      // do reduction
      switch(horovod_datatype) {
          case DataType::HOROVOD_FLOAT16:
            ElementwiseAdd((uint16_t*)grad_buffer, (uint16_t*)recv_buffer, count);
            break;
          case DataType::HOROVOD_FLOAT32:
            ElementwiseAdd((float*)grad_buffer, (float*)recv_buffer, count);
            break;
          case DataType::HOROVOD_FLOAT64:
            ElementwiseAdd((double*)grad_buffer, (double*)recv_buffer, count);
            break;
          default:
            throw std::logic_error("Unsupported data type.");
      }
    }
    else {
      // send grad_buffer to neighbor
      PointToPointSend(grad_buffer, buffer_length, horovod_datatype, neighbor_true_rank, message_tag, communicator);
    }
  }
  // Do a inverse tree to do a broadcast
  // cannot use MPI Broadcast as there can be concurrent Allreduces happening in parallel
  // the same logic as above.
  // at each level l, node X0[0..0] sends to X1[0...],
  // where [0..0] is l zeros in the bit representation of the rank of a node
  level /= 2; // this make sure that level < size
  for (; level > 0; level /= 2) {
    int neighbor_redn_rank = redn_rank ^ level;
    int neighbor_true_rank = (neighbor_redn_rank ^ root_node_rotation);
    if (redn_rank % level != 0)
      continue; // stay idle at this level
    if (neighbor_redn_rank >= size)
      continue; // no neighbor and so stay idle at this level
    if ((redn_rank & level) == 0) {
      // send grad_buffer to neighbor
      // and dont wait for the send to finish
      PointToPointSend(grad_buffer, buffer_length, horovod_datatype, neighbor_true_rank, message_tag, communicator);
    }
    else {
      // recv grad_buffer from neighbor
      PointToPointRecv(grad_buffer, buffer_length, horovod_datatype, neighbor_true_rank, message_tag, communicator);
    }
  }
}
} // namespace common
} // namespace horovod
