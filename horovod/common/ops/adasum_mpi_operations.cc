//TODO license
#include "adasum_mpi_operations.h"
#include <boost/asio/post.hpp>

namespace horovod {
namespace common {
AdasumMPIOp::AdasumMPIOp(MPIContext* mpi_context, HorovodGlobalState* global_state)
    : AdasumOp(global_state), mpi_context_(mpi_context) {
  int local_rank, local_size;
  MPI_Comm_size(mpi_context_->local_comm, &local_size);
  MPI_Comm_rank(mpi_context_->local_comm, &local_rank);
  if (local_rank == 0)
  {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // converting to node-based rank and size
    rank /= local_size;
    size /= local_size;

    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    int nearest_power_2 = 1;
    int log_size;
    for (nearest_power_2 = 1, log_size = 0; (nearest_power_2 << 1) <= size; nearest_power_2 = (nearest_power_2 << 1), log_size++)
    {
    }
    int shift_val;
    int level;
    rank_log_size_ = log_size;
    reduction_comms_ = new MPI_Comm[log_size];
    int *node_rank = new int[size];
    for (level = 1, shift_val = 1; level < nearest_power_2; level = (level << 1), shift_val++)
    {
        int base_rank = ((rank >> shift_val) << shift_val);
        for (int i = 0; i < (level << 1); i++)
        {
            // converting back to world rank
            node_rank[i] = (base_rank + i) * local_size;
        }
        MPI_Group red_group;
        MPI_Group_incl(world_group, (level << 1), node_rank, &red_group);
        MPI_Comm_create_group(MPI_COMM_WORLD, red_group, 0, &reduction_comms_[shift_val - 1]);
        MPI_Group_free(&red_group);
    }
    delete[] node_rank;
  }
}

AdasumMPIOp::~AdasumMPIOp() {
  if(reduction_comms_ != nullptr) {
    LOG(INFO,global_state_->controller->GetRank())<<"Preparing to delete reduction comms.";
    delete reduction_comms_;
  }
}

bool AdasumMPIOp::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return global_state_->adasum_algorithm == AdasumAlgorithm::CPU_TREE;
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
  if(entries.empty()) {
      return Status::OK();
  }
  if (global_state_->adasum_algorithm == AdasumAlgorithm::CPU_TREE) {
    return TreeHierarchical(entries, response);
  }
  else {
      throw std::logic_error("Unsupported Adasum MPI op. To use Adasum_GPU_* algorithms, please re-build Horovod with HOROVOD_GPU_ALLREDUCE=NCCL");
  }
}

void AdasumMPIOp::TreeHierarchicalInternal(TensorTableEntry& entry, int layerid, const Response& response) {
  void* buffer_data;
  int buffer_len;
  void* recv_buffer;

  InitDeviceVariables(entry);

  buffer_data = (void*) entry.tensor->data();

  buffer_len = entry.output->size();

  FusionBufferManager buffer_manager;

  if(entry.tensor->data() == entry.output->data()) {

    // Get the temp buffer to be used for the Op
    {
      std::lock_guard<std::mutex> guard(buffer_lock_);
      assert(!temp_buffers_.empty());
      buffer_manager = temp_buffers_.front();
      temp_buffers_.pop_front();
    }

    // TODO: Maybe add before and after callbacks to timeline?
    Status status = buffer_manager.InitializeBuffer(
        buffer_len,
        entry.device, entry.context,
        global_state_->current_nccl_stream,
        [](){},
        [](){},
        [](int64_t& size, int64_t& threshold){return size >= threshold;});

    if (!status.ok()) {
        throw std::logic_error("AdasumCudaAllreduceOp::Execute: Initialize buffer failed.");
        return;
    }
    auto buffer = buffer_manager.GetBuffer(entry.device, entry.context->framework(), global_state_->current_nccl_stream);
    recv_buffer = const_cast<void*>(buffer->AccessData(entry.context));
  }
  else {
      recv_buffer = (void*) entry.output->data();
  }
  
  MPI_Comm local_comm = mpi_context_->local_comm;
  MPI_Comm* node_comm = NULL;
  if (rank_log_size_ != 0) {
      node_comm = &reduction_comms_[rank_log_size_-1];
  }
  
  int local_rank = 0;

  local_rank = GetLocalRankWithComm(local_comm);
  SyncLocalReduce(buffer_data, recv_buffer, local_comm, layerid, entry);
  // TODO have a version of VHDD that performs asynchronously
  if (local_rank == 0 && node_comm != NULL) {
    DispatchSyncAllreduce(buffer_data, recv_buffer, node_comm, reduction_comms_, layerid, entry);
  }
  SyncLocalBroadcast(buffer_data, local_comm, entry, layerid);

  if(entry.tensor->data() == entry.output->data()) {
    // Return the buffer back into the pool of available buffers
    std::lock_guard<std::mutex> guard(buffer_lock_);
    temp_buffers_.emplace_back(buffer_manager);
  }
  else {
    MemcpyUtil(entry, (void *) entry.output->data(), (void *) entry.tensor->data(), (size_t) entry.tensor->size(), layerid);
  }
}

Status AdasumMPIOp::TreeHierarchical(std::vector<TensorTableEntry>& entries, const Response& response) {
  int layerid = 0;
  int num_reductions = entries.size();
  finished_parallel_reductions_ = 0;
  bool use_main_thread = global_state_->adasum_num_threads == 1;
  for (auto& entry : entries) {
    // skip threadpool if we only have 1 thread in there
    if (use_main_thread) {
        TreeHierarchicalInternal(entry, layerid, response);
    }
    else {
        boost::asio::post(*global_state_->adasum_background_thread_pool,
        [this, &entry, response, layerid, &entries] {
          TreeHierarchicalInternal(entry, layerid, response);
          finished_parallel_reductions_++;
        });
    }
    layerid++;
  }

  if(!use_main_thread) {
    while (finished_parallel_reductions_ < num_reductions) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(50));
    }
  }
  return Status::OK();
}

void AdasumMPIOp::DispatchComputeDotAndNormSqrds(const void* __restrict__  a, const void* __restrict__ b, DataType horovod_datatype, int count, double& dotProduct, double& anormsq, double& bnormsq, HorovodGlobalState *global_state, int layerid) {
  if (horovod_datatype == DataType::HOROVOD_FLOAT16) {
    ComputeDotAndNormSqrdsfp16((uint16_t*)a, (uint16_t*)b, count, dotProduct, anormsq, bnormsq, global_state, layerid);
  }
  else if (horovod_datatype == DataType::HOROVOD_FLOAT32) {
    ComputeDotAndNormSqrds((float*)a, (float*)b, count, dotProduct, anormsq, bnormsq, global_state, layerid);
  }
  else if (horovod_datatype == DataType::HOROVOD_FLOAT64) {
    ComputeDotAndNormSqrds((double*)a, (double*)b, count, dotProduct, anormsq, bnormsq, global_state, layerid);      
  }
  else {
      throw std::logic_error("Unsupported data type.");
  }
}

void AdasumMPIOp::DispatchScaledAdd(DataType horovod_datatype, int count, double acoeff, void* __restrict__ a, double bcoeff, void* __restrict__ b, HorovodGlobalState *global_state, int layerid) {
  if (horovod_datatype == DataType::HOROVOD_FLOAT16) {
    ScaledAddfp16(count, acoeff, (uint16_t*)a, bcoeff, (uint16_t*)b, global_state, layerid);
  }
  else if (horovod_datatype == DataType::HOROVOD_FLOAT32) {
    ScaledAdd(count, acoeff, (float*)a, bcoeff, (float*)b, global_state, layerid);
  }
  else if (horovod_datatype == DataType::HOROVOD_FLOAT64) {
    ScaledAdd(count, acoeff, (double*)a, bcoeff, (double*)b, global_state, layerid);
  }
  else {
      throw std::logic_error("Unsupported data type.");
  }
}

void AdasumMPIOp::InitDeviceVariables(TensorTableEntry entry) {
    // nothing to do here since we only operate on host memory
}

void AdasumMPIOp::PointToPointSend(void* input_data_buffer,
                                   int64_t buffer_length,
                                   DataType horovod_datatype,
                                   int dest_rank,
                                   int tag,
                                   MPI_Comm communicator) {
  int status;           
  int element_size = GetPerElementSize(horovod_datatype);
  int count = buffer_length / element_size;       
  status = MPI_Send(input_data_buffer,
                    count,
                    mpi_context_->GetMPIDataType(horovod_datatype),
                    dest_rank,
                    tag,
                    communicator);
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
  int count = buffer_length / element_size;
  status = MPI_Recv(output_data_buffer,
                    count,
                    mpi_context_->GetMPIDataType(horovod_datatype),
                    src_rank,
                    tag,
                    communicator,
                    MPI_STATUS_IGNORE);

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
  int input_element_size = GetPerElementSize(input_horovod_datatype);
  int output_element_size = GetPerElementSize(output_horovod_datatype);
  int input_count = input_buffer_length / input_element_size;
  int output_count = output_buffer_length / output_element_size;
 
  status = MPI_Sendrecv(input_data_buffer,
                        input_count,
                        mpi_context_->GetMPIDataType(input_horovod_datatype),
                        dst_rank,
                        send_tag,
                        output_data_buffer,
                        output_count,
                        mpi_context_->GetMPIDataType(output_horovod_datatype), 
                        src_rank,
                        recv_tag,
                        communicator,
                        MPI_STATUS_IGNORE);
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
      // TODO the switch here will likely introduce overhead since this is our hot path
      // replace it with something else.
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
