//TODO license
#include "mpi_p2p_operations.h"

namespace horovod {
namespace common {
AdasumMPIP2pOp::AdasumMPIP2pOp(MPIContext* mpi_context)
    : AdasumOp(), mpi_context_(mpi_context) {
  {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    int nearest_power_2 = 1;
    int log_size;
    for (nearest_power_2 = 1, log_size = 0; (nearest_power_2 << 1) <= size; nearest_power_2 = (nearest_power_2 << 1), log_size++)
    {
    }
    int shift_val;
    int level;
    world_rank_log_size_ = log_size;
    world_reduction_comms_ = new MPI_Comm[log_size];
    int *node_rank = new int[size];
    for (level = 1, shift_val = 1; level < nearest_power_2; level = (level << 1), shift_val++)
    {
        int base_rank = ((rank >> shift_val) << shift_val);
        for (int i = 0; i < (level << 1); i++)
        {
            node_rank[i] = (base_rank + i);
        }
        MPI_Group red_group;
        MPI_Group_incl(world_group, (level << 1), node_rank, &red_group);
        MPI_Comm_create_group(MPI_COMM_WORLD, red_group, 0, &world_reduction_comms_[shift_val - 1]);
        MPI_Group_free(&red_group);
    }
    delete[] node_rank;
  }
}

AdasumMPIP2pOp::~AdasumMPIP2pOp() {
  if(world_reduction_comms_ != nullptr) {
    delete world_reduction_comms_;
  }
}

int AdasumMPIP2pOp::GetLocalRankWithComm(MPI_Comm local_comm) {
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    return local_rank;
}

int AdasumMPIP2pOp::GetSizeWithComm(MPI_Comm comm) {
    int size = 0;
    MPI_Comm_size(comm, &size);
    return size;
}

void AdasumMPIP2pOp::SumAllreduceWithComm(void* data, int num_elements, DataType horovod_datatype, MPI_Comm comm) {
  int status;
  status = MPI_Allreduce(MPI_IN_PLACE,
                         data,
                         num_elements,
                         mpi_context_->GetMPIDataType(horovod_datatype),
                         MPI_SUM,
                         comm);
  if (status != MPI_SUCCESS) {
    throw std::logic_error("MPI_Allreduce failed, see MPI output for details.");
  }
}

void AdasumMPIP2pOp::PointToPointSend(void* input_data_buffer,
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

void AdasumMPIP2pOp::PointToPointRecv(void* output_data_buffer,
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
void AdasumMPIP2pOp::PointToPointSendRecv(void* input_data_buffer,
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
} // namespace common
} // namespace horovod