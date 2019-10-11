//TODO license
#ifndef HOROVOD_MPI_P2P_OPERATIONS_H
#define HOROVOD_MPI_P2P_OPERATIONS_H

#include "mpi.h"

#include "../mpi/mpi_context.h"
#include "adasum_operations.h"


namespace horovod {
namespace common {

class AdasumMPIP2pOp : public AdasumOp<MPI_Comm> {
public:
  AdasumMPIP2pOp(MPIContext* mpi_context);
  
  ~AdasumMPIP2pOp();
  
protected:
  void PointToPointSend(void* input_data_buffer,
                        int64_t buffer_length,
                        DataType horovod_datatype,
                        int dest_rank,
                        int tag,
                        MPI_Comm communicator) override;

  void PointToPointRecv(void* output_data_buffer,
                        int64_t buffer_length,
                        DataType horovod_datatype,
                        int src_rank,
                        int tag,
                        MPI_Comm communicator) override;

  void PointToPointSendRecv(void* input_data_buffer,
                            int64_t input_buffer_length,
                            DataType input_horovod_datatype,
                            int dst_rank,
                            int send_tag,
                            void* output_data_buffer,
                            int64_t output_buffer_length,
                            DataType output_horovod_datatype,
                            int src_rank,
                            int recv_tag,
                            MPI_Comm communicator) override;

  int GetLocalRankWithComm(MPI_Comm local_comm) override;

  int GetSizeWithComm(MPI_Comm comm) override;

  void SumAllreduceWithComm(void* data, int num_elements, DataType horovod_datatype, MPI_Comm comm) override;

  MPIContext* mpi_context_;
  // MPI communicators used to do adasum
  MPI_Comm* reduction_comms_ = nullptr;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_MPI_P2P_OPERATIONS_H
