//TODO license
#ifndef HOROVOD_P2P_OPERATIONS_H
#define HOROVOD_P2P_OPERATIONS_H

#include <iostream>

#include "mpi.h"

#include "../common.h"
#include "../global_state.h"
#include "../mpi/mpi_context.h"
#include "collective_operations.h"


namespace horovod {
namespace common {

class PointToPointOp : public AllreduceOp {
public:
  PointToPointOp(MPIContext* mpi_context, HorovodGlobalState* global_state);
  bool Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const override;
  virtual ~PointToPointOp() = default;

protected:
  MPIContext* mpi_context_;
  template<class T>
  void PointToPointSend(void* input_data_buffer,
                        int64_t buffer_length,
                        int dest_rank,
                        int tag,
                        Communicator communicator) {
    int status;                       
    if (!global_state_->msg_chunk_enabled) {
        status = MPI_Send(input_data_buffer,
                          (int)buffer_length,
                          MPI_CHAR,
                          dest_rank,
                          tag,
                          mpi_context_->GetMPICommunicator(communicator));

    }
    else {
          const int chunk_size = P2P_MESSAGE_CHUNK_SIZE / sizeof(T);
          for (int buf_index = 0; buf_index < buffer_length; buf_index += chunk_size) {
            status = MPI_Send((uint8_t *)input_data_buffer + buf_index,
                              std::min((int)buffer_length - buf_index, chunk_size) * sizeof(T),
                              MPI_CHAR,
                              dest_rank,
                              tag,
                              mpi_context_->GetMPICommunicator(communicator));
            status &= status;
          }
    }

    if (status != MPI_SUCCESS) {
      throw std::logic_error("MPI_Send failed, see MPI output for details.");
    }
  }

  template<class T>
  void PointToPointRecv(T* output_data_buffer,
                        int64_t buffer_length,
                        int src_rank,
                        int tag,
                        Communicator communicator) {
    int status;
    if (!global_state_->msg_chunk_enabled) {
        status = MPI_Recv(output_data_buffer,
                          (int)buffer_length,
                          MPI_CHAR,
                          src_rank,
                          tag,
                          mpi_context_->GetMPICommunicator(communicator),
                          MPI_STATUS_IGNORE);
    }
    else {
          const int chunk_size = P2P_MESSAGE_CHUNK_SIZE / sizeof(T);
          for (int buf_index = 0; buf_index < buffer_length; buf_index += chunk_size) {
            status = MPI_Recv((uint8_t *)output_data_buffer + buf_index,
                              std::min((int)buffer_length - buf_index, chunk_size) * sizeof(T),
                              MPI_CHAR,
                              src_rank,
                              tag,
                              mpi_context_->GetMPICommunicator(communicator),
                              MPI_STATUS_IGNORE);
            status &= status;
          }
    }

    if (status != MPI_SUCCESS) {
      throw std::logic_error("MPI_Recv failed, see MPI output for details.");
    }
  }

};

} // namespace common
} // namespace horovod

#endif // HOROVOD_P2P_OPERATIONS_H
