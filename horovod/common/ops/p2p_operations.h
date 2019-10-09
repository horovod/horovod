//TODO license
#ifndef HOROVOD_P2P_OPERATIONS_H
#define HOROVOD_P2P_OPERATIONS_H

#include "../common.h"
#include "../mpi/mpi_context.h"


namespace horovod {
namespace common {

// Abstract class to expose basic point to point communication interface
template<typename Communicator_type>
class PointToPointOp {
public:
  PointToPointOp() {};
  virtual ~PointToPointOp() = default;

protected:
  virtual void PointToPointSend(void* input_data_buffer,
                                int64_t buffer_length,
                                DataType horovod_datatype,
                                int dest_rank,
                                int tag,
                                Communicator_type communicator) = 0;

  virtual void PointToPointRecv(void* output_data_buffer,
                                int64_t buffer_length,
                                DataType horovod_datatype,
                                int src_rank,
                                int tag,
                                Communicator_type communicator) = 0;

  virtual void PointToPointSendRecv(void* input_data_buffer,
                                    int64_t input_buffer_length,
                                    DataType input_horovod_datatype,
                                    int dst_rank,
                                    int send_tag,
                                    void* output_data_buffer,
                                    int64_t output_buffer_length,
                                    DataType output_horovod_datatype,
                                    int src_rank,
                                    int recv_tag,
                                    Communicator_type communicator) = 0;
};

} // namespace common
} // namespace horovod
#endif // HOROVOD_P2P_OPERATIONS_H
