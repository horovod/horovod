//TODO license
#ifndef HOROVOD_MPI_P2P_OPERATIONS_H
#define HOROVOD_MPI_P2P_OPERATIONS_H

#include <iostream>

#include "p2p_operations.h"


namespace horovod {
namespace common {

class MPIPointToPointOp : public PointToPointOp {
public:
  MPIPointToPointOp(MPIContext* mpi_context, HorovodGlobalState* global_state);

  virtual ~MPIPointToPointOp() = default;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:

};

} // namespace common
} // namespace horovod

#endif // HOROVOD_MPI_P2P_OPERATIONS_H
