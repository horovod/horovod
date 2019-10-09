//TODO license
#ifndef HOROVOD_ADASUM_MPI_OPERATIONS_H
#define HOROVOD_ADASUM_MPI_OPERATIONS_H

#include <iostream>
#include "mpi.h"

#include "mpi_p2p_operations.h"
#include "collective_operations.h"

namespace horovod {
namespace common {

class AdasumMPIAllreduceOp : public AdasumMPIP2pOp, public AllreduceOp {
public:
  AdasumMPIAllreduceOp(MPIContext* mpi_context, HorovodGlobalState* global_state);
    
  Status Execute(std::vector<TensorTableEntry>& entries,
                         const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
                       const std::vector<TensorTableEntry>& entries,
                       const Response& response) const override;
protected:

  Status FusedVHDD(std::vector<TensorTableEntry>& entries,
                          const Response& response);
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_ADASUM_MPI_OPERATIONS_H