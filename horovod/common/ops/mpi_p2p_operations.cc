//TODO license
#include "p2p_operations.h"

namespace horovod {
namespace common {

MPIPointToPointOp::MPIPointToPointOp(MPIContext* mpi_context, HorovodGlobalState* global_state)
    : PointToPointOp(mpi_context, global_state) {}

bool MPIPointToPointOp::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

} // namespace common
} // namespace horovod