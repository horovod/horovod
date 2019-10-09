//TODO license
#ifndef HOROVOD_ADASUM_CUDA_OPERATIONS_H
#define HOROVOD_ADASUM_CUDA_OPERATIONS_H

#include <array>
#include "mpi_p2p_operations.h"
#include "nccl_operations.h"

namespace horovod {
namespace common {

class AdasumCudaAllreduceOp : public AdasumMPIP2pOp, public NCCLAllreduce {
  public:
  AdasumCudaAllreduceOp(MPIContext* mpi_context,
                        NCCLContext* nccl_context,
                        CUDAContext* cuda_context,
                        HorovodGlobalState* global_state);

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  protected:

  void PopulateNCCLCommStrategy(int& nccl_rank, int& nccl_size,
                                Communicator& nccl_id_bcast_comm) override;

  Status NcclHierarchical(std::vector<TensorTableEntry>& entries,
                          const Response& response);
};
} // namespace common
} // namespace horovod
#endif // HOROVOD_ADASUM_CUDA_OPERATIONS_H