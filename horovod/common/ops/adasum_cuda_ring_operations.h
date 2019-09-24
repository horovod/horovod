//TODO license
#ifndef HOROVOD_ADASUM_CUDA_RING_OPERATIONS_H
#define HOROVOD_ADASUM_CUDA_RING_OPERATIONS_H

#include <deque>
#include <typeinfo>

#include "adasum_cuda_operations.h"

namespace horovod {
namespace common {

class AdasumCudaRingAllreduceOp : public AdasumCudaAllreduceOp {
  public:
  AdasumCudaRingAllreduceOp(MPIContext* mpi_context, CUDAContext* cuda_context,
                HorovodGlobalState* global_state);

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  protected:
  struct MPIContext* mpi_context_;
  struct CUDAContext* cuda_context_;
  std::deque<FusionBufferManager> buffer_managers_;

  void InitCUDA(const TensorTableEntry& entry, int layerid) override;
  void MemcpyUtil(TensorTableEntry entry, void* dest, void* src, size_t buffer_len, int layerid) override;
};
} // namespace common
} // namespace horovod
#endif // HOROVOD_ADASUM_CUDA_RING_OPERATIONS_H
