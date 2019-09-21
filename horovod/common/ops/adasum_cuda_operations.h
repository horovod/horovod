//TODO license
#ifndef HOROVOD_ADASUM_CUDA_OPERATIONS_H
#define HOROVOD_ADASUM_CUDA_OPERATIONS_H

#include <array>
#include "adasum_operations.h"
#include "cuda_operations.h"
#include "cuda_fp16.h"

namespace horovod {
namespace common {

class AdasumCudaAllreduceOp : public AdasumOp {
  public:
  AdasumCudaAllreduceOp(MPIContext* mpi_context, CUDAContext* cuda_context,
                HorovodGlobalState* global_state);
  ~AdasumCudaAllreduceOp();
  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  protected:
  struct CUDAContext* cuda_context_;

  // This map stores variables we will use to do AdaSum reduction on GPU with
  // elements in tuple being:
  // 1: anormsq
  // 2: bnormsq
  // 3: dotproduct
  static std::unordered_map<std::thread::id, std::array<double*, 3>> thread_to_device_variable_map;

  void InitCUDA(const TensorTableEntry& entry, int layerid);

  void FinalizeCUDA();

  void MemcpyUtil(TensorTableEntry entry, void* dest, void* src, size_t buffer_len, int layerid) override;

  template<typename T>
  void static DotProductImpl(const T* __restrict__  a,
                             const T* __restrict__ b, 
                             int n, 
                             double& dotProduct, 
                             double& anormsq, 
                             double& bnormsq, 
                             HorovodGlobalState *global_state,
                             int layerid);
  
  template<typename T>
  void static ScaleAddImpl(int n, double acoeff, T* __restrict__ a, double bcoeff, T* __restrict__ b, HorovodGlobalState *global_state, int layerid);
};
} // namespace common
} // namespace horovod
#endif // HOROVOD_ADASUM_CUDA_OPERATIONS_H
