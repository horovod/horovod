//TODO license
#ifndef HOROVOD_ADASUM_CUDA_OPERATIONS_H
#define HOROVOD_ADASUM_CUDA_OPERATIONS_H

#include <array>
#include <nccl.h>
#include "adasum_mpi_operations.h"
#include "cuda_operations.h"
#include "cuda_fp16.h"
#include "adasum_cuda_kernels.h"

namespace horovod {
namespace common {

class AdasumCudaAllreduceOp : public AdasumMPIOp {
  public:
  AdasumCudaAllreduceOp(MPIContext* mpi_context,
                        CUDAContext* cuda_context,
                        HorovodGlobalState* global_state);
  ~AdasumCudaAllreduceOp();
  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  protected:
  struct CUDAContext* cuda_context_;
  NCCLContext* nccl_context_;
  ncclComm_t* nccl_comm_;

  // This map stores variables we will use to do AdaSum reduction on GPU with
  // elements in tuple being:
  // 1: anormsq
  // 2: bnormsq
  // 3: dotproduct
  static std::unordered_map<std::thread::id, std::array<double*, 3>> thread_to_device_variable_map;

  virtual void InitCUDA(const TensorTableEntry& entry, int layerid);

  void FinalizeCUDA();
  
  void InitNCCLComm(const std::vector<TensorTableEntry>& entries,
                    const std::vector<int32_t>& nccl_device_map);

  void AdasumInternal(void* gradient_buffer,
                      void* recv_buffer,
                      MPI_Comm* node_comm,
                      MPI_Comm* reduction_comm_pool,
                      MPI_Comm local_comm,
                      int layerid,
                      TensorTableEntry entry) override;

  void NcclHierarchical(std::vector<TensorTableEntry>& entries);

  void MemcpyUtil(TensorTableEntry entry, void* dest, void* src, size_t buffer_len, int layerid) override;

  void ComputeDotAndNormSqrdsWrapper(const void* __restrict__ a,
                                     const void* __restrict__ b,
                                     DataType horovod_datatype,
                                     int count,
                                     double& dotProduct,
                                     double& anormsq,
                                     double& bnormsq,
                                     HorovodGlobalState *global_state,
                                     int layerid) override;

  template<typename T>
  void static DotProductImpl(const T* __restrict__  a,
                             const T* __restrict__ b, 
                             int n, 
                             double& dotProduct, 
                             double& anormsq, 
                             double& bnormsq, 
                             HorovodGlobalState *global_state,
                             int layerid) {
    auto thread_id = std::this_thread::get_id();
    CudaDotProductImpl(n, a, b, thread_to_device_variable_map[thread_id][0], thread_to_device_variable_map[thread_id][1], thread_to_device_variable_map[thread_id][2], anormsq, bnormsq, dotProduct);

  }

  void ScaledAddWrapper(DataType horovod_datatype,
                        int count,
                        double acoeff,
                        void* __restrict__ a,
                        double bcoeff,
                        void* __restrict__ b,
                        HorovodGlobalState *global_state,
                        int layerid) override;

  template<typename T>
  void static ScaleAddImpl(int n, double acoeff, T* __restrict__ a, double bcoeff, T* __restrict__ b, HorovodGlobalState *global_state, int layerid) {
    CudaScaleAddImpl(n, a, b, acoeff, bcoeff);
  }
};
} // namespace common
} // namespace horovod
#endif // HOROVOD_ADASUM_CUDA_OPERATIONS_H
