//TODO license
#ifndef HOROVOD_ADASUM_CUDA_OPERATIONS_H
#define HOROVOD_ADASUM_CUDA_OPERATIONS_H

#include <array>
#include "adasum_mpi_operations.h"
#include "adasum_cuda_ring.h"

#if HAVE_NCCL
#include "nccl_operations.h"
#endif

namespace horovod {
namespace common {

using namespace adasumring;

class AdasumCudaAllreduceOp : public AdasumMPIOp {
  public:
#if HAVE_NCCL
  AdasumCudaAllreduceOp(MPIContext* mpi_context,
                        NCCLContext* nccl_context,
                        CUDAContext* cuda_context,
                        HorovodGlobalState* global_state);
#endif
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
  // This map stores variables we will use to do Adasum reduction on GPU with
  // elements in tuple being:
  // 1: anormsq
  // 2: bnormsq
  // 3: dotproduct
  static std::unordered_map<std::thread::id, std::array<double*, 3>> thread_to_device_variable_map;

  virtual void InitCUDAStreams(const std::vector<TensorTableEntry> entries);
  
  void InitDeviceVariables(TensorTableEntry entry) override;

  void FreeDeviceVariables();

#if HAVE_NCCL
  // TODO: All of the following is closely replicated in ncc_operations.h/cc.
  //       Find a fix for this. One option is to add friend declarations and instantiate
  //       the NcclAllreduce class inside here. Another is to restructure the adasum
  //       code to not require inheritance (mostly now required for the Dispatch* calls
  //       that launch Cuda kernels for GPU reductions) and inherit from the existing
  //       classes.

  std::queue<std::pair<std::string, cudaEvent_t>> event_queue_;

  cudaStream_t* stream_;
  void* host_buffer_;

  NCCLContext* nccl_context_;
  ncclComm_t* nccl_comm_;

  void InitCUDAQueue(const std::vector<TensorTableEntry>& entries, const Response& response);

  Status FinalizeCUDAQueue(const std::vector<TensorTableEntry>& entries);

  void InitNCCLComm(const std::vector<TensorTableEntry>& entries,
                    const std::vector<int32_t>& nccl_device_map);

  void MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                 const TensorTableEntry& e, void* buffer_data_at_offset) override;

  void MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                  const void* buffer_data_at_offset, TensorTableEntry& e) override;

  Status NcclHierarchical(std::vector<TensorTableEntry>& entries,
                          const Response& response);
#endif

  Status RingHierarchical(std::vector<TensorTableEntry>& entries,
                          const Response& response);

  void MemcpyUtil(TensorTableEntry entry, void* dest, void* src, size_t buffer_len, int layerid) override;

  void DispatchComputeDotAndNormSqrds(const void* __restrict__ a,
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
    CudaDotProductImpl(n,
                       a,
                       b,
                       thread_to_device_variable_map[thread_id][0],
                       thread_to_device_variable_map[thread_id][1],
                       thread_to_device_variable_map[thread_id][2],
                       anormsq,
                       bnormsq,
                       dotProduct);
  }

  void DispatchScaledAdd(DataType horovod_datatype,
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

  bool CheckPointerLocation(const void* ptr);
};
} // namespace common
} // namespace horovod
#endif // HOROVOD_ADASUM_CUDA_OPERATIONS_H
