// Copyright 2019 Microsoft. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef HOROVOD_ADASUM_CUDA_RING_OPERATIONS_H
#define HOROVOD_ADASUM_CUDA_RING_OPERATIONS_H

#include "adasum/adasum_mpi.h"
#include "adasum/adasum_cuda_ring.h"
#include "gpu_operations.h"


namespace horovod {
namespace common {

class AdasumCudaRingAllreduceOp : public AdasumMPI, public GPUAllreduce{
  public:
  AdasumCudaRingAllreduceOp(MPIContext* mpi_context,
                        GPUContext* gpu_context,
                        HorovodGlobalState* global_state);

  ~AdasumCudaRingAllreduceOp();
  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  protected:
  Status RingHierarchical(std::vector<TensorTableEntry>& entries,
                          const Response& response);

  void MemcpyUtil(TensorTableEntry entry, void* dest, void* src, size_t buffer_len, int layerid);

  private:
  void* gpu_temp_recv_buffer;
  size_t gpu_buffer_size = 0;
  void* cpu_cross_node_buffer;
  size_t cpu_buffer_size = 0;
};
} // namespace common
} // namespace horovod
#endif // HOROVOD_ADASUM_CUDA_RING_OPERATIONS_H