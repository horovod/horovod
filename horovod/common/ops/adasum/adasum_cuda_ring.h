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

#ifndef HOROVOD_ADASUM_CUDA_RING_H
#define HOROVOD_ADASUM_CUDA_RING_H

#include "mpi.h"

#include "../../common.h"
#include "../../global_state.h"
#include "../../mpi/mpi_context.h"
#include "../gpu_operations.h"
#include "../cuda/adasum_cuda_kernels.h"

namespace horovod {
namespace common {

struct Ring {
  int loop[8];
  int nextGPU;
  int prevGPU;
  int load;
  bool isFat;
  cudaStream_t adasum_stream;

  void InitRing(int tmp[], bool _isFat, int rank, int size, cudaStream_t _adasum_stream);
  int GetAfterLoad(int message_len);
  void AddLoad(int message_len);
  void ReduceLoad(int message_len);
};

struct Message {
  MPIContext* mpi_context;
  MPI_Comm comm;
  MPI_Request req;
  Ring* ring;
  int rank;
  int ring_starter_rank;
  int leg; // number of legs in the ring has been done
  void* grad_buf;
  void* recv_buf;
  DataType datatype;
  int tag;
  int count;
  bool cuda_stream_sync; // used to determine whether previous leg ended with mpi op or cuda async op

  Message(MPIContext* mpi_context);
  void InitMessage(Ring* _ring, int rank, int _ring_starter_rank, int _count, void* _grad_buf, void* _recv_buf, DataType _datatype, MPI_Comm _comm, int _tag);
  virtual bool Test() = 0;
protected:
  virtual void Start() = 0;
};

struct ReduceMessage : public Message {
  ReduceMessage(MPIContext* mpi_context);
  virtual bool Test();
protected:
  virtual void Start();
};

struct BroadcastMessage : public Message {
  BroadcastMessage(MPIContext* mpi_context);
  virtual bool Test();
protected:
  virtual void Start();
};

struct AllRings {
  int num_rings = 4;
  Ring* rings;
  std::vector<Message*> messages;
  cudaStream_t adasum_stream;

  ~AllRings();
  AllRings(int rank, int size);
  Ring* PickRing(int count);
  void InitMessageInRing(Message* message, void* grad_buf, void* recv_buf, int size, DataType datatype, MPI_Comm comm, int grad_tag, int rank);
  void WaitAllMessages();

  // private:
  static std::vector<std::vector<int>> GetTopologyMatrix(int size);
  static std::vector<int> GetRingFromTopologyMatrix(
      int size, const std::vector<std::vector<int>>& topology_matrix,
      int performance_rank);
};

template<typename T>
void AdasumCudaReduce(int count, T* device_a, T* device_b, cudaStream_t stream){

  static double* device_vals;
  static bool device_vals_init = false;
  if (!device_vals_init) {
    cudaMalloc(&device_vals, 3*sizeof(double));
    device_vals_init = true;
  }

  CudaSingleAdasumImpl(count, device_a, device_b, device_vals, stream);
}

} // common
} // horovod
#endif // HOROVOD_ADASUM_CUDA_RING_H