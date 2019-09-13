// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Microsoft Corp.
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
#ifndef HOROVOD_MSALLREDUCE_CUDA_RING_OPERATIONS_H
#define HOROVOD_MSALLREDUCE_CUDA_RING_OPERATIONS_H

#include <deque>
#include <typeinfo>

#include "msallreduce_operations.h"
#include "cuda_operations.h"
#include "cuda_fp16.h"

namespace horovod {
namespace common {

class MsCudaRingAllreduceOp : public MsAllreduceOp {
  public:
  MsCudaRingAllreduceOp(MPIContext* mpi_context, CUDAContext* cuda_context,
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

  void InitCUDA(const TensorTableEntry& entry, int layerid);
  void memcpyUtil(TensorTableEntry entry, void* dest, void* src, size_t buffer_len, int layerid) override;
};

namespace msallreduce {

struct Ring {
  int loop[8];
  int nextGPU;
  int prevGPU;
  int load;
  bool isFat;

  void InitRing(int tmp[], bool _isFat, int rank, int size);
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

  Message(MPIContext* mpi_context);
  void InitMessage(Ring* _ring, int rank, int _ring_starter_rank, int _count, void* _grad_buf, void* _recv_buf, DataType _datatype, MPI_Comm _comm, int _tag);
  virtual bool Test() = 0;
protected:
  virtual void Start() = 0;
};

struct AllreduceMessage : public Message {
  AllreduceMessage(MPIContext* mpi_context);
  virtual bool Test();
protected:
  virtual void Start();
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

  ~AllRings();
  AllRings(int rank, int size);
  Ring* PickRing(int count);
  void InitMessageInRing(Message* message, void* grad_buf, void* recv_buf, int size, DataType datatype, MPI_Comm comm, int grad_tag, int rank);
  void WaitAllMessages();
};

} // namespace msallreduce
} // namespace common
} // namespace horovod
#endif // HOROVOD_MSALLREDUCE_CUDA_RING_OPERATIONS_H
