// TODO license
#ifndef HOROVOD_ADASUM_CUDA_RING_H
#define HOROVOD_ADASUM_CUDA_RING_H

#include "mpi.h"

#include "../common.h"
#include "../global_state.h"
#include "../mpi/mpi_context.h"
#include "cuda_operations.h"
#include "cuda_fp16.h"
#include "adasum_cuda_kernels.h"

namespace horovod {
namespace common {
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

} // common
} // horovod
#endif // HOROVOD_ADASUM_CUDA_RING_H