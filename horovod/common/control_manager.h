//
// Created by Sihan Zeng on 2019-06-07.
//

#ifndef HOROVOD_CONTROL_MANAGER_H
#define HOROVOD_CONTROL_MANAGER_H

#include <iostream>
#include <vector>
#include "global_state.h"
#include "logging.h"
#include "gloo_context.h"
#include "half.h"

#if HAVE_MPI
#define IN_PLACE MPI_IN_PLACE
#else
#define IN_PLACE ((void *) 1)
#endif

namespace horovod{
namespace common{

enum OpType {
  HOROVOD_SUM = 0,
  HOROVOD_BAND = 1,
};

class Controller{
  public:

  enum CommunicatorType {MPI, GLOO};

  Controller(){};
  virtual void Initialize() = 0;
  virtual void Finalize() = 0;

  virtual void AllReduce(const void* sendbuf, void* recvbuf, int count,
      DataType datatype, OpType optype) = 0;
  virtual void Gather(const void* sendbuf, int sendcount, DataType
      sendtype, void* recvbuf, int recvcount, DataType recvtype, int root) = 0;
  virtual void AllGather(const void* sendbuf, int sendcount, DataType
      sendtype, void* recvbuf, int recvcount, DataType recvtype) = 0;
  virtual void Gatherv(const void* sendbuf, int sendcount, DataType
      sendtype, void* recvbuf, const int recvcount[], const int displs[], DataType
      recvtype, int root) = 0;
  virtual void Brodcast(void *buffer, int count, DataType datatype, int root)
  = 0;
  virtual void Barrier() = 0;

  int get_rank();
  void set_rank(int rank);
  int get_local_rank();
  void set_local_rank(int local_rank);
  int get_cross_rank();
  void set_cross_rank(int cross_rank);
  int get_size();
  void set_size(int size);
  int get_local_size();
  void set_local_size(int local_size);
  int get_cross_size();
  void set_cross_size(int cross_size);

  CommunicatorType GetCommunicatorType();

  MPIContext GetMPIContext();
  GlooContext GetGlooContext();

  protected:

  int rank = 0;
  int local_rank = 0;
  int cross_rank = 0;
  int size = 1;
  int local_size = 1;
  int cross_size = 1;
  bool mpi_threads_supported = false;
  bool is_homogeneous = false;
  std::vector<int> ranks;

  // COMM_WORLD ranks of processes running on this node.
  std::vector<int> local_comm_ranks;

  // Numbers of ranks running per node
  std::vector<int> local_sizes;

  CommunicatorType type = MPI;
  MPIContext mpi_ctx_;
  GlooContext gloo_ctx_;

  HorovodGlobalState &global_state_;
};

class MPIController : public Controller{
  public:
  MPIController(){};
  void Initialize();
  void Finalize();

  void AllReduce(const void* sendbuf, void* recvbuf, int count,
                        DataType datatype, OpType optype) ;
  void Gather(const void* sendbuf, int sendcount, DataType
  sendtype, void* recvbuf, int recvcount, DataType recvtype, int root);
  void AllGather(const void* sendbuf, int sendcount, DataType
  sendtype, void* recvbuf, int recvcount, DataType recvtype);
  void Gatherv(const void* sendbuf, int sendcount, DataType
  sendtype, void* recvbuf, const int recvcount[], const int displs[], DataType
                      recvtype, int root);
  void Brodcast(void *buffer, int count, DataType datatype, int root);
  void Barrier();

  MPIDatatype GetMPIDataType(DataType data_type);
  MPI_Op GetMPIOp(OpType op_type);


  protected:

};

}
}

#endif //HOROVOD_CONTROL_MANAGER_H
