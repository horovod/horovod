//
// Created by Sihan Zeng on 2019-06-07.
//

#ifndef HOROVOD_CONTROL_MANAGER_H
#define HOROVOD_CONTROL_MANAGER_H

#include <iostream>
#include <vector>
#include "logging.h"
#include "gloo_context.h"
#include "half.h"
#include "mpi_context.h"
#include "operations.h"

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
  HOROVOD_BOR= 2
};

class Controller{
  public:

  enum ControllerType {MPI, GLOO};

  virtual void Initialize() = 0;
  virtual void Finalize() = 0;

  virtual int GetTypeSize(DataType dtype) = 0;

  virtual void AllReduce(const void* sendbuf, void* recvbuf, int count,
      DataType datatype, OpType optype, Communicator comm) = 0;
  virtual void Gather(const void* sendbuf, int sendcount, DataType
      sendtype, void* recvbuf, int recvcount, DataType recvtype, int root, Communicator comm) = 0;
  virtual void AllGather(const void* sendbuf, int sendcount, DataType
      sendtype, void* recvbuf, int recvcount, DataType recvtype, Communicator comm) = 0;
  virtual void Gatherv(const void* sendbuf, int sendcount, DataType
      sendtype, void* recvbuf, const int recvcount[], const int displs[], DataType
      recvtype, int root, Communicator comm) = 0;
  virtual void Brodcast(void *buffer, int count, DataType datatype, int root, Communicator comm)
  = 0;
  virtual void Barrier(Communicator comm) = 0;

  // communicator related funcitons
  void SetRank(const int* ranks, int nrank);
  int GetRank();
  int GetLocalRank();
  int GetCrossRank();
  int GetSize();
  int GetLocalSize();
  int get_cross_size();
  int get_ith_node_local_size(int i);
  const std::vector<int>& get_local_comm_ranks();
  bool isCoordinator() const;
  bool isHomogeneous() const;
  bool isMpiThreadsSupported() const;

  std::string get_cpu_operation();
  void set_cpu_operation(const char* string);

  ControllerType GetCommunicatorType();


protected:

  bool should_finalize_ = false;
  int rank_ = 0;
  int local_rank_ = 0;
  int cross_rank_ = 0;
  int size_ = 1;
  int local_size_ = 1;
  int cross_size_ = 1;
  bool is_coordinator_ = false;
  bool is_homogeneous_ = false;
  bool mpi_threads_supported_;

  // ranks of the mpi world
  std::vector<int> ranks_;

  // COMM_WORLD ranks of processes running on this node.
  std::vector<int> local_comm_ranks_;

  // Numbers of ranks running per node
  std::vector<int> local_sizes_;

  std::string cpu_operation_;

  ControllerType type_ = MPI;

};

class MPIController : public Controller{
  public:
  void Initialize() override;
  void Finalize() override;

  int GetTypeSize(DataType dtype) override;

  void AllReduce(const void* sendbuf, void* recvbuf, int count,
                        DataType datatype, OpType optype, Communicator comm) override ;
  void Gather(const void* sendbuf, int sendcount, DataType
  sendtype, void* recvbuf, int recvcount, DataType recvtype, int root, Communicator comm) override;
  void AllGather(const void* sendbuf, int sendcount, DataType
  sendtype, void* recvbuf, int recvcount, DataType recvtype, Communicator comm) override;
  void Gatherv(const void* sendbuf, int sendcount, DataType
  sendtype, void* recvbuf, const int recvcount[], const int displs[], DataType
                      recvtype, int root, Communicator comm) override;
  void Brodcast(void *buffer, int count, DataType datatype, int root, Communicator comm) override;
  void Barrier(Communicator comm) override;

  MPI_Datatype GetMPIDataType(DataType data_type);
  MPI_Op GetMPIOp(OpType op_type, DataType data_type);

  MPIContext& GetMPIContext();
  void SetMPIComm(MPI_Comm comm);
protected:
  MPIContext mpi_ctx_;
};

}
}

#endif //HOROVOD_CONTROL_MANAGER_H
