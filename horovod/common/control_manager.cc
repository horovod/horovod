//
// Created by Sihan Zeng on 2019-06-07.
//

#include "control_manager.h"
#include "operations.h"
#include "parameter_manager.h"
#include "logging.h"

namespace horovod{
namespace common{

#if HAVE_GLOO
extern GlooContext gloo_context;
#endif

#if HAVE_CUDA
extern CUDAContext cuda_context;
#endif

#if HAVE_NCCL
extern NCCLContext nccl_context;
#endif

#if HAVE_DDL
extern DDLContext ddl_context;
#endif

#if HAVE_MLSL
extern MLSLContext mlsl_context;
#endif

void Controller::SetRank(const int* ranks, int nrank){
  for (auto i = 0; i < nrank; ++i){
    ranks_.push_back(ranks[i]);
  }
}

void Controller::set_cpu_operation(const char* string){
  cpu_operation_ = std::string(string);
}

int Controller::GetRank() {return rank_;}
int Controller::GetLocalRank() {return local_rank_;}
int Controller::GetCrossRank() {return cross_rank_;}
int Controller::GetSize() {return size_;}
int Controller::GetLocalSize() {return local_size_;}
int Controller::get_cross_size() {return cross_size_;}
std::string Controller::get_cpu_operation() {return cpu_operation_;}

Controller::ControllerType Controller::GetControllerType(){
  return type_;
}

int Controller::get_ith_node_local_size(int i){
  return local_sizes_[i];
}

const std::vector<int>& Controller::get_local_comm_ranks(){
  return local_comm_ranks_;
}

bool Controller::isCoordinator() const { return is_coordinator_; }
bool Controller::isHomogeneous() const { return is_homogeneous_; }
bool Controller::isMpiThreadsSupported() const {
  return mpi_threads_supported_;
}

MPIController::MPIController(){
  type_ = MPI;
}
void MPIController::Initialize() {
  auto mpi_threads_disable = std::getenv(HOROVOD_MPI_THREADS_DISABLE);
  int required = MPI_THREAD_MULTIPLE;
  if (mpi_threads_disable != nullptr &&
      std::strtol(mpi_threads_disable, nullptr, 10) > 0) {
    required = MPI_THREAD_SINGLE;
  }
  int provided;
  int is_mpi_initialized = 0;
  MPI_Initialized(&is_mpi_initialized);
  if (is_mpi_initialized) {
    MPI_Query_thread(&provided);
    if (provided < MPI_THREAD_MULTIPLE) {
      LOG(WARNING) << "MPI has already been initialized without "
                      "multi-threading support (MPI_THREAD_MULTIPLE). This will "
                      "likely cause a segmentation fault.";
    }
  } else {
#if HAVE_DDL
    // DDL comes with IBM Spectrum MPI
    // and needs to initialize MPI with the proper license.
    DDLAllreduce::DDLInit(&ddl_context, &cuda_context);
#else
    MPI_Init_thread(nullptr, nullptr, required, &provided);
#endif
    should_finalize_ = true;
  }

  mpi_threads_supported_ = (provided == MPI_THREAD_MULTIPLE);

  if (ranks_.size() > 0) {
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group work_group;
    MPI_Group_incl(world_group, ranks_.size(), &(ranks_[0]),
                   &work_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, work_group, 0, &(mpi_ctx_.mpi_comm));
    if (mpi_ctx_.mpi_comm == MPI_COMM_NULL) {
      LOG(WARNING) << "Unable to create Horovod communicator, using "
                      "MPI_COMM_WORLD instead.";
      mpi_ctx_.mpi_comm = MPI_COMM_WORLD;
    }
    MPI_Group_free(&world_group);
    MPI_Group_free(&work_group);
  } else if (!mpi_ctx_.mpi_comm) {
    // No ranks were given and no communicator provided to horovod_init() so use
    // MPI_COMM_WORLD
    MPI_Comm_dup(MPI_COMM_WORLD, &(mpi_ctx_.mpi_comm));
  }

  // Get MPI rank to determine if we are rank zero.
  MPI_Comm_rank(mpi_ctx_.mpi_comm, &rank_);
  is_coordinator_ = rank_ == 0;

  // Get MPI size to determine how many tensors to wait for before reducing.
  MPI_Comm_size(mpi_ctx_.mpi_comm, &size_);
#if HAVE_MLSL
  mlsl_context.Init(size_);
#endif
  if (is_coordinator_) {
    LOG(INFO) << "Started Horovod with " << size_ << " processes";
  }

  // Determine local rank by querying the local communicator.
  MPI_Comm local_comm;
  MPI_Comm_split_type(mpi_ctx_.mpi_comm, MPI_COMM_TYPE_SHARED, 0,
                      MPI_INFO_NULL,
                      &local_comm);
  MPI_Comm_rank(local_comm, &local_rank_);
  MPI_Comm_size(local_comm, &local_size_);
  local_comm_ranks_ = std::vector<int> ((size_t) local_size_);
  local_comm_ranks_[local_rank_] = rank_;
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                local_comm_ranks_.data(), 1,
                MPI_INT, local_comm);

  // Determine if cluster is homogeneous, i.e., if every node has the same
  // local_size
  local_sizes_ = std::vector<int>(size_);
  MPI_Allgather(&local_size_, 1, MPI_INT, local_sizes_.data(), 1, MPI_INT,
                mpi_ctx_.mpi_comm);

  bool is_homogeneous = true;
  for (int i = 0; i < size_; ++i) {
    if (local_sizes_[i] != local_size_) {
      is_homogeneous = false;
      break;
    }
  }

  is_homogeneous_ = is_homogeneous;

  // Set up cross-communicator in case of hierarchical allreduce.
  MPI_Comm cross_comm;
  MPI_Comm_split(mpi_ctx_.mpi_comm, local_rank_, rank_, &cross_comm);
  MPI_Comm_rank(cross_comm, &cross_rank_);
  MPI_Comm_size(cross_comm, &cross_size_);
  mpi_ctx_.local_comm = local_comm;
  mpi_ctx_.cross_comm = cross_comm;

  // Create custom MPI float16 data type.
  MPI_Datatype mpi_float16_t;
  MPI_Type_contiguous(2, MPI_BYTE, &mpi_float16_t);
  MPI_Type_commit(&mpi_float16_t);
  mpi_ctx_.mpi_float16_t = mpi_float16_t;

  // Create custom MPI param struct data type.
  ParameterManager::CreateMpiTypes(mpi_ctx_.mpi_param_t);

  // Create custom MPI float16 summation op.
  MPI_Op mpi_float16_sum;
  MPI_Op_create(&float16_sum, 1, &mpi_float16_sum);
  mpi_ctx_.mpi_float16_sum = mpi_float16_sum;

}

MPIContext& MPIController::GetMPIContext(){
  return mpi_ctx_;
}

void MPIController::Finalize(){
  if (mpi_ctx_.mpi_comm != MPI_COMM_NULL &&
      mpi_ctx_.mpi_comm != MPI_COMM_WORLD) {
    MPI_Comm_free(&mpi_ctx_.mpi_comm);
  }

  if (mpi_ctx_.local_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&mpi_ctx_.local_comm);
  }

  if (mpi_ctx_.cross_comm != MPI_COMM_NULL) {
    MPI_Comm_free(&mpi_ctx_.cross_comm);
  }

  if (mpi_ctx_.mpi_float16_t != MPI_DATATYPE_NULL) {
    MPI_Type_free(&mpi_ctx_.mpi_float16_t);
  }

  if (mpi_ctx_.mpi_float16_sum != MPI_OP_NULL) {
    MPI_Op_free(&mpi_ctx_.mpi_float16_sum);
  }

  if (mpi_ctx_.mpi_param_t != MPI_DATATYPE_NULL){
    MPI_Type_free(&mpi_ctx_.mpi_param_t);
  }

  if (should_finalize_) {
#if HAVE_DDL
    // ddl_finalize calls MPI_Finalize
    ddl_finalize();
#else
    int is_mpi_finalized = 0;
    MPI_Finalized(&is_mpi_finalized);
    if (!is_mpi_finalized) {
      MPI_Finalize();
    }
#endif
  }
}

int MPIController::GetTypeSize(DataType dtype){
  return mpi_ctx_.GetMPITypeSize(dtype);
}

void MPIController::AllReduce(const void* sendbuf, void* recvbuf, int count,
              DataType datatype, OpType optype, Communicator comm) {
  int ret_code = MPI_Allreduce(sendbuf, recvbuf, count, GetMPIDataType
  (datatype), GetMPIOp(optype, datatype), mpi_ctx_.GetMPICommunicator(comm));

  if (ret_code != MPI_SUCCESS){
    throw std::logic_error("MPI_AllReduce failed, see MPI output for details.");
  }
}

void MPIController::Gather(const void* sendbuf, int sendcount, DataType
sendtype, void* recvbuf, int recvcount, DataType recvtype, int root, Communicator comm){
  int ret_code = MPI_Gather(sendbuf, sendcount, GetMPIDataType(sendtype),
      recvbuf, recvcount, GetMPIDataType(recvtype), root, mpi_ctx_.GetMPICommunicator(comm));

  if (ret_code != MPI_SUCCESS){
    throw std::logic_error("MPI_Gather failed, see MPI output for details.");
  }
}

void MPIController::AllGather(const void* sendbuf, int sendcount, DataType
sendtype, void* recvbuf, int recvcount, DataType recvtype, Communicator comm){
  int ret_code = MPI_Allgather(sendbuf, sendcount, GetMPIDataType(sendtype),
      recvbuf, recvcount, GetMPIDataType(recvtype), mpi_ctx_.GetMPICommunicator(comm));
  if (ret_code != MPI_SUCCESS){
    throw std::logic_error("MPI_AllGather failed, see MPI output for details.");
  }
}

void MPIController::Gatherv(const void* sendbuf, int sendcount, DataType
sendtype, void* recvbuf, const int recvcount[], const int displs[], DataType
            recvtype, int root, Communicator comm){

  int ret_code = MPI_Gatherv(sendbuf, sendcount, GetMPIDataType(sendtype),
      recvbuf, recvcount, displs, GetMPIDataType(recvtype), root, mpi_ctx_
      .GetMPICommunicator(comm));
  if (ret_code != MPI_SUCCESS){
    throw std::logic_error("MPI_Gatherv failed, see MPI output for details.");
  }
}
void MPIController::Brodcast(void *buffer, int count, DataType datatype, int
root, Communicator comm){
  int ret_code = MPI_Bcast(buffer, count, GetMPIDataType(datatype),root,
      mpi_ctx_.GetMPICommunicator(comm));
  if (ret_code != MPI_SUCCESS){
    throw std::logic_error("MPI_Bcast failed, see MPI output for details.");
  }
}

void MPIController::Barrier(Communicator comm) {
  int ret_code = MPI_Barrier(mpi_ctx_.GetMPICommunicator(comm));
  if (ret_code != MPI_SUCCESS){
    throw std::logic_error("MPI_Barrier failed, see MPI output for details.");
  }
}

MPI_Datatype MPIController::GetMPIDataType(DataType data_type){
  switch (data_type) {
    case HOROVOD_UINT8:
      return MPI_UINT8_T;
    case HOROVOD_INT8:
      return MPI_INT8_T;
    case HOROVOD_UINT16:
      return MPI_UINT16_T;
    case HOROVOD_INT16:
      return MPI_INT32_T;
    case HOROVOD_INT32:
      return MPI_INT32_T;
    case HOROVOD_INT64:
      return MPI_INT64_T;
    case HOROVOD_FLOAT16:
      return mpi_ctx_.mpi_float16_t;
    case HOROVOD_FLOAT32:
      return MPI_FLOAT;
    case HOROVOD_FLOAT64:
      return MPI_DOUBLE;
    case HOROVOD_BOOL:
      return MPI_CXX_BOOL;
    case HOROVOD_LONG_LONG_INT:
      return MPI_LONG_LONG_INT;
    case HOROVOD_NULL:
      return MPI_DATATYPE_NULL;
    case HOROVOD_PARAM:
      return mpi_ctx_.mpi_param_t;
    case HOROVOD_BYTE:
      return MPI_BYTE;
    default:
      LOG(INFO) << data_type;
      throw std::logic_error("Type not supported in MPI mode.");
  }
}

MPI_Op MPIController::GetMPIOp(OpType op_type, DataType data_type){
  switch (op_type){
    case HOROVOD_SUM:
      return data_type == HOROVOD_FLOAT16 ? mpi_ctx_.mpi_float16_sum : MPI_SUM;
    case HOROVOD_BAND:
      return MPI_BAND;
    case HOROVOD_BOR:
      return MPI_BOR;
    default:
      throw std::logic_error("Op not supported buy MPI.");
  }
}

void MPIController::SetMPIComm(MPI_Comm comm){
  MPI_Comm_dup(comm, &mpi_ctx_.mpi_comm);
}

}
}