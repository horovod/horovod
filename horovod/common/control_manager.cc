//
// Created by Sihan Zeng on 2019-06-07.
//

#include "control_manager.h"

namespace horovod{
namespace common{

extern HorovodGlobalState horovod_global;

extern MPIContext mpi_context;

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

int Controller::get_rank() {return rank;}
void Controller::set_rank(int rank_) {rank = rank_;}
int Controller::get_local_rank() {return local_rank;}
void Controller::set_local_rank(int local_rank_) {local_rank = local_rank_;}
int Controller::get_cross_rank() {return cross_rank;}
void Controller::set_cross_rank(int cross_rank_) {cross_rank = cross_rank_;}

int Controller::get_size() {return size;}
void Controller::set_size(int size_) {size = size_;}
int Controller::get_local_size() {return local_size;}
void Controller::set_local_size(int local_size_) {local_size = local_size_;}
int Controller::get_cross_size() {return cross_size;}
void Controller::set_cross_size(int cross_size_) {cross_size = cross_size_;}

Controller::CommunicatorType Controller::GetCommunicatorType(){
  return type;
}

GlooContext Controller::GetGlooContext(){
  return gloo_ctx_;
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
    global_state_.should_finalize = true;
  }

  if (global_state_.ranks.size() > 0) {
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group work_group;
    MPI_Group_incl(world_group, global_state_.ranks.size(), &(global_state_.ranks[0]),
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
  int rank;
  MPI_Comm_rank(mpi_ctx_.mpi_comm, &rank);
  bool is_coordinator = rank == 0;

  // Get MPI size to determine how many tensors to wait for before reducing.
  int size;
  MPI_Comm_size(mpi_ctx_.mpi_comm, &size);
#if HAVE_MLSL
  mlsl_context.Init(size);
#endif
  if (is_coordinator) {
    LOG(INFO) << "Started Horovod with " << size << " processes";
  }

  // Determine local rank by querying the local communicator.
  MPI_Comm local_comm;
  MPI_Comm_split_type(mpi_ctx_.mpi_comm, MPI_COMM_TYPE_SHARED, 0,
                      MPI_INFO_NULL,
                      &local_comm);
  int local_rank, local_size;
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_size(local_comm, &local_size);
  std::vector<int> local_comm_ranks((size_t) local_size);
  local_comm_ranks[local_rank] = rank;
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                local_comm_ranks.data(), 1,
                MPI_INT, local_comm);

  // Determine if cluster is homogeneous, i.e., if every node has the same
  // local_size
  auto local_sizes = new int[size];
  MPI_Allgather(&local_size, 1, MPI_INT, local_sizes, 1, MPI_INT,
                mpi_ctx_.mpi_comm);

  bool is_homogeneous = true;
  for (int i = 0; i < size; ++i) {
    if (local_sizes[i] != local_size) {
      is_homogeneous = false;
      break;
    }
  }
  for (int i = 0; i < size; i += local_sizes[i]) {
    global_state_.local_sizes.push_back(local_sizes[i]);
  }

  delete[] local_sizes;
  global_state_.is_homogeneous = is_homogeneous;

  // Set up cross-communicator in case of hierarchical allreduce.
  MPI_Comm cross_comm;
  MPI_Comm_split(mpi_ctx_.mpi_comm, local_rank, rank, &cross_comm);
  int cross_rank, cross_size;
  MPI_Comm_rank(cross_comm, &cross_rank);
  MPI_Comm_size(cross_comm, &cross_size);

  // Create custom MPI float16 data type.
  MPI_Datatype mpi_float16_t;
  MPI_Type_contiguous(2, MPI_BYTE, &mpi_float16_t);
  MPI_Type_commit(&mpi_float16_t);

  // Create custom MPI float16 summation op.
  MPI_Op mpi_float16_sum;
  MPI_Op_create(&float16_sum, 1, &mpi_float16_sum);

  // Create custom datatypes for the parameter manager.
  global_state_.param_manager.CreateMpiTypes();

  global_state_.rank = rank;
  global_state_.local_rank = local_rank;
  global_state_.cross_rank = cross_rank;
  global_state_.size = size;
  global_state_.local_size = local_size;
  global_state_.cross_size = cross_size;
  mpi_ctx_.local_comm = local_comm;
  mpi_ctx_.cross_comm = cross_comm;
  mpi_ctx_.mpi_float16_t = mpi_float16_t;
  mpi_ctx_.mpi_float16_sum = mpi_float16_sum;
  global_state_.mpi_threads_supported = (provided == MPI_THREAD_MULTIPLE);
  global_state_.local_comm_ranks = local_comm_ranks;
}

MPIContext Controller::GetMPIContext(){
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
}
void MPIController::AllReduce(const void* sendbuf, void* recvbuf, int count,
              DataType datatype, OpType optype) {
  int ret_code = MPI_Allreduce(sendbuf, recvbuf, count, GetMPIDataType
  (datatype), GetMPIOp(optype), mpi_ctx_.mpi_comm);

  if (ret_code != MPI_SUCCESS){
    throw std::logic_error("MPI_AllReduce failed, see MPI output for details.");
  }
}

void MPIController::Gather(const void* sendbuf, int sendcount, DataType
sendtype, void* recvbuf, int recvcount, DataType recvtype, int root){
  int ret_code = MPI_Gather(sendbuf, sendcount, GetMPIDataType(sendtype),
      recvbuf, recvcount, GetMPIDataType(recvtype), root, mpi_ctx_.mpi_comm);

  if (ret_code != MPI_SUCCESS){
    throw std::logic_error("MPI_Gather failed, see MPI output for details.");
  }
}

void MPIController::AllGather(const void* sendbuf, int sendcount, DataType
sendtype, void* recvbuf, int recvcount, DataType recvtype){
  int ret_code = MPI_Allgather(sendbuf, sendcount, GetMPIDataType(sendtype),
      recvbuf, recvcount, GetMPIDataType(recvtype), mpi_ctx_.mpi_comm);
  if (ret_code != MPI_SUCCESS){
    throw std::logic_error("MPI_AllGather failed, see MPI output for details.");
  }
}

void MPIController::Gatherv(const void* sendbuf, int sendcount, DataType
sendtype, void* recvbuf, const int recvcount[], const int displs[], DataType
            recvtype, int root){

  int ret_code = MPI_Gatherv(sendbuf, sendcount, GetMPIDataType(sendtype),
      recvbuf, recvcount, displs, GetMPIDataType(recvtype), root, mpi_ctx_
      .mpi_comm);
  if (ret_code != MPI_SUCCESS){
    throw std::logic_error("MPI_Gatherv failed, see MPI output for details.");
  }
}
void MPIController::Brodcast(void *buffer, int count, DataType datatype, int
root){
  int ret_code = MPI_Bcast(buffer, count, GetMPIDataType(datatype),root,
      mpi_ctx_.mpi_comm);
  if (ret_code != MPI_SUCCESS){
    throw std::logic_error("MPI_Bcast failed, see MPI output for details.");
  }
}

void MPIController::Barrier() {
  int ret_code = MPI_Barrier(mpi_ctx_.mpi_comm);
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
    case HOROVOD_NULL:
      return MPI_DATATYPE_NULL;
    default:
      throw std::logic_error("Type " + DataType_Name(data_type) +
                             " is not supported in MPI mode.");
  }
}

MPI_Op MPIController::GetMPIOp(OpType op_type){
  switch (op_type){
    case HOROVOD_SUM:
      return MPI_SUM;
    case HOROVOD_BAND:
      return MPI_BAND;
    default:
      throw std::logic_error("Op not supported buy MPI.");
  }
}



}
}