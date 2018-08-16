// DGC host function implementations
// by Yuechao Pan
// for NVIDIA

// past compile with following command:
// nvcc -std=c++11 -c -o dgc.cu.o horovod_nvidia/horovod/common/dgc.cu.cc      \
   -x cu -Xcompiler -fPIC -dlink --expt-extended-lambda                        \
   -gencode=arch=compute_70,code=\"sm_70,compute_70\"

//#pragma once

#include <string>
#include <algorithm>
#include <chrono>
#include <thread>
#include <locale>
#include <curand_kernel.h>
//#include <thrust/sort.h>
//#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <mpi.h>
#include "dgc.h"
#include "dgc_kernel.cu.cc"

namespace horovod {
namespace dgc {

// Pertector for CUDA calls
#define GUARD_CU2(op_name, op)                                                 \
{                                                                              \
  retval = (op);                                                               \
  if (retval != cudaSuccess) {                                                 \
    std::string error_message = std::string(__FILE__) + std::string(":")       \
      + std::to_string(__LINE__) + std::string("(")                            \
      + std::string(op_name) + std::string(") failed: ")                       \
      + cudaGetErrorString(retval);                                            \
    fprintf(stderr, "%s\n", error_message.c_str());                            \
    fflush(stderr);                                                            \
    return retval;                                                             \
  }                                                                            \
}

// Pertector for CUDA calls
#define GUARD_CU(op)                                                           \
{                                                                              \
  retval = (op);                                                               \
  if (retval != cudaSuccess) {                                                 \
    std::string error_message = std::string(__FILE__) + std::string(":")       \
      + std::to_string(__LINE__) + std::string(" failed: ")                    \
      + cudaGetErrorString(retval);                                            \
    fprintf(stderr, "%s\n", error_message.c_str());                            \
    fflush(stderr);                                                            \
    return retval;                                                             \
  }                                                                            \
}

// Pertector for NCCL calls
#define GUARD_NCCL2(op_name, op)                                               \
{                                                                              \
  auto nccl_result = (op);                                                     \
  if (nccl_result != ncclSuccess) {                                            \
    std::string error_message = std::string(__FILE__) + std::string(":")       \
      + std::to_string(__LINE__) + std::string("(")                            \
      + std::string(op_name) + std::string(") failed: ")                       \
      + ncclGetErrorString(nccl_result);                                       \
    fprintf(stderr, "%s\n", error_message.c_str());                            \
    fflush(stderr);                                                            \
    return cudaErrorUnknown;                                                   \
  }                                                                            \
}

// Pertector for MPI calls
#define GUARD_MPI2(op_name, op)                                                \
{                                                                              \
  auto mpi_result = (op);                                                      \
  if (mpi_result != MPI_SUCCESS) {                                             \
    char  error_string[MPI_MAX_ERROR_STRING + 1];                              \
    error_string[MPI_MAX_ERROR_STRING] = 0;                                    \
    int   error_length = 0;                                                    \
    MPI_Error_string(mpi_result, error_string, &error_length);                 \
    std::string error_message = std::string(__FILE__) + std::string(":")       \
      + std::to_string(__LINE__) + std::string("(")                            \
      + std::string(op_name) + std::string(") failed: ")                       \
      + std::string(error_string);                                             \
    fprintf(stderr, "%s\n", error_message.c_str());                            \
    fflush(stderr);                                                            \
    return cudaErrorUnknown;                                                   \
  }                                                                            \
}

// ****************************
// Memory management
// ****************************

enum Malloc_t
{
  Default,
  Host,
  Managed,
  Raw,
};

// Unified free function
template <typename T>
cudaError_t Free(
  T* &ptr,
  Malloc_t malloc_type = Malloc_t::Default)
{
  cudaError_t retval = cudaSuccess;
  if (ptr == NULL)
    return retval;

  //printf("Freeing @ %p\n", ptr);
  if (malloc_type == Host) {
    GUARD_CU2("cudaFreeHost",
      cudaFreeHost(ptr));
  } else if (malloc_type == Default || malloc_type == Managed) {
    GUARD_CU2("cudaFree",
      cudaFree(ptr));
  } else if (malloc_type == Raw)
    free(ptr);

  //printf("Freed @ %p\n", ptr);
  ptr = NULL;
  return retval;
}

// Unified malloc function
template <typename T>
cudaError_t Malloc(
  T* &ptr,
  size_t target,
  Malloc_t malloc_type = Malloc_t::Default,
  unsigned int flags = cudaMemAttachGlobal,
  cudaStream_t stream = 0)
{
  cudaError_t retval = cudaSuccess;

  size_t size = target * sizeof(T);
  //printf("Allocating %ld x %ld bytes on %s\n", target, sizeof(T),
  //   malloc_type == Default ? "Default" :
  //  (malloc_type == Host    ? "Host" :
  //  (malloc_type == Managed ? "Managed" : "Raw")));

  if (malloc_type == Default) {
    GUARD_CU2("cudaMalloc",
      cudaMalloc(&ptr, size));
  } else if (malloc_type == Host) {
    GUARD_CU2("cudaMallocHost",
      cudaMallocHost(&ptr, size));
  } else if (malloc_type == Managed) {
    GUARD_CU2("cudaMallocManaged",
      cudaMallocManaged(&ptr, size, flags));
  } else if (malloc_type == Raw)
    ptr = (T*)malloc(size);

  //printf("Allocated %ld x %ld bytes @ %p\n", target, sizeof(T), ptr);
  return retval;
}

// Unified memcpy function
template <typename T, typename SizeT>
cudaError_t Memcpy(
  T* dest,
  T* src,
  SizeT num_elements,
  Malloc_t malloc_type = Malloc_t::Default,
  cudaStream_t stream = 0)
{
  cudaError_t retval = cudaSuccess;
  if (num_elements == 0)
    return retval;
  if (dest == NULL || src == NULL)
    return retval;

  if (malloc_type != Raw) {
    if (stream == 0) {
      retval = cudaMemcpyAsync(dest, src, sizeof(T) * num_elements,
        cudaMemcpyDefault, stream);
    } else {
      retval = cudaMemcpy(dest, src, sizeof(T) * num_elements,
        cudaMemcpyDefault);
    }
  } else {
    memcpy(dest, src, sizeof(T) * num_elements);
  }
  return retval;
}

// Unified memset function
template <typename T, typename SizeT>
cudaError_t Memset(
  T* ptr,
  int value,
  SizeT num_elements,
  Malloc_t malloc_type = Malloc_t::Default,
  cudaStream_t stream = 0)
{
  cudaError_t retval = cudaSuccess;
  if (num_elements == 0 || ptr == NULL)
    return retval;

  if (malloc_type != Malloc_t::Raw) {
    if (stream == 0) {
      retval = cudaMemset(ptr, value, num_elements * sizeof(T));
    } else {
      retval = cudaMemsetAsync(ptr, value, num_elements * sizeof(T), stream);
    }
  } else {
    memset(ptr, value, num_elements * sizeof(T));
  }

  return retval;
}

// Garentee sufficient allocation
template <typename T, typename SizeT>
cudaError_t GarenteeAllocation(
  T*      &ptr,
  SizeT   &allocated,
  size_t   target,
  Malloc_t malloc_type = Malloc_t::Default,
  unsigned int flags = cudaMemAttachGlobal,
  cudaStream_t stream = 0,
  bool     keep_content = false,
  bool     init_to_zero = false)
{
  cudaError_t retval = cudaSuccess;
  if (allocated >= target)
    return retval;

  if (!keep_content) {
    auto temp_ptr = ptr;
    GUARD_CU(Free<T> (temp_ptr, malloc_type));
    GUARD_CU(Malloc(ptr, target, malloc_type, flags));
    if (init_to_zero) {
      GUARD_CU(Memset(ptr, 0, target, malloc_type, stream));
    }
  }

  else {
    T* temp_ptr = NULL;
    GUARD_CU(Malloc(temp_ptr, target, malloc_type, flags));
    GUARD_CU(Memcpy(temp_ptr, ptr, allocated, malloc_type, stream));
    if (init_to_zero) {
      GUARD_CU(Memset(temp_ptr + allocated, 0, target - allocated,
        malloc_type, stream));
    }
    GUARD_CU(Free(ptr, malloc_type));
    ptr = temp_ptr;
    temp_ptr = NULL;
  }

  allocated = target;
  return retval;
}

template <typename T>
std::string ToString(const T& val)
{
  return std::to_string(val);
}

template <>
std::string ToString(const common::Framework& val)
{
  std::string str = "Unknown";
  if (val == common::Framework::TENSORFLOW)
    str = "TensorFlow";
  else if (val == common::Framework::PYTORCH)
    str = "PyTorch";
  return str;
}

template <typename T>
cudaError_t FreePersistent(
  std::string  name,
  T*          &ptr,
  DgcConfig   &config,
  DgcState    &state)
{
  cudaError_t retval = cudaSuccess;
  printf("Freeing %s @ %p\n", name.c_str(), ptr);

  //auto tuple = std::make_tuple(
  //  config.device, config.context -> framework(), name);
  std::string key = std::to_string(config.device) + "::"
    + ToString(config.context -> framework()) + "::" + name;
  if (state.memory_table[key].second == 0)
    return retval;
  //state.memory_table[key].first
  //  = std::make_shared<common::PersistentBuffer>(config.context, 0);
  auto buffer = state.memory_table[key].first;
  ptr = (T*)(buffer -> AccessData(config.context));
  auto status = config.context -> AllocatePersistent(0, &buffer);
  if (!status.ok()) {
    GUARD_CU2("Allocating 0 byte", cudaErrorUnknown);
  }
  state.memory_table[key].first = buffer;
  state.memory_table[key].second = 0;

  printf("Freed %s @ %p\n", name.c_str(), ptr);
  ptr = NULL;
  return retval;
}

template <typename T>
cudaError_t AccessPersistent(
  std::string  name,
  T*          &ptr,
  DgcConfig   &config,
  DgcState    &state)
{
  cudaError_t retval = cudaSuccess;
  //auto tuple = std::make_tuple(
  //  config.device, config.context -> framework(), name);
  std::string key = std::to_string(config.device) + "::"
    + ToString(config.context -> framework()) + "::" + name;
  auto& buffer = state.memory_table[key].first;

  ptr = (T*)(buffer -> AccessData(config.context));
  printf("Accessing %s @ %p\n", name.c_str(), ptr);
  return retval;
}

template <typename T, typename SizeT>
cudaError_t MallocPersistent(
  std::string  name,
  T*          &ptr,
  SizeT        num_elements,
  DgcConfig   &config,
  DgcState    &state)
{
  cudaError_t retval = cudaSuccess;
  std::string str = std::to_string(sizeof(T)) + " * "
    + std::to_string(num_elements) + " bytes for " + name;
  printf("Allocating %s\n", str.c_str());

  //auto tuple = std::make_tuple(
  //  config.device, config.context -> framework(), name);
  std::string key = std::to_string(config.device) + "::"
    + ToString(config.context -> framework()) + "::" + name;
  auto buffer = state.memory_table[key].first;
  size_t allocated = state.memory_table[key].second;
  size_t request_bytes = sizeof(T) * num_elements;

  if (allocated != 0) {
    ptr = (T*)(buffer -> AccessData(config.context));
    printf("Warnning: %s has been allocated %ld bytes on GPU %d, ptr = %p. "
      "Reallocating %d * %ld = %ld bytes.",
      name.c_str(), (long)allocated, config.device, ptr,
      sizeof(T), (long)num_elements, (long)request_bytes);

    GUARD_CU(FreePersistent(name, ptr, config, state));
  }

  auto status = config.context -> AllocatePersistent(request_bytes, &buffer);
  if (!status.ok()) {
    GUARD_CU2("Allocating " + str, cudaErrorUnknown);
  }

  state.memory_table[key] = std::make_pair(buffer, request_bytes);
  ptr = (T*)(buffer -> AccessData(config.context));

  printf("Allocated %s, ptr = %p\n", str.c_str(), ptr);
  return retval;
}

template <typename T, typename SizeT>
cudaError_t GarenteeAllocationPersistent(
  std::string   name,
  T*           &ptr,
  SizeT         request_num_elements,
  DgcConfig    &config,
  DgcState     &state,
  cudaStream_t  stream = 0,
  bool          keep_content = false,
  bool          init_to_zero = false)
{
  cudaError_t retval = cudaSuccess;

  //auto tuple = std::make_tuple(
  //  config.device, config.context -> framework(), name);
  std::string key = std::to_string(config.device) + "::"
    + ToString(config.context -> framework()) + "::" + name;
  size_t allocated = state.memory_table[key].second;
  size_t request_bytes = sizeof(T) * request_num_elements;
  if (allocated >= request_bytes) {
    ptr = (T*)(state.memory_table[key].first -> AccessData(config.context));
    return retval;
  }

  if (!keep_content || allocated == 0) {
    GUARD_CU(FreePersistent  (name, ptr, config, state));
    GUARD_CU(MallocPersistent(name, ptr, request_num_elements, config, state));
    if (init_to_zero) {
      GUARD_CU(Memset(ptr, 0, request_num_elements, Malloc_t::Default, stream));
    }
  }

  else {
    auto old_buffer = state.memory_table[key].first;
    T* old_ptr = (T*)(old_buffer -> AccessData(config.context));
    SizeT allocated_num_elements = allocated / sizeof(T);
    GUARD_CU(FreePersistent  (name, ptr, config, state));
    GUARD_CU(MallocPersistent(name, ptr, request_num_elements, config, state));

    GUARD_CU(Memcpy(ptr, old_ptr, allocated_num_elements,
      Malloc_t::Default, stream));
    if (init_to_zero) {
      GUARD_CU(Memset(ptr + allocated_num_elements, 0,
        request_num_elements - allocated_num_elements,
        Malloc_t::Default, stream));
    }
    old_ptr = NULL;
    //old_buffer = std::make_shared<common::PersistentBuffer>(config.context, 0);
    auto status = config.context -> AllocatePersistent(0, &old_buffer);
    if (!status.ok()) {
      GUARD_CU2("Allocating 0 byte", cudaErrorUnknown);
    }
  }
  return retval;
}

// ****************************
// DGC Functions
// ****************************

void str2bool(std::string str, bool &val)
{
  if (str == "True")
    val = true;
  if (str == "False")
    val = false;
}

// Setting config parameters
void DgcConfig::Set(std::string key, std::string value)
{
  printf("Setting %s to %s\n", key.c_str(), value.c_str());

  if (key == "dgc_sparsity_warmup_epochs")
    warmup_epochs = std::stoi(value);

  else if (key == "dgc_init_sparsity")
    init_sparsity = std::stod(value);

  else if (key == "dgc_final_sparsity")
    final_sparsity = std::stod(value);

  else if (key == "dgc_sampling_rate")
    sampling_rate = std::stod(value);

  else if (key == "dgc_rand_seed")
    rand_seed = std::stoi(value);

  else if (key == "dgc_grid_size")
    grid_size = std::stoi(value);

  else if (key == "dgc_block_size")
    block_size = std::stoi(value);

  else if (key == "dgc_min_sampling_num")
    min_sampling_num = std::stoi(value);

  else if (key == "dgc_local_gradient_clipping")
    str2bool(value, local_gradient_clipping);

  else if (key == "dgc_clipping_threshold")
    clipping_threshold = std::stof(value);

  else if (key == "dgc_use_allreduce")
    str2bool(value, use_allReduce);

  else if (key == "dgc_use_hierarchical_allreduce")
    str2bool(value, use_hierarchical_allreduce);

  else if (key == "dgc_overlap_mask_allreduce")
    str2bool(value, overlap_mask_allreduce);

  else if (key == "dgc_learning_rate_decay_factor")
    learning_rate_decay_factor = std::stof(value);

  else if (key == "dgc_num_epochs_per_decay")
    num_epochs_per_decay = std::stof(value);

  else if (key == "dgc_min_learning_rate_factor")
    min_learning_rate_factor = std::stof(value);

  else if (key == "dgc_flush_steps")
    flush_steps = std::stoi(value);

  else if (key == "dgc_use_momentum_correction")
    str2bool(value, use_momentum_correction);

  else if (key == "dgc_use_gradient_accumulation")
    str2bool(value, use_gradient_accumulation);

  else if (key == "dgc_smooth_sparsity")
    str2bool(value, smooth_sparsity);

  else if (key == "momentum")
    momentum = std::stof(value);

  else if (key == "num_examples_per_epoch")
    num_examples_per_epoch = std::stoi(value);

  else if (key == "batch_size")
    batch_size_per_gpu = std::stoi(value);

}

// Get configuration from environmental variables
void DgcConfig::ReadFromENV()
{
  const std::string env_list[] = {
    "dgc_sparsity_warmup_epochs",
    "dgc_init_sparsity",
    "dgc_final_sparsity",
    "dgc_sampling_rate",
    "dgc_rand_seed",
    "dgc_grid_size",
    "dgc_block_size",
    "dgc_min_sampling_num",
    "dgc_local_gradient_clipping",
    "dgc_clipping_threshold",
    "dgc_use_allreduce",
    "dgc_use_hierarchical_allreduce",
    "dgc_overlap_mask_allreduce",
    "dgc_learning_rate_decay_factor",
    "dgc_num_epochs_per_decay",
    "dgc_min_learning_rate_factor",
    "dgc_flush_steps",
    "dgc_use_momentum_correction",
    "dgc_use_gradient_accumulation",
    "dgc_smooth_sparsity",
    "momentum",
    "num_examples_per_epoch",
    "batch_size"};
  const int num_parameters = 23;
  auto& f = std::use_facet<std::ctype<char>>(std::locale());

  for (int i = 0; i < num_parameters; i++) {
    std::string env_name = env_list[i];
    std::string env_name_upper = env_name;
    f.toupper(&env_name_upper[0], &env_name_upper[0] + env_name_upper.size());
    char* value = std::getenv(env_name_upper.c_str());
    std::string value_str = "";

    if (value == NULL) {
      value = std::getenv(("NO" + env_name_upper).c_str());
      if (value != NULL) {
        value_str = std::string(value);
        if (value_str == "True")
          value_str = "False";
        else if (value_str == "False")
          value_str = "True";
      }
    } else
      value_str = std::string(value);

    if (value != NULL) {
      Set(env_name, value_str);
    }
  }
}

// Unified sort function
template <typename T, typename SizeT, typename Compare>
cudaError_t Sort(
  T           *elements,
  SizeT        num_elements,
  Compare      compare,
  cudaStream_t stream = 0,
  Malloc_t     malloc_type = Malloc_t::Default,
  char       **temp_storage = NULL,
  size_t      *temp_storage_bytes = NULL,
  unsigned int flags = cudaMemAttachGlobal)
{
  cudaError_t retval = cudaSuccess;

  if (malloc_type == Raw) {
    std::sort(elements, elements + num_elements, compare);
    return retval;
  }

  // Not using thrust for now;
  // if sort becomes performance bottleneck, change to cub
  // Note: thrust::sort hit a bug that produced illegal memory access
  //thrust::sort(thrust::cuda::par.on(stream),
  //  elements, elements + num_elements, compare);

  // Cub sorting
  bool temp_storage_allocated = false;
  if (temp_storage == NULL && temp_storage_bytes == NULL) {
    temp_storage = new char*;
    temp_storage[0] = NULL;
    temp_storage_bytes = new size_t;
    temp_storage_bytes[0] = 0;
    temp_storage_allocated = true;
  }

  size_t required_bytes = 0;
  GUARD_CU(cub::DeviceRadixSort::SortKeys(
    (char*)NULL, required_bytes,
    elements, elements,
    num_elements, 0, sizeof(T) * 8, stream));

  GUARD_CU(GarenteeAllocation(temp_storage[0],
    temp_storage_bytes[0], required_bytes, malloc_type, flags));

  GUARD_CU(cub::DeviceRadixSort::SortKeys(
    temp_storage[0], temp_storage_bytes[0],
    elements, elements,
    num_elements, 0, sizeof(T) * 8, stream));

  if (temp_storage_allocated) {
    GUARD_CU(Free(temp_storage[0], malloc_type));
    free(temp_storage);
    free(temp_storage_bytes);
    temp_storage = NULL;
    temp_storage_bytes = NULL;
    temp_storage_allocated = false;
  }

  return retval;
}

// Sort with default less than comparator
template <typename T, typename SizeT>
cudaError_t Sort(
  T      *elements,
  SizeT   num_elements,
  cudaStream_t stream = 0,
  Malloc_t malloc_type = Malloc_t::Default,
  char       **temp_storage = NULL,
  size_t      *temp_storage_bytes = NULL,
  unsigned int flags = cudaMemAttachGlobal)
{
  return Sort(elements, num_elements,
    [] __host__ __device__ (T a, T b){return a < b;},
    stream, malloc_type, temp_storage, temp_storage_bytes, flags);
}

// Sort template switch
template <typename SizeT>
cudaError_t Sort(
  ncclDataType_t nccl_type,
  void        *elements,
  SizeT        num_elements,
  cudaStream_t stream = 0,
  Malloc_t     malloc_type = Malloc_t::Default,
  char       **temp_storage = NULL,
  size_t      *temp_storage_bytes = NULL,
  unsigned int flags = cudaMemAttachGlobal)
{
  cudaError_t retval = cudaSuccess;

  switch (nccl_type) {
  case ncclFloat32:
    retval = Sort<float> ((float*)elements, num_elements,
      stream, malloc_type, temp_storage, temp_storage_bytes, flags);
    break;

  case ncclFloat64:
    retval = Sort<double> ((double*)elements, num_elements,
      stream, malloc_type, temp_storage, temp_storage_bytes, flags);
    break;

  case ncclInt32:
    retval = Sort<int32_t> ((int32_t*)elements, num_elements,
      stream, malloc_type, temp_storage, temp_storage_bytes, flags);
    break;

  case ncclInt64:
    retval = Sort<int64_t> ((int64_t*)elements, num_elements,
      stream, malloc_type, temp_storage, temp_storage_bytes, flags);
    break;

  default:
    break;
  }
  return retval;
}

// Segmeted Sort
template <typename T, typename SizeT, typename Compare>
cudaError_t SegSort(
  T           *elements,
  SizeT        num_elements,
  SizeT       *seg_starts,
  int          num_segments,
  Compare      compare,
  cudaStream_t stream = 0,
  Malloc_t     malloc_type = Malloc_t::Default,
  char       **temp_storage = NULL,
  size_t      *temp_storage_bytes = NULL,
  unsigned int flags = cudaMemAttachGlobal)
{
  cudaError_t retval = cudaSuccess;

  if (malloc_type == Raw) {
    for (int i = 0; i < num_segments; i++)
      std::sort(elements + seg_starts[i], elements + seg_starts[i+1], compare);
    return retval;
  }

  bool temp_storage_allocated = false;
  if (temp_storage == NULL && temp_storage_bytes == NULL) {
    temp_storage = new char*;
    temp_storage[0] = NULL;
    temp_storage_bytes = new size_t;
    temp_storage_bytes[0] = 0;
    temp_storage_allocated = true;
  }

  // Cub segmented sort
  size_t required_bytes = 0;
  GUARD_CU(cub::DeviceSegmentedRadixSort::SortKeys(
    (char*)NULL, required_bytes,
    elements, elements, num_elements,
    num_segments, seg_starts, seg_starts + 1,
    0, sizeof(T) * 8, stream));

  GUARD_CU(GarenteeAllocation(temp_storage[0],
    temp_storage_bytes[0], required_bytes, malloc_type, flags));

  GUARD_CU(cub::DeviceSegmentedRadixSort::SortKeys(
    temp_storage[0], temp_storage_bytes[0],
    elements, elements, num_elements,
    num_segments, seg_starts, seg_starts + 1,
    0, sizeof(T) * 8, stream));

  if (temp_storage_allocated) {
    GUARD_CU(Free(temp_storage[0], malloc_type));
    free(temp_storage);
    free(temp_storage_bytes);
    temp_storage = NULL;
    temp_storage_bytes = NULL;
    temp_storage_allocated = false;
  }

  return retval;
}

// Segmented sort with default less than comparator
template <typename T, typename SizeT>
cudaError_t SegSort(
  T      *elements,
  SizeT   num_elements,
  SizeT  *seg_starts,
  int     num_segments,
  cudaStream_t stream = 0,
  Malloc_t malloc_type = Malloc_t::Default,
  char       **temp_storage = NULL,
  size_t      *temp_storage_bytes = NULL,
  unsigned int flags = cudaMemAttachGlobal)
{
  return SegSort(elements, num_elements, seg_starts, num_segments,
    [] __host__ __device__ (T a, T b){ return a < b;},
    stream, malloc_type, temp_storage, temp_storage_bytes, flags);
}

// Local gradient clipping
template <typename T>
cudaError_t ClipGradient(
  T          *gradients,
  std::vector<std::pair<std::string, uint64_t> > &layers,
             // <name, #elements> of layers
  DgcConfig  &config,
  DgcState   &state,
  DgcToken   *token)
{
  cudaError_t retval = cudaSuccess;

  // skip first step, because total number of layers are unknown
  if (state.step == 0)
    return retval;

  int num_layers = layers.size();
  GUARD_CU(GarenteeAllocation(state.temp_storage, state.temp_storage_bytes,
    sizeof(T) * 2 * num_layers + sizeof(uint32_t) * (num_layers + 1)));
  GUARD_CU(GarenteeAllocation(token -> h_layer_starts,
    token -> h_layer_starts_allocated, num_layers + 1, Malloc_t::Host));
  uint32_t start_counter = 0;
  for (int i = 0; i < num_layers; i++) {
    token -> h_layer_starts[i] = start_counter;
    start_counter += layers[i].second;
  }
  token -> h_layer_starts[num_layers] = start_counter;

  T* sums         = (T*)(state.temp_storage);
  T* coefficients = (T*)(state.temp_storage + sizeof(T) * num_layers);
  uint32_t* layer_starts
    = (uint32_t*)(state.temp_storage + sizeof(T) * 2 * num_layers);
  auto stream     = config.stream;
  int  grid_size  = config.grid_size;
  int  block_size = config.block_size;
  auto clipping_threshold = config.clipping_threshold;

  GUARD_CU(Memset(sums, 0, num_layers, Malloc_t::Default, stream));
  GUARD_CU2("cudaMemcpyAsync",
    cudaMemcpyAsync(layer_starts, token -> h_layer_starts,
      sizeof(uint32_t) * (num_layers + 1), cudaMemcpyHostToDevice, stream));

  // Get per -layer L2 norms
  L2norm_kernel<<<grid_size, block_size, 0, stream>>>(
    gradients, layer_starts, num_layers, sums);

  int total_num_layers = state.layer_offset_bytes.size();
  uint64_t total_num_gradients = state.offset_byte_counter / sizeof(T);

  // Get per-layer coefficients
  loop_kernel<<<grid_size, block_size, 0, stream>>>(num_layers,
    [sums, coefficients, total_num_layers, total_num_gradients,
    clipping_threshold, layer_starts]
    __device__ (const int &layer) {
      coefficients[layer] = clipping_threshold /
        (sqrt(sums[layer]) * total_num_layers + 1e-6);
    });

  // Update gradients
  loop_kernel<<<grid_size, block_size, 0, stream>>>(start_counter,
    [layer_starts, gradients, coefficients, num_layers]
    __device__ (const uint32_t &i) {
      int layer = binarySearch(layer_starts, 0, num_layers, i);
      auto coefficient = coefficients[layer];
      if (coefficient < 1)
        gradients[i] *= coefficient;
    });

  return retval;
}

cudaError_t DgcToken::Init()
{
  cudaError_t retval = cudaSuccess;
  GUARD_CU2("cudaEventCreateWithFlags",
    cudaEventCreateWithFlags(&(this -> dgc_finish    ), cudaEventDisableTiming));
  GUARD_CU2("cudaEventCreateWithFlags",
    cudaEventCreateWithFlags(&(this -> stream2_begin ), cudaEventDisableTiming));
  GUARD_CU2("cudaEventCreateWithFlags",
    cudaEventCreateWithFlags(&(this -> stream2_finish), cudaEventDisableTiming));
  GUARD_CU2("cudaEventCreateWithFlags",
    cudaEventCreateWithFlags(&(this -> stream3_begin ), cudaEventDisableTiming));
  return retval;
}

cudaError_t DgcToken::isFinished(bool &finished, int check)
{
  cudaError_t retval = cudaSuccess;
  if (this -> dgc_finished) {
    finished = true;
    return retval;
  }

  retval = cudaEventQuery(this -> dgc_finish);
  if (retval == cudaSuccess) {
    finished = true;
    this -> dgc_finished = true;
  } else if (retval == cudaErrorNotReady) {
    finished = false;
    retval = cudaSuccess;
  }
  return retval;
}

cudaError_t MaskToken::Init()
{
  cudaError_t retval = cudaSuccess;
  GUARD_CU2("cudaEventCreateWithFlags",
    cudaEventCreateWithFlags(&(this -> d2h_finish), cudaEventDisableTiming));
  GUARD_CU2("cudaEventCreateWithFlags",
    cudaEventCreateWithFlags(&(this -> h2d_finish), cudaEventDisableTiming));
  return retval;
}

cudaError_t MaskToken::isFinished(bool &finished, int check)
{
  cudaError_t retval = cudaSuccess;
  if (check == 0) {
    if (this -> d2h_finished) {
      finished = true;
      return retval;
    }
    retval = cudaEventQuery(this -> d2h_finish);
    if (retval == cudaSuccess) {
      finished = true;
      this -> d2h_finished = true;
    } else if (retval == cudaErrorNotReady) {
      finished = false;
      retval = cudaSuccess;
    }
  }

  else if (check == 1)
  {
    if (this -> mpi_finished) {
      finished = true;
      return retval;
    }
    if (!this -> mpi_started) {
      finished = false;
      return retval;
    }

    int flag = 0;
    GUARD_MPI2("MPI_Test",
      MPI_Test(&(this -> mpi_request), &flag, MPI_STATUS_IGNORE));
    if (flag) {
      //printf("\t token = %p, received %ld masks from MPI, "
      //       "first 3: %#X, %#X, %#X\n",
      //  this, (long)this -> num_masks,
      //  this -> h_recv_masks[0], this -> h_recv_masks[1],
      //  this -> h_recv_masks[2]);
      finished = true;
      this -> mpi_finished = true;
      this -> mpi_started = false;
    } else {
      finished = false;
    }
  }

  else if (check == 2) {
    if (this -> h2d_finished) {
      finished = true;
      return retval;
    }
    retval = cudaEventQuery(this -> h2d_finish);
    if (retval == cudaSuccess) {
      finished = true;
      this -> h2d_finished = true;
    } else if (retval == cudaErrorNotReady) {
      finished = false;
      retval = cudaSuccess;
    }
  }

  return retval;
}

template <typename TokenT>
cudaError_t GetToken(
  std::list<TokenT*> &free_queue,
  std::list<TokenT*> &busy_queue,
  TokenT* &token,
  int check = 0)
{
  cudaError_t retval = cudaSuccess;
  if (free_queue.size() != 0) {
    token = free_queue.front();
    free_queue.pop_front();
    return retval;
  }

  if (busy_queue.size() != 0 && check != -1) {
    auto first_token = busy_queue.front();
    bool finished = false;
    GUARD_CU(first_token -> isFinished(finished, check));
    if (finished) {
      token = first_token;
      busy_queue.pop_front();
      return retval;
    }
  }

  token = new TokenT;
  GUARD_CU(token -> Init());
  return retval;
}

// Wait for mask to be ready on host and push MPIAllReduce
cudaError_t TryPushMask(
  int             max_requests_allowed_waiting,
  DgcConfig      &config,        // DGC configuration
  DgcState       &state)         // DGC running states
{
  cudaError_t retval = cudaSuccess;
  if (max_requests_allowed_waiting != 0) {
    int total_num_layers = 0;
    for (auto &token : state.d2h_mask_queue) {
      total_num_layers += token -> num_layers;
    }
    // if the whole model is waiting, push everything out
    if (total_num_layers >= state.layer_offset_bytes.size())
      max_requests_allowed_waiting = 0;
  }

  while (state.d2h_mask_queue.size() > max_requests_allowed_waiting) {
    auto token = state.d2h_mask_queue.front();
    GUARD_CU2("cudaEventSynchronize",
      cudaEventSynchronize(token -> d2h_finish));
    token -> d2h_finished = true;

    state.d2h_mask_queue.pop_front();
    GUARD_MPI2("MPI_Iallreduce",
      MPI_Iallreduce(
        token -> h_send_masks, token -> h_recv_masks, (int)token -> num_masks,
        PreDefinedValues<uint32_t>::getMpiDataType(), MPI_BOR,
        config.use_hierarchical_allreduce ? config.cross_comm : config.mpi_comm,
        &(token -> mpi_request)));
    token -> mpi_started  = true;
    token -> mpi_finished = false;
    //printf("%ld\t token = %p, %ld masks pushed to MPI, "
    //       "first 3: %#X, %#X, %#X\n",
    //  (long)state.step, token, (long)token -> num_masks,
    //  token -> h_send_masks[0], token -> h_send_masks[1],
    //  token -> h_send_masks[2]);
    state.mpi_mask_queue.push_back(token);
  }
  return retval;
}

// Learning rate adjustment via gradient, only use if really necessary
template <typename T, typename SizeT>
cudaError_t LearningRateAdjustment(
  T              *gradients,
  SizeT           num_gradients,
  uint64_t        epoch,
  DgcConfig      &config,
  DgcState       &state)
{
  cudaError_t retval = cudaSuccess;
  float learning_rate_adjustment = 1;
  auto epoch_ = epoch;
  while (epoch_ >= config.num_epochs_per_decay)
  {
    learning_rate_adjustment *= config.learning_rate_decay_factor;
    epoch_ -= config.num_epochs_per_decay;
  }
  if (learning_rate_adjustment < config.min_learning_rate_factor)
    learning_rate_adjustment = config.min_learning_rate_factor;
  //if (config.global_gpu_rank == 0)
  //  printf("%ld\t learning_rate_adjustment = %f\n",
  //    (long)state.step, learning_rate_adjustment);

  loop_kernel <<<config.grid_size, config.block_size, 0, config.stream>>>(
    num_gradients,
    [learning_rate_adjustment, gradients] __device__ (const SizeT &i)
    {
      gradients[i] *= learning_rate_adjustment;
    });

  return retval;
}

// Main DGC routine
template <typename T, typename SizeT>
cudaError_t GradientAllReduce(
  T              *input_gradients,     // GPU pointer to the input_gradients
  T              *output_gradients,     // GPU pointer to the output_gradients
  std::vector<std::pair<std::string, uint64_t> > &layers,
                                  // <name, #elements> of layers
  DgcConfig      &config,        // DGC configuration
  DgcState       &state)         // DGC running states
{
  cudaError_t retval = cudaSuccess;
  auto  block_size   = config.block_size;
  auto  grid_size    = config.grid_size;
  auto  stream       = config.stream;
  int   num_layers   = layers.size();
  SizeT num_gradients = 0;

  // find the step number
  for (auto& layer : layers) {
    auto name = layer.first;
    auto counter_it = state.step_counters.find(name);
    if (counter_it == state.step_counters.end())
      state.step_counters[name] = 0;
    else {
      auto step = counter_it -> second;
      counter_it -> second ++;
      if (state.step < step)
        state.step = step;
    }

    num_gradients += layer.second;
  }

  // Determine the epoch number
  uint64_t num_examples_per_step
    = config.batch_size_per_gpu * config.global_num_gpus;
  uint64_t steps_per_epoch
    = config.num_examples_per_epoch / num_examples_per_step;
  if (steps_per_epoch * num_examples_per_step < config.num_examples_per_epoch)
    steps_per_epoch ++;
  uint64_t epoch    = state.step * 1.0 / steps_per_epoch;

  // if bypass both momentum correction and gradient accumulation
  if (!config.use_momentum_correction && !config.use_gradient_accumulation) {
    GUARD_NCCL2("ncclAllReduce",
      ncclAllReduce(input_gradients, output_gradients, num_gradients,
      PreDefinedValues<T>::NCCLDataType, ncclSum,
      config.use_hierarchical_allreduce ?
      config.nccl_cross_comm : config.nccl_comm, stream));

    GUARD_CU(LearningRateAdjustment(output_gradients,
      num_gradients, epoch, config, state));
    return retval;
  }

  // Calcuate sparsity based on epoch number
  double sparsity   = config.final_sparsity;
  if (epoch < config.warmup_epochs) {
    auto init_comm_rate = 1 - config.init_sparsity;
    auto final_comm_rate = 1 - config.final_sparsity;
    if (config.smooth_sparsity) {
      auto comm_rate = init_comm_rate * exp(
        log(final_comm_rate / init_comm_rate)
        / config.warmup_epochs * state.step * 1.0 / steps_per_epoch);
      sparsity = 1 - comm_rate;
    } else {
      auto comm_rate = init_comm_rate * exp(
        log(final_comm_rate / init_comm_rate)
        / config.warmup_epochs * epoch);
    }

    //if (epoch * steps_per_epoch == state.step && config.global_gpu_rank == 0)
    if (config.global_gpu_rank == 0)
      printf("Epoch %ld, Step %ld, sparsity = %lf\n",
        epoch, state.step, sparsity);
  }
  SizeT  target_num = num_gradients * (1 - sparsity);

  // Prepare token and streams
  DgcToken *token = NULL;
  GUARD_CU(GetToken(state.free_tokens, state.busy_tokens, token));
  if (config.stream2 == 0) {
    int greatest_priority;
    GUARD_CU2("cudaDeviceGetStreamPriorityRange",
      cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
    GUARD_CU2("cudaStreamCreateWithPriority",
      cudaStreamCreateWithPriority(&(config.stream2), cudaStreamNonBlocking,
        greatest_priority));
    GUARD_CU2("cudaStreamCreateWithPriority",
      cudaStreamCreateWithPriority(&(config.stream3), cudaStreamNonBlocking,
        greatest_priority));
    GUARD_CU2("cudaStreamCreateWithPriority",
      cudaStreamCreateWithPriority(&(config.stream4), cudaStreamNonBlocking,
        greatest_priority));
  }
  auto stream2 = config.stream2;
  auto stream3 = config.stream3;
  auto stream4 = config.stream4;

  if (config.local_gradient_clipping)
    GUARD_CU(ClipGradient(input_gradients, layers, config, state, token));

  // find which step is currently in and look for unallocated layers
  std::vector<std::pair<std::string, uint64_t> > layers_to_allocate;
  SizeT num_gradients_to_allocate = 0;
  for (auto &layer : layers)
  {
    auto name = layer.first;
    auto offset_it = state.layer_offset_bytes.find(name);
    if (offset_it == state.layer_offset_bytes.end()) {
      layers_to_allocate.push_back(std::make_pair(layer.first, layer.second));
      num_gradients_to_allocate += layer.second;
    }
  }

  // Test persistent memory
  //char* state_pervious_verlocity = NULL;
  //for (int i = 0; i < 1000; i++)
  //{
  //  char *temp_ptr = NULL;
  //  size_t size = 1024 * 1024;
  //  size *= 64;
  //  GUARD_CU(MallocPersistent("temp" + std::to_string(i), temp_ptr,
  //    size, config, state));

  //  GUARD_CU(FreePersistent("temp" + std::to_string(i), temp_ptr,
  //    config, state));
  //}
  // allocate new layers
  if (num_gradients_to_allocate > 0) {
    if (config.use_momentum_correction) {
      //GUARD_CU(GarenteeAllocation(state.pervious_verlocity,
      //  state.pervious_verlocity_allocated,
      //  state.offset_byte_counter + sizeof(T) * num_gradients_to_allocate,
      //  Malloc_t::Default, cudaMemAttachGlobal, stream, true, true));
      GUARD_CU(GarenteeAllocationPersistent("pervious_verlocity",
        state_pervious_verlocity,
        state.offset_byte_counter + sizeof(T) * num_gradients_to_allocate,
        config, state, stream, true, true));
      GUARD_CU(GarenteeAllocation(state.pervious_accumulated_verlocity,
        state.pervious_accumulated_verlocity_allocated,
        state.offset_byte_counter + sizeof(T) * num_gradients_to_allocate,
        Malloc_t::Default, cudaMemAttachGlobal, stream, true, true));
    } else {
      GUARD_CU(GarenteeAllocation(state.pervious_accumulated_gradients,
        state.pervious_accumulated_gradients_allocated,
        state.offset_byte_counter + sizeof(T) * num_gradients_to_allocate,
        Malloc_t::Default, cudaMemAttachGlobal, stream, true, true));
    }
    for (auto& layer : layers_to_allocate) {
      state.layer_offset_bytes[layer.first] = state.offset_byte_counter;
      state.offset_byte_counter += layer.second * sizeof(T);
    }
  }
  GUARD_CU(AccessPersistent("pervious_verlocity", state_pervious_verlocity,
    config, state));
  GUARD_CU(GarenteeAllocation(token -> h_layer_starts,
    token -> h_layer_starts_allocated, num_layers + 1, Malloc_t::Host));

  // find continous layers as
  // <start, size, offset> of chunks
  std::vector<std::tuple<SizeT, SizeT, size_t> > chunks;
  size_t chunk_offset_bytes = state.layer_offset_bytes[layers.begin() -> first];
  SizeT  layer_start = 0;
  SizeT  chunk_start = 0;
  SizeT  chunk_size  = 0;
  for (int i = 0; i < num_layers; i++) {
    auto &layer = layers[i];
    token -> h_layer_starts[i] = layer_start;
    if (chunk_offset_bytes + chunk_size * sizeof(T) !=
      state.layer_offset_bytes[layer.first]) {
      // mismatch, means new layer starts
      chunks.push_back(std::make_tuple(
        chunk_start, chunk_size, chunk_offset_bytes));
      chunk_size  = 0;
      chunk_start = layer_start;
      chunk_offset_bytes = state.layer_offset_bytes[layer.first];
    }

    chunk_size  += layer.second;
    layer_start += layer.second;
  } // end of for layers
  token -> h_layer_starts[num_layers] = layer_start;
  if (chunk_size != 0)
    chunks.push_back(std::make_tuple(
      chunk_start, chunk_size, chunk_offset_bytes));

  auto &layer_starts = state.layer_starts;
  GUARD_CU(GarenteeAllocation(state.layer_starts,
    state.layer_starts_allocated, num_layers + 1));
  GUARD_CU2("cudaMemcpyAsync",
    cudaMemcpyAsync(state.layer_starts, token -> h_layer_starts,
      sizeof(uint32_t) * (num_layers + 1), cudaMemcpyHostToDevice, stream));

  // Memory allocation and type conversion
  if (config.use_momentum_correction) {
    GUARD_CU(GarenteeAllocation(state.verlocity,
      state.verlocity_allocated, num_gradients * sizeof(T)));
    GUARD_CU(GarenteeAllocation(state.accumulated_verlocity,
      state.accumulated_verlocity_allocated, num_gradients * sizeof(T)));
  } else {
    GUARD_CU(GarenteeAllocation(state.accumulated_gradients,
      state.accumulated_gradients_allocated, num_gradients * sizeof(T)));
  }
  T* verlocity = (T*)(state.verlocity);
  T* accumulated_verlocity = (T*)(state.accumulated_verlocity);
  T* accumulated_gradients = (T*)(state.accumulated_gradients);
  T* elements = NULL;

  if (config.use_momentum_correction) {
    // momentum correction by chunks
    for (auto& chunk : chunks) {
      SizeT chunk_start = std::get<0>(chunk);
      SizeT chunk_size  = std::get<1>(chunk);
      size_t chunk_offset = std::get<2>(chunk);

      T* pervious_verlocity
        = (T*)(state_pervious_verlocity + chunk_offset);
      T* pervious_accumulated_verlocity
        = (T*)(state.pervious_accumulated_verlocity + chunk_offset);
      auto &momentum = config.momentum;

      loop_kernel<<<grid_size, block_size, 0, stream>>>(chunk_size,
        [momentum, input_gradients, chunk_start,
        pervious_verlocity, verlocity,
        accumulated_verlocity, pervious_accumulated_verlocity]
        __device__ (const SizeT &i) {
          auto pos = i + chunk_start;
          auto u = pervious_verlocity[i] * momentum + input_gradients[pos];
          accumulated_verlocity[pos] = pervious_accumulated_verlocity[i] + u;
          verlocity[pos] = u;
        });
    }
    elements = accumulated_verlocity;
  }

  else {
    // accumulate gradients
    for (auto& chunk : chunks) {
      SizeT chunk_start = std::get<0>(chunk);
      SizeT chunk_size  = std::get<1>(chunk);
      size_t chunk_offset = std::get<2>(chunk);

      T* pervious_accumulated_gradients
        = (T*)(state.pervious_accumulated_gradients + chunk_offset);

      loop_kernel<<<grid_size, block_size, 0, stream>>>(chunk_size,
        [input_gradients, chunk_start,
        accumulated_gradients, pervious_accumulated_gradients]
        __device__ (const SizeT &i) {
          auto pos = i + chunk_start;
          auto g = pervious_accumulated_gradients[i] + input_gradients[pos];
          accumulated_gradients[pos] = g;
        });
    }
    elements = accumulated_gradients;
  }

  // Prepare for mask communication overlapping
  bool to_overlap_mask = config.use_allReduce && config.overlap_mask_allreduce;
  if (to_overlap_mask) {
    GUARD_CU2("cudaEventRecord",
      cudaEventRecord(token -> stream3_begin, stream));
    GUARD_CU2("cudaStreamWaitEvent",
      cudaStreamWaitEvent(stream3, token -> stream3_begin, 0));
  }

  // Communicate all gradients if it's a flushing step
  bool to_flush = false;
  if (config.flush_steps > 0) {
    if ((state.step >= config.flush_steps) &&
        (state.step % config.flush_steps) == 0)
      to_flush = true;
  }
  if (to_flush) {
    //printf("%ld\t Flushing %ld elements\n",
    //  (long)state.step, (long)num_gradients);

    GUARD_CU2("cudaEventRecord",
      cudaEventRecord(token -> stream2_begin, stream));
    GUARD_NCCL2("ncclAllReduce",
      ncclAllReduce(elements, output_gradients,
        (size_t)num_gradients, PreDefinedValues<T>::NCCLDataType, ncclSum,
        config.use_hierarchical_allreduce ?
        config.nccl_cross_comm : config.nccl_comm, stream));

    GUARD_CU2("cudaStreamWaitEvent",
      cudaStreamWaitEvent(stream2, token -> stream2_begin, 0));
    if (config.use_momentum_correction) {
      for (auto& chunk : chunks) {
        SizeT  chunk_start  = std::get<0>(chunk);
        SizeT  chunk_size   = std::get<1>(chunk);
        size_t chunk_offset = std::get<2>(chunk);

        T* pervious_verlocity
          = (T*)(state_pervious_verlocity + chunk_offset);
        T* pervious_accumulated_verlocity
          = (T*)(state.pervious_accumulated_verlocity + chunk_offset);

        GUARD_CU(Memset(pervious_verlocity,
          0, chunk_size, Malloc_t::Default, stream2));
        GUARD_CU(Memset(pervious_accumulated_verlocity,
          0, chunk_size, Malloc_t::Default, stream2));
      }
    }

    else {
      for (auto& chunk : chunks) {
        SizeT chunk_start = std::get<0>(chunk);
        SizeT chunk_size  = std::get<1>(chunk);
        size_t chunk_offset = std::get<2>(chunk);

        T* pervious_accumulated_gradients
          = (T*)(state.pervious_accumulated_gradients + chunk_offset);
        GUARD_CU(Memset(pervious_accumulated_gradients,
          0, chunk_size, Malloc_t::Default, stream2));
      }
    }
    GUARD_CU2("cudaEventRecord",
      cudaEventRecord(token -> stream2_finish, stream2));

    if (config.learning_rate_decay_factor > 0 &&
        epoch >= config.num_epochs_per_decay) {
      GUARD_CU(LearningRateAdjustment(output_gradients,
        num_gradients, epoch, config, state));
    }

    GUARD_CU2("cudaStreamWaitEvent",
      cudaStreamWaitEvent(stream, token -> stream2_finish, 0));
    GUARD_CU2("cudaEventRecord",
      cudaEventRecord(token -> dgc_finish, stream));
    token -> dgc_finished = false;
    state.busy_tokens.push_back(token);

    return retval;
  }

  // Sampling
  auto &samp_starts = state.samp_starts;
  GUARD_CU(GarenteeAllocation(state.samp_starts, state.samp_starts_allocated,
    num_layers + 1));
  GUARD_CU(GarenteeAllocation(token -> h_samp_starts,
    token -> h_samp_starts_allocated, num_layers + 1, Malloc_t::Host));
  uint32_t samp_counter = 0;
  // Find number of samples for each layer
  for (int i = 0; i < num_layers; i++) {
    auto &layer = layers[i];
    token -> h_samp_starts[i] = samp_counter;

    uint32_t num_samples = 0;
    if (config.sampling_rate < 1 &&
        layer.second > config.min_sampling_num) {

      num_samples = layer.second * config.sampling_rate;
      if (num_samples < config.min_sampling_num)
        num_samples = config.min_sampling_num;
      uint32_t num_selected_samples = config.min_gradients_comm_per_layer
        * config.sampling_rate;
      if (num_selected_samples < config.min_selected_samples_per_layer)
        num_selected_samples = config.min_selected_samples_per_layer;
      if (num_samples < num_selected_samples * 1.0f / (1 - sparsity)) {
        num_samples = num_selected_samples * 1.0f / (1 - sparsity);
      }
      if (num_samples > layer.second)
        num_samples = layer.second;
    }

    else { // no sampling
      num_samples = layer.second;
    }
    samp_counter += num_samples;
  }
  token -> h_samp_starts[num_layers] = samp_counter;
  GUARD_CU2("cudaMemcpyAsync",
    cudaMemcpyAsync(state.samp_starts, token -> h_samp_starts,
      sizeof(uint32_t) * (num_layers + 1), cudaMemcpyHostToDevice,
      (to_overlap_mask ? stream3 : stream)));

  // Prepare rand states
  auto &rand_states = state.rand_states;
  auto &rand_seed   = config.rand_seed;
  if (rand_states == NULL) {
    GUARD_CU(Malloc(rand_states, grid_size * block_size));

    loop_kernel
      <<<grid_size, block_size, 0, (to_overlap_mask ? stream3 : stream)>>>(
      (SizeT)grid_size * block_size,
      [rand_states, rand_seed] __device__ (const SizeT &i){
        curand_init(rand_seed, i, 0, rand_states + i);
      });
  }

  GUARD_CU(GarenteeAllocation(state.samp_data, state.samp_allocated,
    samp_counter * sizeof(T)));
  T* samp_data = (T*)(state.samp_data);

  sample_kernel2
    <<<grid_size, block_size, 0, (to_overlap_mask ? stream3 : stream)>>>(
    elements, num_gradients,
    state.layer_starts, num_layers,
    state.samp_starts, samp_data, state.rand_states);

  // Sort the samples
  GUARD_CU(SegSort(samp_data, samp_counter, state.samp_starts, num_layers,
    (to_overlap_mask ? stream3 : stream), Malloc_t::Default,
    &(state.temp_storage), &(state.temp_storage_bytes)));

  //loop_kernel<<<1, 1, 0, (to_overlap_mask ? stream3 : stream)>>>((SizeT)1,
  //  [threshold, samp_data, num_samples, sparsity] __device__ (const SizeT &i){
  //    SizeT pos = num_samples * sparsity;
  //    if (pos >= num_samples)
  //      pos = num_samples - 1;
  //    threshold[0] = samp_data[pos];
  //    //printf("selecting samp[%d] from [%d] {%f, %f, ... %f, %f, %f, ... %f, %f}\n",
  //    //  pos, num_samples,
  //    //  num_samples > 0 ? samp_data[0] : -1,
  //    //  num_samples > 1 ? samp_data[1] : -1,
  //    //  num_samples + 1 > pos  && pos > 0 ? samp_data[pos - 1] : -1,
  //    //  num_samples > pos && pos >= 0 ? samp_data[pos] : -1,
  //    //  num_samples > pos + 1 && pos + 1 >= 0 ? samp_data[pos + 1] : -1,
  //    //  num_samples > 1 ? samp_data[num_samples - 2] : -1,
  //    //  num_samples > 0 ? samp_data[num_samples - 1] : -1);
  //  });
  auto &thresholds = state.thresholds;
  auto &min_selected_samples_per_layer = config.min_gradients_comm_per_layer;
  GUARD_CU(
    GarenteeAllocation(thresholds, state.thresholds_allocated, num_layers));

  // Get the per-layer thresholds
  loop_kernel
    <<<grid_size, block_size, 0, (to_overlap_mask ? stream3 : stream)>>>(
      num_layers, [thresholds, samp_data, samp_starts, sparsity,
      min_selected_samples_per_layer]
    __device__ (const int &layer) {
      auto samp_start = samp_starts[layer];
      auto samp_end   = samp_starts[layer + 1];
      auto samp_size  = samp_end - samp_start;
      SizeT pos = samp_size * sparsity;
      if (pos >= samp_size)
        pos = samp_size;
      if (min_selected_samples_per_layer < samp_size &&
        pos > samp_size - min_selected_samples_per_layer)
        pos = samp_size - min_selected_samples_per_layer;
      thresholds[layer] = samp_data[samp_start + pos];
    });

  if (config.use_allReduce) {
    // use allReduce on mask to communicate
    SizeT num_masks = num_gradients / 32;
    if (num_masks * 32 < num_gradients)
      num_masks ++;

    // Garentee sufficient memory allocation
    auto mask_allocated_ = state.mask_allocated;
    GUARD_CU(GarenteeAllocation(state.send_masks  , mask_allocated_, num_masks));
    mask_allocated_ = state.mask_allocated;
    GUARD_CU(GarenteeAllocation(state.recv_masks  , mask_allocated_, num_masks));
    if (!config.overlap_mask_allreduce) {
      mask_allocated_ = state.mask_allocated;
      GUARD_CU(GarenteeAllocation(state.h_send_masks, mask_allocated_,
        num_masks, Malloc_t::Host));
      mask_allocated_ = state.mask_allocated;
      GUARD_CU(GarenteeAllocation(state.h_recv_masks, mask_allocated_,
        num_masks, Malloc_t::Host));
    }
    if (state.mask_allocated < num_masks)
      state.mask_allocated = num_masks;

    auto &mask_counters = state.mask_counters;
    auto &mask_offsets  = state.mask_offsets;
    GUARD_CU(GarenteeAllocation(
        mask_counters, state.mask_counters_allocated, (num_masks + 1)));
    GUARD_CU(GarenteeAllocation(
        mask_offsets , state.mask_offsets_allocated , (num_masks + 1)));

    if (state.h_num_gradients_to_communicate == NULL)
        GUARD_CU(Malloc(state.h_num_gradients_to_communicate, 1, Malloc_t::Host));

    // Prepare the mask
    auto &send_masks = state.send_masks;
    auto &recv_masks = state.recv_masks;
    loop_kernel
      <<<grid_size, block_size, 0, (to_overlap_mask ? stream3 : stream)>>>(
        num_masks, [send_masks, num_gradients, thresholds,
        layer_starts, num_layers, elements]
      __device__ (const SizeT &i) {
        uint32_t mask = 0;
        SizeT offset = i * 32;
        int end_j = 32, j = 0;
        if (offset + end_j > num_gradients)
          end_j = num_gradients - offset;

        while (j < end_j) {
          auto pos = j + offset;
          T element = elements[pos];
          int layer = binarySearch(layer_starts, 0, num_layers, pos);
          if (!isfinite(element * 1.0f)) {
            j ++;
            continue;
          }

          if (!(abs(element) < thresholds[layer])) {
            mask |= (((uint32_t)1) << j);
          }
          j++;
        }
        send_masks[i] = mask;
      });

    if (config.overlap_mask_allreduce) {
      MaskToken *mask_token = NULL;

      // Get token and allocate host space
      GUARD_CU(GetToken(state.free_mask_tokens, state.h2d_mask_queue,
        mask_token, 2));
      mask_allocated_ = mask_token -> mask_allocated;
      GUARD_CU(GarenteeAllocation(mask_token -> h_send_masks, mask_allocated_,
        num_masks, Malloc_t::Host));
      mask_allocated_ = mask_token -> mask_allocated;
      GUARD_CU(GarenteeAllocation(mask_token -> h_recv_masks, mask_allocated_,
        num_masks, Malloc_t::Host));
      if (mask_token -> mask_allocated < num_masks)
        mask_token -> mask_allocated = num_masks;

      // Move the send mask from GPU to CPU
      GUARD_CU2("cudaMemcpyAsync",
        cudaMemcpyAsync(mask_token -> h_send_masks, send_masks,
          sizeof(uint32_t) * num_masks, cudaMemcpyDeviceToHost, stream3));
      GUARD_CU2("cudaEventRecord",
        cudaEventRecord(mask_token -> d2h_finish, stream3));
      mask_token -> d2h_finished = false;

      // Record the token for every layer
      auto &current_layer_records = state.layer_records[state.step % 2];
      uint32_t layer_start = 0;
      for (auto &layer : layers) {
        auto name = layer.first;
        current_layer_records[name].token = mask_token;
        current_layer_records[name].layer_start = layer_start;
        layer_start += layer.second;
      }
      mask_token -> num_masks = num_masks;
      mask_token -> num_layers = layers.size();
      if (state.step + 1 == config.overlap_skip_steps)
        mask_token -> num_layers_produced = layers.size() * 2;
      else
        mask_token -> num_layers_produced = layers.size();
      mask_token -> num_layers_comsumed = 0;
      state.d2h_mask_queue.push_back(mask_token);

      if (state.step < config.overlap_skip_steps)
      {
        // Force sync mask communication for the first few steps
        GUARD_CU(TryPushMask(0, config, state));
      }

      // wait for the mask from pervious step
      bool all_layers_ready = false;
      int pervious_step_index = (state.step < config.overlap_skip_steps) ?
        (state.step % 2) : ((state.step + 1) % 2);
      auto &pervious_layer_records = state.layer_records[pervious_step_index];

      while (!all_layers_ready) {
        all_layers_ready = true;
        for (auto &layer : layers) {
          auto name   = layer.first;
          auto record = pervious_layer_records[name];
          bool finished = false;
          GUARD_CU(record.token -> isFinished(finished, 1));
          if (!finished) {
            all_layers_ready = false;
            break;
          }
        }

        if (!all_layers_ready) {
          std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
      }

      // Reuse temp_storage to hold masks before bit-swift copy to recv_masks
      size_t request_bytes = sizeof(uint32_t) * (num_masks + layers.size() * 2);
      GUARD_CU(GarenteeAllocation(state.temp_storage2, state.temp_storage2_bytes,
        request_bytes));
      uint32_t* temp_masks_ = (uint32_t*)(state.temp_storage2);

      GUARD_CU(
        Memset(recv_masks + num_masks - 1, 0, 1, Malloc_t::Default, stream4));
      // move to GPU with bit swift
      SizeT chunk_start = 0;
      SizeT chunk_size = 0;
      SizeT temp_start = 0;
      uint32_t chunk_num_layers = 0;
      SizeT pervious_chunk_start = 0;
      MaskToken *current_token = NULL;
      for (auto i = 0; i <= layers.size(); i++) {
        if (layers.empty())
          break;
        auto layer = layers[(i == layers.size()) ? i - 1 : i];
        auto name = layer.first;
        auto layer_size = layer.second;
        auto record = pervious_layer_records[name];
        bool new_chunk = false;

        if (i == layers.size()) {
        current_token = record.token;
          new_chunk = true;
        } else if (current_token == NULL) {
          new_chunk = false;
          current_token = record.token;
          pervious_chunk_start = record.layer_start;
        } else if (current_token != record.token)
          new_chunk = true;
        else if (pervious_chunk_start + chunk_size != record.layer_start)
          new_chunk = true;

        if (new_chunk) {
          if (chunk_size != 0) {
            SizeT dest_mask_start  = chunk_start / 32;
            SizeT dest_mask_offset = chunk_start % 32;
            int dest_mask_end    = (chunk_start + chunk_size) / 32;
            if (dest_mask_end * 32 != chunk_start + chunk_size)
              dest_mask_end += 1;
            SizeT dest_mask_size   = dest_mask_end - dest_mask_start;
            uint32_t *dest_masks   = recv_masks + dest_mask_start;

            SizeT src_mask_start   = pervious_chunk_start / 32;
            SizeT src_mask_end     = (pervious_chunk_start + chunk_size) / 32;
            int src_mask_offset  = pervious_chunk_start % 32;
            if (src_mask_end * 32 != pervious_chunk_start + chunk_size)
              src_mask_end += 1;
            SizeT src_mask_size = src_mask_end - src_mask_start;
            uint32_t *temp_masks = temp_masks_ + temp_start;
            int ro = src_mask_offset - dest_mask_offset; // relative offset

            //printf("%ld\t token = %p, Copy with bitswift: src = %ld + %ld, src_size = %ld, "
            //  "dest = %ld + %ld, dest_size = %ld, temp_start = %ld, ro = %d, "
            //  "token -> num_masks = %ld, chunk_size = %ld\n",
            //  (long)state.step, current_token,
            //  (long)src_mask_start, (long)src_mask_offset, (long)src_mask_size,
            //  (long)dest_mask_start, (long)dest_mask_offset, (long)dest_mask_size,
            //  (long)temp_start, ro, (long)current_token -> num_masks, (long)chunk_size);
            GUARD_CU2("cudaMemcpyAsync",
              cudaMemcpyAsync(temp_masks,
                current_token -> h_recv_masks + src_mask_start,
                sizeof(uint32_t) * src_mask_size,
                cudaMemcpyHostToDevice, stream4));

            loop_kernel<<<grid_size, block_size, 0, stream4>>>(dest_mask_size,
              [temp_masks, dest_masks, dest_mask_size, ro,
              dest_mask_offset, src_mask_offset, chunk_size]
              __device__ (const SizeT &i){
                uint32_t dest_mask = 0, mask0 = 0, mask1 = 0;
                if (i != 0 && i+1 != dest_mask_size) {
                  if (ro > 0) {
                    // move src_mask to the right
                    mask0  = temp_masks[i];
                    mask1  = temp_masks[i+1];
                    // (32-ro) bits from mask0
                    dest_mask = mask0 >> ro;
                    // ro bits from mask1
                    dest_mask |= (mask1 & ((uint32_t(1) << ro) -1)) << (32-ro);
                  } else if (ro < 0) {
                    // move src_mask to the left
                    mask0 = temp_masks[i-1];
                    mask1 = temp_masks[i];
                    // -ro bits from mask0
                    dest_mask = mask0 >> (32 + ro);
                    // (32+ro) bits from mask1
                    dest_mask |= (mask1 & ((uint32_t(1) << (32 + ro))-1)) << (-ro);
                  } else {
                    // direct copy
                    dest_mask = temp_masks[i];
                  }
                }

                else if (i == 0) {
                  // front
                  int num_gradients_in_first_mask = 32 - dest_mask_offset;
                  if (num_gradients_in_first_mask > chunk_size)
                    num_gradients_in_first_mask = chunk_size;
                  SizeT pervious_pos = src_mask_offset;
                  dest_mask = dest_masks[i];
                  for (int k = 0; k < num_gradients_in_first_mask; k++) {
                    mask0 = temp_masks[pervious_pos / 32];
                    mask1 = (mask0 >> (pervious_pos % 32)) & uint32_t(1);
                    mask1 = mask1 << (k + dest_mask_offset);
                    dest_mask |= mask1;
                    pervious_pos ++;
                  }
                }

                else { // i+1 == dest_mask_size
                  // back
                  int num_gradients_in_last_mask
                    = (dest_mask_offset + chunk_size) % 32;
                  SizeT pervious_pos = src_mask_offset + chunk_size
                    - num_gradients_in_last_mask;
                  dest_mask = dest_masks[i];
                  for (int k = 0; k < num_gradients_in_last_mask; k++)
                  {
                    mask0 = temp_masks[pervious_pos / 32];
                    mask1 = (mask0 >> (pervious_pos % 32)) & uint32_t(1);
                    mask1 = mask1 << k;
                    dest_mask |= mask1;
                    pervious_pos ++;
                  }
                }

                dest_masks[i] = dest_mask;
              });

            temp_start = temp_start + src_mask_size;
            current_token -> num_layers_comsumed += chunk_num_layers;
            if (current_token -> num_layers_comsumed
              == current_token -> num_layers_produced) {
              GUARD_CU2("cudaEventRecord",
                cudaEventRecord(current_token -> h2d_finish, stream4));
              current_token -> h2d_finished = false;
            }
          }

          if (i == layers.size())
            break;
          pervious_chunk_start = record.layer_start;
          current_token = record.token;
          chunk_start += chunk_size;
          chunk_size = 0;
          chunk_num_layers = 0;
        }

        chunk_size += layer_size;
        chunk_num_layers ++;
      }

      while (!state.mpi_mask_queue.empty()) {
        auto first_token = state.mpi_mask_queue.front();
        if (first_token -> num_layers_comsumed !=
          first_token -> num_layers_produced)
          break;

        state.mpi_mask_queue.pop_front();
        state.h2d_mask_queue.push_back(first_token);
      }
    } // enf of if (config.overlap_mask_allreduce)

    else {
      // not overlapping mask allreduce
      GUARD_CU2("cudaMemcpyAsync",
        cudaMemcpyAsync(state.h_send_masks, send_masks,
          sizeof(uint32_t) * num_masks, cudaMemcpyDeviceToHost, stream));
      GUARD_CU2("cudaStreamSynchronize after mask",
        cudaStreamSynchronize(stream));

      GUARD_MPI2("MPI_Allreduce",
        MPI_Allreduce(state.h_send_masks, state.h_recv_masks,
          (int)num_masks, PreDefinedValues<uint32_t>::getMpiDataType(), MPI_BOR,
          config.use_hierarchical_allreduce ? config.cross_comm : config.mpi_comm));

      GUARD_CU2("cudaMemcpyAsync",
        cudaMemcpyAsync(recv_masks, state.h_recv_masks,
          sizeof(uint32_t) * num_masks, cudaMemcpyHostToDevice, stream));
    }

    // Count received mask
    loop_kernel<<<grid_size, block_size, 0,
      (to_overlap_mask ? stream4 : stream)>>>(num_masks,
      [recv_masks, mask_counters] __device__ (const SizeT &i) {
        mask_counters[i] = __popc(recv_masks[i]);
      });

    // Use inclusive sum to calculate the offsets for gradient compaction
    size_t required_bytes = 0;
    GUARD_CU(cub::DeviceScan::InclusiveSum(
      (char*)NULL, required_bytes,
      mask_counters, mask_offsets + 1, num_masks,
      (to_overlap_mask ? stream4 : stream)));
    GUARD_CU(GarenteeAllocation(
      state.temp_storage2, state.temp_storage2_bytes, required_bytes));

    GUARD_CU(Memset(mask_offsets, 0, 1, Malloc_t::Default,
      (to_overlap_mask ? stream4 : stream)));
    GUARD_CU(cub::DeviceScan::InclusiveSum(
      state.temp_storage2, required_bytes,
      mask_counters, mask_offsets + 1, num_masks,
      (to_overlap_mask ? stream4 : stream)));

    // Get the total number of gradients selected
    GUARD_CU2("cudaMemcpyAsync",
      cudaMemcpyAsync(state.h_num_gradients_to_communicate,
        mask_offsets + num_masks, sizeof(uint32_t),
        cudaMemcpyDeviceToHost, to_overlap_mask ? stream4 : stream));
    GUARD_CU2("cudaStreamSynchronize after InclusiveSum",
      cudaStreamSynchronize(to_overlap_mask ? stream4 : stream));

    auto num_gradients_comm = state.h_num_gradients_to_communicate[0];
    if (config.global_gpu_rank == 0)
      printf("%d\t #gradients to comm = %ld, #gradients = %ld, rate = %f\n",
        state.step, (long)num_gradients_comm, (long)num_gradients,
        1.0f * num_gradients_comm / num_gradients);

    auto send_allocated_ = state.send_allocated * sizeof(T);
    GUARD_CU(GarenteeAllocation(
      state.send_data, send_allocated_, sizeof(T) * num_gradients_comm));
    if (state.send_allocated < num_gradients_comm)
      state.send_allocated = num_gradients_comm;
    auto recv_allocated_ = state.recv_allocated * sizeof(T);
    GUARD_CU(GarenteeAllocation(
      state.recv_data, recv_allocated_, sizeof(T) * num_gradients_comm));
    if (state.recv_allocated < num_gradients_comm)
      state.recv_allocated = num_gradients_comm;

    // Compact gradients
    T* send_data = (T*)(state.send_data);
    T* recv_data = (T*)(state.recv_data);
    auto global_num_gpus = config.global_num_gpus;
    loop_kernel<<<grid_size, block_size, 0, stream>>>(num_masks,
      [recv_masks, mask_offsets, send_data, global_num_gpus,
      num_gradients, elements]
      __device__ (const SizeT &i) {

        uint32_t mask = recv_masks[i];
        if (mask == 0)
          return;
        SizeT offset = i * 32, output_offset = mask_offsets[i];
        int end_j = 32, j = 0, output_count = 0;
        if (offset + end_j > num_gradients)
          end_j = num_gradients - offset;

        while (j < end_j) {
          if ((mask & (((uint32_t)1) << j)) == 0) {
            j ++;
            continue;
          }
          T element = elements[j + offset];
          if (!isfinite(element * 1.0f))
            element = 0;

          send_data[output_offset + output_count] = element; // / global_num_gpus;
          output_count ++;
          j++;
        }
      });

    GUARD_CU2("cudaEventRecord",
      cudaEventRecord(token -> stream2_begin, stream));

    GUARD_NCCL2("ncclAllReduce",
      ncclAllReduce(send_data   , (void*)recv_data,
        (size_t)num_gradients_comm,
        PreDefinedValues<T>::NCCLDataType, ncclSum,
        config.use_hierarchical_allreduce ? config.nccl_cross_comm :
          config.nccl_comm, stream));

    // Unpack received gradients and indices
    GUARD_CU(
      Memset(output_gradients, 0, num_gradients, Malloc_t::Default, stream));
    loop_kernel<<<grid_size, block_size, 0, stream>>>(num_masks,
      [recv_masks, mask_offsets, recv_data, output_gradients, num_gradients]
      __device__ (const SizeT &i) {
        uint32_t mask = recv_masks[i];
        if (mask == 0)
          return;

        SizeT offset = i * 32, output_offset = mask_offsets[i];
        int end_j = 32, j = 0, output_count = 0;
        if (offset + end_j > num_gradients)
          end_j = num_gradients - offset;
        while (j < end_j) {
          if ((mask & (((uint32_t)1) << j)) == 0) {
            j ++;
            continue;
          }

          output_gradients[j + offset]
            = recv_data[output_offset + output_count];
          output_count ++;
          j++;
        }
      });

    // Updates pervious_verlocity and pervious_accumulated_verlocity
    // Can be overlap with communication
    GUARD_CU2("cudaStreamWaitEvent",
      cudaStreamWaitEvent(stream2, token -> stream2_begin, 0));
    if (config.use_momentum_correction) {
      for (auto& chunk : chunks) {
        SizeT  chunk_start  = std::get<0>(chunk);
        SizeT  chunk_size   = std::get<1>(chunk);
        size_t chunk_offset = std::get<2>(chunk);

        T* pervious_verlocity
          = (T*)(state_pervious_verlocity + chunk_offset);
        T* pervious_accumulated_verlocity
          = (T*)(state.pervious_accumulated_verlocity + chunk_offset);

        loop_kernel <<<grid_size, block_size, 0, stream2>>>(chunk_size,
          [recv_masks, chunk_start, chunk_size,
           verlocity, pervious_verlocity,
           accumulated_verlocity, pervious_accumulated_verlocity]
          __device__ (const SizeT &i) {
            auto gradient_pos = i + chunk_start;
            auto mask_pos = gradient_pos / 32;
            auto mask = recv_masks[mask_pos];
            auto mask_offset = (gradient_pos & ((uint32_t)31));

            if ((mask & (((uint32_t)1) << mask_offset)) != 0) {
              pervious_verlocity[i] = 0;
              pervious_accumulated_verlocity[i] = 0;
            } else {
              pervious_verlocity[i] = verlocity[gradient_pos];
              pervious_accumulated_verlocity[i]
                = accumulated_verlocity[gradient_pos];
            }
          });
      }
    }

    else { // gradient accumulation
      for (auto& chunk : chunks) {
        SizeT  chunk_start  = std::get<0>(chunk);
        SizeT  chunk_size   = std::get<1>(chunk);
        size_t chunk_offset = std::get<2>(chunk);

        T* pervious_accumulated_gradients
          = (T*)(state.pervious_accumulated_gradients + chunk_offset);
        loop_kernel <<<grid_size, block_size, 0, stream2>>>(chunk_size,
          [recv_masks, chunk_start, chunk_size,
          accumulated_gradients, pervious_accumulated_gradients]
          __device__ (const SizeT &i) {
            auto gradient_pos = i + chunk_start;
            auto mask_pos = gradient_pos / 32;
            auto mask = recv_masks[mask_pos];
            auto mask_offset = (gradient_pos & ((uint32_t)31));

            if ((mask & (((uint32_t)1) << mask_offset)) != 0) {
              pervious_accumulated_gradients[i] = 0;
            } else {
              pervious_accumulated_gradients[i]
                = accumulated_gradients[gradient_pos];
            }
          });
      }
    }
    GUARD_CU2("cudaEventRecord",
      cudaEventRecord(token -> stream2_finish, stream2));
    GUARD_CU2("cudaStreamWaitEvent",
      cudaStreamWaitEvent(stream, token -> stream2_finish, 0));
  } // end of if (use_allReduce)

  else {
    // use allGather to communicate
    // Pick those larger than threshold
    auto &send_counter   = state.send_counter;
    auto &send_indices   = state.send_indices;
    auto &send_allocated = state.send_allocated;
    auto send_allocated_ = send_allocated * sizeof(T);
    if (send_counter == NULL) {
      GUARD_CU(Malloc(send_counter, 1));
    }

    // Prepare send buffer
    GUARD_CU(GarenteeAllocation(
      state.send_data, send_allocated_, target_num * sizeof(T)));
    GUARD_CU(GarenteeAllocation(
      send_indices, send_allocated , target_num));
    if (state.max_gradient == NULL) {
      GUARD_CU(Malloc(state.max_gradient, 1));
    }
    GUARD_CU(Memset(send_counter, 0, 1, Malloc_t::Default, stream));
    GUARD_CU(Memset(state.max_gradient, 0, 1, Malloc_t::Default, stream));

    T* send_data = (T*)(state.send_data);
    // Compact gradients
    select_kernel3
      <<<grid_size, block_size, 0, stream>>>
      (elements, config.global_num_gpus,
      thresholds, layer_starts, num_layers, target_num,
      send_data, send_indices, send_counter, state.max_gradient);

    // pad if num_slected < target_num
    pad_kernel
      <<<grid_size, block_size, 0, stream>>>
      ((T*)send_data, send_indices, target_num, send_counter, state.max_gradient);

    // Reallocate if not enough
    SizeT recv_count      = target_num * (config.use_hierarchical_allreduce ?
        config.global_num_nodes : config.global_num_gpus);
    auto &recv_allocated  = state.recv_allocated;
    auto  recv_allocated_ = state.recv_allocated * sizeof(T);
    auto &recv_indices    = state.recv_indices;

    GUARD_CU(GarenteeAllocation(
        state.recv_data, recv_allocated_, recv_count * sizeof(T)));
    GUARD_CU(GarenteeAllocation(
        recv_indices, recv_allocated, recv_count));

    GUARD_CU2("cudaEventRecord",
      cudaEventRecord(token -> stream2_begin, stream));

    T* recv_data = (T*)(state.recv_data);
    // Collect selected data & indices from all peers
    GUARD_NCCL2("ncclAllGather",
      ncclAllGather(send_data   , (void*)recv_data,
        (size_t)target_num, PreDefinedValues<T       >::NCCLDataType,
        config.use_hierarchical_allreduce ? config.nccl_cross_comm :
          config.nccl_comm, stream));
    GUARD_NCCL2("ncclAllGather",
      ncclAllGather(send_indices, (void*)recv_indices,
        (size_t)target_num, PreDefinedValues<uint32_t>::NCCLDataType,
        config.use_hierarchical_allreduce ? config.nccl_cross_comm :
          config.nccl_comm, stream));

    // Unpack received gradients and indices
    GUARD_CU(
      Memset(output_gradients, 0, num_gradients, Malloc_t::Default, stream));
    loop_kernel <<<grid_size, block_size, 0, stream>>>(recv_count,
      [recv_data, recv_indices, output_gradients] __device__ (const SizeT &i) {
        T     element = recv_data   [i];
        SizeT index   = recv_indices[i];
        if (isValid(index))
          atomicAdd(output_gradients + index, element);
      });

    // Updates pervious_verlocity and pervious_accumulated_verlocity
    // Can be overlap with communication
    GUARD_CU2("cudaStreamWaitEvent",
      cudaStreamWaitEvent(stream2, token -> stream2_begin, 0));
    if (config.use_momentum_correction) {
      for (auto& chunk : chunks) {
        SizeT  chunk_start  = std::get<0>(chunk);
        SizeT  chunk_size   = std::get<1>(chunk);
        size_t chunk_offset = std::get<2>(chunk);

        T* pervious_verlocity
          = (T*)(state_pervious_verlocity + chunk_offset);
        T* pervious_accumulated_verlocity
          = (T*)(state.pervious_accumulated_verlocity + chunk_offset);

        loop_kernel <<<grid_size, block_size, 0, stream2>>>(chunk_size,
          [thresholds, chunk_start, chunk_size,
          verlocity, pervious_verlocity,
          accumulated_verlocity, pervious_accumulated_verlocity,
          layer_starts, num_layers]
          __device__ (const SizeT &i) {
            auto gradient_pos = i + chunk_start;
            auto v = accumulated_verlocity[gradient_pos];
            int layer = binarySearch(layer_starts, 0, num_layers, gradient_pos);
            if (isfinite(v * 1.0f) && abs(v) > thresholds[layer]) {
              pervious_verlocity[i] = 0;
              pervious_accumulated_verlocity[i] = 0;
            } else {
              pervious_verlocity[i] = verlocity[gradient_pos];
              pervious_accumulated_verlocity[i] = v;
            }
          });
      }
    }
    else {
      for (auto& chunk : chunks) {
        SizeT   chunk_start  = std::get<0>(chunk);
        SizeT   chunk_size   = std::get<1>(chunk);
        size_t  chunk_offset = std::get<2>(chunk);

        T* pervious_accumulated_gradients
          = (T*)(state.pervious_accumulated_gradients + chunk_offset);
        loop_kernel <<<grid_size, block_size, 0, stream2>>>(chunk_size,
          [thresholds, chunk_start, chunk_size,
          accumulated_gradients, pervious_accumulated_gradients,
          layer_starts, num_layers]
          __device__ (const SizeT &i) {
            auto gradient_pos = i + chunk_start;
            auto g = accumulated_gradients[gradient_pos];
            int layer = binarySearch(layer_starts, 0, num_layers, gradient_pos);
            if (isfinite(g * 1.0f) && abs(g) > thresholds[layer]) {
              pervious_accumulated_gradients[i] = 0;
            } else {
              pervious_accumulated_gradients[i] = g;
            }
          });
      }
    }

    GUARD_CU2("cudaEventRecord",
      cudaEventRecord(token -> stream2_finish, stream2));
    GUARD_CU2("cudaStreamWaitEvent",
      cudaStreamWaitEvent(stream, token -> stream2_finish, 0));
  }

  if (config.learning_rate_decay_factor > 0 &&
      epoch >= config.num_epochs_per_decay) {
    GUARD_CU(LearningRateAdjustment(output_gradients,
      num_gradients, epoch, config, state));
  }

  GUARD_CU2("cudaEventRecord",
    cudaEventRecord(token -> dgc_finish, stream));
  token -> dgc_finished = false;
  state.busy_tokens.push_back(token);

  if (to_overlap_mask) {
    GUARD_CU(TryPushMask(2, config, state));
  }
  return retval;
}

// Entry warper function
cudaError_t GradientAllReduce(
  ncclDataType_t  gradient_type, // type of gradient
  void           *input_gradients, // GPU pointer to the input graients
  void           *output_gradients,// GPU pointer to the output gradients
  std::vector<std::pair<std::string, uint64_t> > &layers,
                                 // <name, #elements> of layers
  DgcConfig      &config,        // DGC configuration
  DgcState       &state)         // DGC running states
{
  typedef uint32_t SizeT;
  cudaError_t retval = cudaSuccess;

  if (config.use_hierarchical_allreduce &&
      !config.cross_comm_inited) {
    ncclUniqueId nccl_cross_id;
    if (config.global_node_rank == 0) {
      GUARD_NCCL2("ncclGetUniqueId",
        ncclGetUniqueId(&nccl_cross_id));
    }

    GUARD_MPI2("MPI_Bcast",
      MPI_Bcast((void*)&nccl_cross_id, sizeof(nccl_cross_id),
        MPI_BYTE, 0, config.cross_comm));

    ncclComm_t new_nccl_comm;
    GUARD_NCCL2("ncclCommInitRank",
      ncclCommInitRank(&new_nccl_comm, config.global_num_nodes,
        nccl_cross_id, config.global_node_rank));
    config.nccl_cross_comm = new_nccl_comm;

    ncclUniqueId nccl_local_id;
    if (config.local_gpu_rank == 0) {
      GUARD_NCCL2("ncclGetUniqueId",
        ncclGetUniqueId(&nccl_local_id));
    }

    GUARD_MPI2("MPI_Bcast",
      MPI_Bcast((void*)&nccl_local_id, sizeof(nccl_local_id),
        MPI_BYTE, 0, config.local_comm));

    ncclComm_t new_nccl_comm2;
    GUARD_NCCL2("ncclCommInitRank",
      ncclCommInitRank(&new_nccl_comm2, config.local_num_gpus,
        nccl_local_id, config.local_gpu_rank));
    config.nccl_local_comm = new_nccl_comm2;

    GUARD_MPI2("MPI_Barrier",
      MPI_Barrier(config.mpi_comm));
    //printf("local = %d of %d, cross = %d of %d, global = %d of %d\n",
    //    config.local_gpu_rank, config.local_num_gpus,
    //    config.global_node_rank, config.global_num_nodes,
    //    config.global_gpu_rank, config.global_num_gpus);
    config.cross_comm_inited = true;
  }

  size_t num_gradients = 0;
  if (config.use_hierarchical_allreduce) {
    for (auto& layer : layers)
      num_gradients += layer.second;

    GUARD_NCCL2("ncclReduce",
      ncclReduce(input_gradients, input_gradients, num_gradients,
        gradient_type, ncclSum, 0, config.nccl_local_comm, config.stream));
  }

  if ((config.use_hierarchical_allreduce && config.local_gpu_rank == 0) ||
      !config.use_hierarchical_allreduce) {
    switch (gradient_type) {
    case ncclFloat32:
      retval = GradientAllReduce <float, SizeT> (
        (float*)input_gradients, (float*)output_gradients,
        layers, config, state);
      break;

    case ncclFloat64:
      retval = GradientAllReduce<double, SizeT> (
        (double*)input_gradients, (double*)output_gradients,
        layers, config, state);
      break;

    case ncclInt32:
      retval = GradientAllReduce<int32_t, SizeT> (
        (int32_t*)input_gradients, (int32_t*)output_gradients,
        layers, config, state);
      break;

    case ncclInt64:
      retval = GradientAllReduce<int64_t, SizeT> (
        (int64_t*)input_gradients, (int64_t*)output_gradients,
        layers, config, state);
      break;

    default:
      break;
    }

    if (retval)
      return retval;
  }

  if (config.use_hierarchical_allreduce) {
    GUARD_NCCL2("ncclBcast",
      ncclBcast(output_gradients, num_gradients,
        gradient_type, 0, config.nccl_local_comm, config.stream));
  }
  return retval;
}

} // end of namespace dgc
} // end of namespace horovod
