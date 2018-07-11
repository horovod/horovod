// DGC host function implementations
// by Yuechao Pan
// for NVIDIA

// past compile with following command:
// nvcc -std=c++11 -c -o dgc.cu.o horovod_nvidia/horovod/common/dgc.cu.cc      \
   -x cu -Xcompiler -fPIC -dlink --expt-extended-lambda -gencode=arch=compute_70,code=\"sm_70,compute_70\"

#pragma once

#include <string>
#include <algorithm>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "dgc.h"
#include "dgc_kernel.cu.cc"

namespace horovod {
namespace dgc {

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

template <typename T>
cudaError_t Free(
  T* &ptr,
  Malloc_t malloc_type = Malloc_t::Default)
{
  cudaError_t retval = cudaSuccess;
  if (ptr == NULL)
    return retval;

  if (malloc_type == Host) {
    GUARD_CU2("cudaFreeHost",
      cudaFreeHost(ptr));
  } else if (malloc_type == Default || malloc_type == Managed) {
    GUARD_CU2("cudaFree",
      cudaFree(ptr));
  } else if (malloc_type == Raw)
    free(ptr);

  ptr = NULL;
  return retval;
}

template <typename T>
cudaError_t Malloc(
  T* &ptr,
  size_t target,
  Malloc_t malloc_type = Malloc_t::Default,
  unsigned int flags = cudaMemAttachGlobal)
{
  cudaError_t retval = cudaSuccess;

  size_t size = target * sizeof(T);
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

  return retval;
}

template <typename T, typename SizeT>
cudaError_t GarenteeAllocation(
  T*      &ptr,
  SizeT   &allocated,
  size_t   target,
  Malloc_t malloc_type = Malloc_t::Default,
  unsigned int flags = cudaMemAttachGlobal)
{
  cudaError_t retval = cudaSuccess;
  if (allocated >= target)
    return retval;

  auto temp_ptr = ptr;
  GUARD_CU(Free<T> (temp_ptr, malloc_type));
  GUARD_CU(Malloc(ptr, target, malloc_type, flags));
  allocated = target;
  return retval;
}

// ****************************
// Warpper functions
// ****************************

/*template <typename HorovodStateT, typename SizeT>
cudaError_t Sample(
  ncclDataType_t nccl_type,
  HorovodStateT &horovod_global,
  void *buffer_data,
  SizeT num_elements,
  void *sample_data,
  SizeT num_samples,
  curandState *rand_states,
  cudaStream_t stream)
{
  cudaError_t retval = cudaSuccess;
  auto grid_size = horovod_global.grid_size;
  auto bock_size = horovod_global.block_size;

  switch (nccl_type)
  {
  case ncclFloat32:
    sample_kernel <float>
      <<<grid_size, block_size, 0, stream>>>(
        (float*)buffer_data, num_elements,
        (float*)sample_data, num_samples,
        rand_states);
    break;

  case ncclFloat64:
    sample_kernel <double>
      <<<grid_size, block_size, 0, stream>>>(
        (double*)buffer_data, num_elements,
        (double*)sample_data, num_samples,
        rand_states);
    break;

  case ncclInit32:
    sample_kernel <int32_t>
      <<<grid_size, block_size, 0, stream>>>(
        (int32_t*)buffer_data, num_elements,
        (int32_t*)sample_data, num_samples,
        rand_states);
    break;

  case ncclInt64:
    sample_kernel <int64_t>
      <<<grid_size, block_size, 0, stream>>>(
      (int64_t*)buffer_data, num_elements,
      (int64_t*)sample_data, num_samples,
      rand_states);
    break;

  default:
    // Unsupported data type
  }

  return retval;
}

template <typename SizeT>
cudaError_t Threshold(
  ncclDataType_t nccl_type,
  void *elements,
  SizeT num_elements,
  double top_ratio,
  float  *threshold,
  cudaStream_t stream)
{
  cudaError_t retval = cudaSuccess;

  switch (nccl_type)
  {
  case ncclFloat32:
    threshold_kernel <float>
      <<<1, 1, 0, stream>>>(
      (float*)elements, num_elements,
      top_ratio, threshold);

  case ncclFloat64:
    threshold_kernel <double>
      <<<1, 1, 0, stream>>>(
      (double*)elements, num_elements,
      top_ratio, threshold);

  case ncclInt32:
    threshold_kernel <int32_t>
      <<<1, 1, 0, stream>>>(
      (int32_t*)elements, num_elements,
      top_ratio, threshold);

  case ncclFloat64:
    threshold_kernel <int64_t>
      <<<1, 1, 0, stream>>>(
      (int64_t*)elements, num_elements,
      top_ratio, threshold);
  default:
  }

  return retval;
}*/

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

  if (malloc_type == Raw)
  {
    std::sort(elements, elements + num_elements, compare);
    return retval;
  }

  // Use thrust for now;
  // if sort becomes performance bottleneck, change to cub
  thrust::sort(thrust::cuda::par.on(stream),
    elements, elements + num_elements, compare);

  /* Cub sorting
  bool temp_storage_allocated = false;
  if (temp_storage == NULL && temp_storage_bytes == NULL)
  {
    temp_storage = new char*;
    temp_storage[0] = NULL;
    temp_storage_bytes = new size_t;
    temp_storage_bytes[0] = 0;
    temp_storage_allocated = true;
  }

  size_t required_bytes = 0;
  cub::DeviceRadixSort::SortKeys(
    (char*)NULL, required_bytes,
    elements, elements + num_elements,
    num_elements, 0, sizeof(T) * 8, stream);

  retval = GarenteeAllocation(temp_storage[0],
    temp_storage_bytes[0], required_bytes, malloc_type, flags);
  if (retval)
    return retval;

  cub::DeviceRadixSort::SortKeys(
    temp_storage[0], temp_storage_bytes[0],
    elements, elements + num_elements,
    num_elements, 0, sizeof(T) * 8, stream);

  if (temp_storage_allocated)
  {
    retval = Free(temp_storage[0], malloc_type);
    free(temp_storage);
    free(temp_storage_bytes);
    temp_storage = NULL;
    temp_storage_bytes = NULL;
    temp_storage_allocated = false;
  }
  */

  return retval;
}

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
    [] __host__ __device__ (T a, T b){ return a < b;},
    stream, malloc_type, temp_storage, temp_storage_bytes, flags);
}

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

  switch (nccl_type)
  {
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

/*template <typename T, typename SizeT>
cudaError_t Select(
  T         *elements,
  SizeT      num_elements,
  float     *threshold,
  SizeT      target_num,
  T*        &selected_data,
  uint32_t* &selected_indices,
  SizeT     &selected_allocated,
  uint32_t* &selected_counter,
  cudaStream_t stream,
  int        grid_size,
  int        block_size)
{
  cudaError_t retval = cudaSuccess;

  if (selected_counter == NULL)
  {
    retval = Malloc(selected_counter, 1);
    if (retval) return retval;
  }
  assign_kernel
    <<<1, 1, 0, stream>>>(
    selected_counter, 1, (uint32_t)0);

  auto selected_allocated_ = selected_allocated;
  retval = GarenteeAllocation(selected_data   , selected_allocated_, target_num);
  if (retval) return retval;
  retval = GarenteeAllocation(selected_indices, selected_allocated , target_num);
  if (retval) return retval;

  // select at most target_num elements
  select_kernel
    <<<grid_size, block_size, 0, stream>>>
    (elements, num_elements, threshold, target_num,
    selected_data, selected_indices, selected_counter);

  // pad if num_slected < target_num
  pad_kernel
    <<<grid_size, block_size, 0, stream>>>
    (selected_data, selected_indices, target_num, selected_counter);

  return retval;
}

template <typename SizeT>
cudaError_t Select(
  ncclDataType_t nccl_type,
  void      *elements,
  SizeT      num_elements,
  float     *threshold,
  SizeT      target_num,
  void*     &selected_data,
  uint32_t* &selected_indices,
  SizeT     &selected_allocated,
  uint32_t* &selected_counter,
  cudaStream_t stream,
  int        grid_size,
  int        block_size)
{
  cudaError_t retval = cudaSuccess;

  switch (nccl_type)
  {
  case ncclFloat32:
    float* selected_data_f = (float*)selected_data;
    retval = Select<float>(
      (float*)elements, num_elements,
      threshold, target_num,
      selected_data_f, selected_indices,
      selected_allocated, selected_counter,
      stream, grid_size, block_size);
    selected_data = selected_data_f;
    break;

  case ncclFloat64:
    double* selected_data_d = (double*)selected_data;
    retval = Select<double>(
      (double*)elements, num_elements,
      threshold, target_num,
      selected_data_d, selected_indices,
      selected_allocated, selected_counter,
      stream, grid_size, block_size);
    selected_data = selected_data_d;
    break;

  case ncclInt32:
    int32_t* selected_data_i = (int32_t*)selected_data;
    retval = Select<int32_t>(
      (int32_t*)elements, num_elements,
      threshold, target_num,
      selected_data_i, selected_indices,
      selected_allocated, selected_counter,
      stream, grid_size, block_size);
    selected_data = selected_data_i;
    break;

  case ncclInt64:
    int64_t* selected_data_l = (int64_t*)selected_data;
    retval = Select<int64_t>(
      (int64_t*)elements, num_elements,
      threshold, target_num,
      selected_data_l, selected_indices,
      selected_allocated, selected_counter,
      stream, grid_size, block_size);
    selected_data = selected_data_l;
    break;

  default:
  }

  return retval;
}

template <typename T, typename SizeT>
cudaError_t Unpack(
  T        *recv_data,
  uint32_t *recv_indices,
  SizeT     recv_count,
  T*       &global_gradients,
  SizeT    &global_allocated,
  int       grid_size,
  int       block_size,
  cudaStream_t stream)
{
  cudaError_t retval = cudaSuccess;

  retval = GarenteeAllocation(global_gradients, global_allocated, recv_count);
  if (retval)
    return retval;

  assign_kernel
    <<<grid_size, block_size, 0, stream>>>
    (global_gradients, recv_count, (T)0);

  unpack_kernel
    <<<grid_size, block_size, 0, stream>>>
    (recv_data, recv_indices, recv_count,
    global_gradients);

  return retval;
}

template <typename SizeT>
cudaError_t Unpack(
  ncclDataType_t nccl_type,
  void     *recv_data,
  uint32_t *recv_indices,
  SizeT     recv_count,
  void*    &global_gradients,
  SizeT    &global_allocated,
  int       grid_size,
  int       block_size,
  cudaStream_t stream)
{
  cudaError_t retval = cudaSuccess;

  switch (nccl_type)
  {
  case ncclFloat32:
    float* global_gradients_f = (float*)global_gradients;
    retval = Unpack<float>(
      (float*)recv_data, recv_indices, recv_selected,
      global_gradients_f, global_allocated,
      grid_size, block_size, stream);
    global_gradients = global_gradients_f;
    break;

  case ncclFloat64:
    double* global_gradients_d = (double*)global_gradients;
    retval = Unpack<double>(
      (double*)recv_data, recv_indices, recv_selected,
      global_gradients_d, global_allocated,
      grid_size, block_size, stream);
    global_gradients = global_gradients_d;
    break;

  case ncclInt32:
    int32_t* global_gradients_i = (int32_t*)global_gradients;
    retval = Unpack<int32_t>(
      (int32_t*)recv_data, recv_indices, recv_selected,
      global_gradients_i, global_allocated,
      grid_size, block_size, stream);
    global_gradients = global_gradients_i;
    break;

  case ncclInt64:
    int64_t* global_gradients_l = (int64_t*)global_gradients;
    retval = Unpack<int64_t>(
      (int64_t*)recv_data, recv_indices, recv_selected,
      global_gradients_l, global_allocated,
      grid_size, block_size, stream);
    global_gradients = global_gradients_l;
  break;

  default:
  }

  return retval;
}*/

// Entry function
template <typename T, typename SizeT>
cudaError_t GradientAllReduce(
  T              *elements,     // GPU pointer to the elements
  SizeT           num_elements, // number of elements
  DgcConfig      &config,       // DGC configuration
  DgcState       &state)        // DGC running states
{
  cudaError_t retval = cudaSuccess;

  SizeT num_samples  = 0;
  auto  block_size   = config.block_size;
  auto  grid_size    = config.grid_size;
  auto  stream       = config.stream;

  if (config.sampling_rate < 1) {
    auto &rand_states = state.rand_states;
    auto &rand_seed   = config.rand_seed;
    if (rand_states == NULL) {
      GUARD_CU(Malloc(rand_states, grid_size * block_size));

      //rand_init_kernel
      //  <<<grid_size, block_size, 0, stream>>>(
      //  config.rand_seed, state.rand_states);
      loop_kernel<<<grid_size, block_size, 0, stream>>>(
        (SizeT)grid_size * block_size,
        [rand_states, rand_seed] __device__ (const SizeT &i){
          curand_init(rand_seed, i, 0, rand_states + i);
        });
    }

    auto &samp_counter = state.samp_counter;
    // Init counter
    if (samp_counter == NULL) {
      GUARD_CU(Malloc(samp_counter, 1));
    }
    //assign_kernel
    //  <<<1, 1, 0, stream>>>(
    //  samp_counter, 1, (uint64_t)0);
    loop_kernel <<<1, 1, 0, stream>>>((SizeT)1,
      [samp_counter] __device__ (const SizeT &i)
      {
        samp_counter[0] = 0;
      });

    num_samples = num_elements * config.sampling_rate;
    GUARD_CU(GarenteeAllocation(state.samp_data, state.samp_allocated,
      num_samples * sizeof(T)));

    // Sampling
    sample_kernel <T, SizeT>
      <<<grid_size, block_size, 0, stream>>>(
      elements, num_elements,
      (T*)(state.samp_data), num_samples,
      state.rand_states);

    //CUDA_CHECK(entries, "Sample",
    //  dgc::Sample(
    //    element_type, horovod_global,
    //    buffer_data, num_elements,
    //    samp_data, num_samples,
    //    rand_states, stream));
  }

  else { // no sampling
    num_samples = num_elements;
    GUARD_CU(GarenteeAllocation(state.samp_data, state.samp_allocated,
      num_samples * sizeof(T)));

    GUARD_CU2("cudaMemcpyAsync",
      cudaMemcpyAsync(state.samp_data, elements,
        sizeof(T) * num_samples, cudaMemcpyDeviceToDevice, stream));
  }

  // Sort the samples
  GUARD_CU(Sort(state.samp_data, num_samples, stream));

  // Determine the threshold
  double sparsity   = 0; // TODO: calculate the sparsity value
  SizeT  target_num = num_elements * (1 - sparsity);
  auto &threshold = state.gradient_threshold;
  if (threshold == NULL) {
    GUARD_CU(Malloc(threshold, 1));
  }
  loop_kernel<<<1, 1, 0, stream>>>((SizeT)1,
    [threshold, elements, target_num] __device__ (const SizeT &i){
      threshold[0] = elements[target_num];
    });

  //GUARD_CU(entries, "dgc::Threshold",
  //  dgc::Threshold(
  //    element_type, samp_data, num_samples,
  //    (1 - sparsity), threshold));

  // Pick those larger than threshold
  auto &send_counter   = state.send_counter;
  auto &send_data      = state.send_data;
  auto &send_indices   = state.send_indices;
  auto &send_allocated = state.send_allocated;
  auto send_allocated_ = send_allocated * sizeof(T);
  //CUDA_CHECK(entries, "dgc::Select",
  //  dgc::Select(
  //    element_type, buffer_data, num_elements,
  //    threshold, target_num,
  //    send_data, send_indices,
  //    send_allocated, send_counter,
  //    stream, grid_size, block_size));
  if (send_counter == NULL) {
    GUARD_CU(Malloc(send_counter, 1));
  }

  GUARD_CU(GarenteeAllocation(
    send_data   , send_allocated_, target_num * sizeof(T)));
  GUARD_CU(GarenteeAllocation(
    send_indices, send_allocated , target_num));
  loop_kernel <<<1, 1, 0, stream>>>((SizeT)1,
    [send_counter] __device__ (const SizeT &i)
    {
      send_counter[0] = 0;
    });
  //assign_kernel
  //  <<<1, 1, 0, stream>>>(
  //  selected_counter, 1, (uint32_t)0);

  // select at most target_num elements
  select_kernel
    <<<grid_size, block_size, 0, stream>>>
    (elements, num_elements, threshold, target_num,
    (T*)send_data, send_indices, send_counter);

  // pad if num_slected < target_num
  pad_kernel
    <<<grid_size, block_size, 0, stream>>>
    ((T*)send_data, send_indices, target_num, send_counter);

  // Reallocate if not enough
  SizeT recv_count      = target_num * config.global_num_gpus;
  auto &recv_allocated  = state.recv_allocated;
  auto  recv_allocated_ = state.recv_allocated * sizeof(T);
  auto &recv_data       = state.recv_data;
  auto &recv_indices    = state.recv_indices;
  GUARD_CU(GarenteeAllocation(
      recv_data, recv_allocated_, recv_count * sizeof(T)));
  GUARD_CU(GarenteeAllocation(
      recv_indices, recv_allocated, recv_count));

  // Collect selected data & indices from all peers
  GUARD_NCCL2("ncclAllGather",
    ncclAllGather(send_data   , (void*)recv_data,
      (size_t)target_num, PreDefinedValues<T       >::NCCLDataType,
      config.nccl_comm, stream));
  GUARD_NCCL2("ncclAllGather",
    ncclAllGather(send_indices, (void*)recv_indices,
      (size_t)target_num, PreDefinedValues<uint32_t>::NCCLDataType,
      config.nccl_comm, stream));

  auto &global_gradients_= state.global_gradients;
  auto &global_allocated = state.global_allocated;
  //CUDA_CHECK(entries, "dgc::Unpack",
  //  dgc::Unpack(element_type,
  //    recv_data, recv_indices, recv_count,
  //    global_gradients, global_allocated,
  //    grid_size, block_size, stream));
  GUARD_CU(GarenteeAllocation(
    global_gradients_, global_allocated, num_elements * sizeof(T)));

  //assign_kernel
  //  <<<grid_size, block_size, 0, stream>>>
  //  (global_gradients, recv_count, (T)0);
  T* global_gradients = (T*)global_gradients_;
  loop_kernel <<<grid_size, block_size, 0, stream>>>(num_elements,
    [global_gradients] __device__ (const SizeT &i)
    {
      global_gradients[i] = 0;
    });

  //unpack_kernel
  //  <<<grid_size, block_size, 0, stream>>>
  //  (recv_data, recv_indices, recv_count,
  //  global_gradients);
  loop_kernel <<<grid_size, block_size, 0, stream>>>(recv_count,
    [recv_data, recv_indices, global_gradients] __device__ (const SizeT &i)
    {
      T     element = ((T*)recv_data)[i];
      SizeT index   = recv_indices[i];
      atomicAdd(global_gradients + index, element);
    });

  return retval;
}

// Entry warper function
cudaError_t GradientAllReduce(
  ncclDataType_t  element_type, // type of element
  void           *elements,     // GPU pointer to the elements
  uint64_t        num_elements, // number of elements
  DgcConfig      &config,       // DGC configuration
  DgcState       &state)        // DGC running states
{
  typedef uint32_t SizeT;
  cudaError_t retval = cudaSuccess;

  switch (element_type)
  {
  case ncclFloat32:
    retval = GradientAllReduce <float, SizeT> (
      (float*)elements, (SizeT)num_elements, config, state);
    break;

  case ncclFloat64:
    retval = GradientAllReduce<double, SizeT> (
      (double*)elements, (SizeT)num_elements, config, state);
    break;

  case ncclInt32:
    retval = GradientAllReduce<int32_t, SizeT> (
      (int32_t*)elements, (SizeT)num_elements, config, state);
    break;

  case ncclInt64:
    retval = GradientAllReduce<int64_t, SizeT> (
      (int64_t*)elements, (SizeT)num_elements, config, state);
    break;

  default:
    break;
  }
  return retval;
}

} // end of namespace dgc
} // end of namespace horovod
