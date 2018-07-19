// DGC host function implementations
// by Yuechao Pan
// for NVIDIA

// past compile with following command:
// nvcc -std=c++11 -c -o dgc.cu.o horovod_nvidia/horovod/common/dgc.cu.cc      \
   -x cu -Xcompiler -fPIC -dlink --expt-extended-lambda -gencode=arch=compute_70,code=\"sm_70,compute_70\"

//#pragma once

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
  unsigned int flags = cudaMemAttachGlobal,
  cudaStream_t stream = 0)
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

  if (malloc_type != Raw)
  {
    if (stream == 0)
    {
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

  if (malloc_type != Malloc_t::Raw)
  {
    if (stream == 0)
    {
      retval = cudaMemset(ptr, value, num_elements * sizeof(T));
    } else {
      retval = cudaMemsetAsync(ptr, value, num_elements * sizeof(T), stream);
    }
  } else {
    memset(ptr, value, num_elements * sizeof(T));
  }

  return retval;
}

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

  //if (stream != 0)
  //{
  //  GUARD_CU2("cudaStreamSynchronize",
  //    cudaStreamSynchronize(stream));
  //}
  if (!keep_content)
  {
    auto temp_ptr = ptr;
    GUARD_CU(Free<T> (temp_ptr, malloc_type));
    GUARD_CU(Malloc(ptr, target, malloc_type, flags));
    if (init_to_zero)
    {
      GUARD_CU(Memset(ptr, 0, target, malloc_type, stream));
    }
  } else {
    T* temp_ptr = NULL;
    GUARD_CU(Malloc(temp_ptr, target, malloc_type, flags));
    GUARD_CU(Memcpy(temp_ptr, ptr, allocated, malloc_type, stream));
    if (init_to_zero)
    {
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

// ****************************
// DGC Functions
// ****************************

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

// Main DGC routine
template <typename T, typename SizeT>
cudaError_t GradientAllReduce(
  T              *gradients,     // GPU pointer to the gradients
  SizeT           num_gradients, // number of gradients
  std::vector<std::tuple<uint64_t, uint64_t, size_t> >
                 &offset_map,    // <start, length, offset> mappings for
                                 // continous chunks of gradients
  DgcConfig      &config,        // DGC configuration
  DgcState       &state)         // DGC running states
{
  cudaError_t retval = cudaSuccess;

  SizeT num_samples  = 0;
  auto  block_size   = config.block_size;
  auto  grid_size    = config.grid_size;
  auto  stream       = config.stream;

  // Memory allocation and type conversion
  size_t current_size = num_gradients * sizeof(T);
  GUARD_CU(GarenteeAllocation(state.verlocity,
    state.verlocity_allocated, current_size));
  GUARD_CU(GarenteeAllocation(state.accumulated_verlocity,
    state.accumulated_verlocity_allocated, current_size));
  T* verlocity = (T*)(state.verlocity);
  T* accumulated_verlocity = (T*)(state.accumulated_verlocity);

  size_t max_size = 0;
  for (auto it = offset_map.begin(); it != offset_map.end(); it++)
  {
    size_t size = std::get<1>(*it) * sizeof(T) + std::get<2>(*it);
    if (max_size < size)
      max_size = size;
  }
  GUARD_CU(GarenteeAllocation(state.pervious_verlocity,
    state.pervious_verlocity_allocated, max_size,
    Malloc_t::Default, cudaMemAttachGlobal, stream, true, true));
  GUARD_CU(GarenteeAllocation(state.pervious_accumulated_verlocity,
    state.pervious_accumulated_verlocity_allocated, max_size,
    Malloc_t::Default, cudaMemAttachGlobal, stream, true, true));

  // Process by chunks
  for (auto it = offset_map.begin(); it != offset_map.end(); it++)
  {
    SizeT gradient_start_chunk = std::get<0>(*it);
    SizeT num_gradients_chunk  = std::get<1>(*it);
    size_t offset              = std::get<2>(*it);

    T* pervious_verlocity
      = (T*)(state.pervious_verlocity + offset);
    T* pervious_accumulated_verlocity
      = (T*)(state.pervious_accumulated_verlocity + offset);
    auto &momentum = config.momentum;

    loop_kernel<<<grid_size, block_size, 0, stream>>>(num_gradients_chunk,
      [momentum, gradients, gradient_start_chunk,
      pervious_verlocity, verlocity,
      accumulated_verlocity, pervious_accumulated_verlocity]
      __device__ (const SizeT &i) {
        auto u = pervious_verlocity[i] * momentum + gradients[i + gradient_start_chunk];
        accumulated_verlocity[i] = pervious_accumulated_verlocity[i] + u;
        verlocity[i + gradient_start_chunk] = u;
      });
  }

  if (config.sampling_rate < 1 &&
      num_gradients > config.min_sampling_num) {
    auto &rand_states = state.rand_states;
    auto &rand_seed   = config.rand_seed;
    if (rand_states == NULL) {
      GUARD_CU(Malloc(rand_states, grid_size * block_size));

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
    //loop_kernel <<<1, 1, 0, stream>>>((SizeT)1,
    //  [samp_counter] __device__ (const SizeT &i)
    //  {
    //    samp_counter[0] = 0;
    //  });
    GUARD_CU(Memset(samp_counter, 0, 1, Malloc_t::Default, stream));

    num_samples = num_gradients * config.sampling_rate;
    if (num_samples < config.min_sampling_num)
      num_samples = config.min_sampling_num;
    //if (num_samples > num_gradients)
    //  num_samples = num_gradients;
    //printf("#elments = %ld, #samples = %ld\n",
    //  (long long)num_gradients, (long long)num_samples);
    GUARD_CU(GarenteeAllocation(state.samp_data, state.samp_allocated,
      num_samples * sizeof(T)));

    GUARD_CU(cudaDeviceSynchronize());
    // Sampling
    sample_kernel <T, SizeT>
      <<<grid_size, block_size, 0, stream>>>(
      accumulated_verlocity, num_gradients,
      (T*)(state.samp_data), num_samples,
      state.rand_states);
  }

  else { // no sampling
    num_samples = num_gradients;
    GUARD_CU(GarenteeAllocation(state.samp_data, state.samp_allocated,
      num_samples * sizeof(T)));

    //GUARD_CU2("cudaMemcpyAsync",
    //  cudaMemcpyAsync(state.samp_data, gradients,
    //    sizeof(T) * num_samples, cudaMemcpyDeviceToDevice, stream));
    T* samp_data = (T*)(state.samp_data);
    loop_kernel<<<grid_size, block_size, 0, stream>>>(num_samples,
      [samp_data, accumulated_verlocity] __device__ (const SizeT &i){
        samp_data[i] = abs(accumulated_verlocity[i]);
      });
  }

  T* samp_data = (T*)(state.samp_data);
  //loop_kernel<<<grid_size, block_size, 0, stream>>>((SizeT)10,
  //  [samp_data, num_samples] __device__ (const SizeT &i)
  //  {
  //    printf("before Sort samp[%d] = %f\n", i,
  //      (i < num_samples) ? samp_data[i] : 0.12345678);
  //  });

  // Sort the samples
  GUARD_CU(Sort(samp_data, num_samples, stream));

  // Determine the threshold
  double sparsity   = config.final_sparsity; // TODO: calculate the sparsity value
  SizeT  target_num = num_gradients * (1 - sparsity);
  auto &threshold = state.gradient_threshold;
  if (threshold == NULL) {
    GUARD_CU(Malloc(threshold, 1));
  }

  loop_kernel<<<1, 1, 0, stream>>>((SizeT)1,
    [threshold, samp_data, num_samples, sparsity] __device__ (const SizeT &i){
      SizeT pos = num_samples * sparsity;
      if (pos >= num_samples)
        pos = num_samples - 1;
      threshold[0] = samp_data[pos];
      //printf("selecting samp[%d] from [%d] {%f, %f, ... %f, %f, %f, ... %f, %f}\n",
      //  pos, num_samples,
      //  num_samples > 0 ? samp_data[0] : -1,
      //  num_samples > 1 ? samp_data[1] : -1,
      //  num_samples + 1 > pos  && pos > 0 ? samp_data[pos - 1] : -1,
      //  num_samples > pos && pos >= 0 ? samp_data[pos] : -1,
      //  num_samples > pos + 1 && pos + 1 >= 0 ? samp_data[pos + 1] : -1,
      //  num_samples > 1 ? samp_data[num_samples - 2] : -1,
      //  num_samples > 0 ? samp_data[num_samples - 1] : -1);
    });

  //auto &samp_counter = state.samp_counter;
  //loop_kernel <<<1, 1, 0, stream>>>((SizeT)1,
  //  [samp_counter] __device__ (const SizeT &i)
  //  {
  //    samp_counter[0] = 0;
  //  });
  //loop_kernel <<<grid_size, block_size, 0, stream>>>(num_samples,
  //  [samp_data, num_samples, samp_counter, threshold] __device__ (const SizeT &i)
  //  {
  //    if (!(samp_data[i] < threshold[0]))
  //    {
  //      atomicAdd(samp_counter, (uint64_t)1);
  //    }
  //  });
  //loop_kernel <<<1, 1, 0, stream>>>((SizeT)1,
  //  [samp_counter] __device__ (const SizeT &i)
  //  {
  //    printf("Recount = %d\n", samp_counter[0]);
  //  });

  // Pick those larger than threshold
  auto &send_counter   = state.send_counter;
  //auto &send_data      = state.send_data;
  auto &send_indices   = state.send_indices;
  auto &send_allocated = state.send_allocated;
  auto send_allocated_ = send_allocated * sizeof(T);
  if (send_counter == NULL) {
    GUARD_CU(Malloc(send_counter, 1));
  }

  GUARD_CU(GarenteeAllocation(
    state.send_data, send_allocated_, target_num * sizeof(T)));
  GUARD_CU(GarenteeAllocation(
    send_indices, send_allocated , target_num));
  //loop_kernel <<<1, 1, 0, stream>>>((SizeT)1,
  //  [send_counter] __device__ (const SizeT &i)
  //  {
  //    send_counter[0] = 0;
  //  });
  GUARD_CU(Memset(send_counter, 0, 1, Malloc_t::Default, stream));

  T* send_data = (T*)(state.send_data);
  // select at most target_num gradients
  select_kernel
    <<<grid_size, block_size, 0, stream>>>
    (accumulated_verlocity, num_gradients, config.global_num_gpus,
    threshold, target_num, send_data, send_indices, send_counter);

  // pad if num_slected < target_num
  pad_kernel
    <<<grid_size, block_size, 0, stream>>>
    ((T*)send_data, send_indices, target_num, send_counter);

  // Reallocate if not enough
  SizeT recv_count      = target_num * config.global_num_gpus;
  auto &recv_allocated  = state.recv_allocated;
  auto  recv_allocated_ = state.recv_allocated * sizeof(T);
  //auto &recv_data       = state.recv_data;
  auto &recv_indices    = state.recv_indices;

  printf("recv_count = %lld\n", (long long)recv_count);
  GUARD_CU(GarenteeAllocation(
      state.recv_data, recv_allocated_, recv_count * sizeof(T)));
  GUARD_CU(GarenteeAllocation(
      recv_indices, recv_allocated, recv_count));

  T* recv_data = (T*)(state.recv_data);
  // Collect selected data & indices from all peers
  GUARD_NCCL2("ncclAllGather",
    ncclAllGather(send_data   , (void*)recv_data,
      (size_t)target_num, PreDefinedValues<T       >::NCCLDataType,
      config.nccl_comm, stream));
  GUARD_NCCL2("ncclAllGather",
    ncclAllGather(send_indices, (void*)recv_indices,
      (size_t)target_num, PreDefinedValues<uint32_t>::NCCLDataType,
      config.nccl_comm, stream));

  //auto &global_gradients_= state.global_gradients;
  //auto &global_allocated = state.global_allocated;
  //GUARD_CU(GarenteeAllocation(
  //  state.global_gradients, global_allocated, num_gradients * sizeof(T)));
  //T* global_gradients = (T*)(state.global_gradients);

  // Post process gradients
  //loop_kernel <<<grid_size, block_size, 0, stream>>>(num_gradients,
  //  [global_gradients] __device__ (const SizeT &i)
  //  {
  //    global_gradients[i] = 0;
  //  });
  GUARD_CU(Memset(gradients, 0, num_gradients, Malloc_t::Default, stream));

  // Unpack recv data
  loop_kernel <<<grid_size, block_size, 0, stream>>>(recv_count,
    [recv_data, recv_indices, gradients] __device__ (const SizeT &i)
    {
      T     element = recv_data   [i];
      SizeT index   = recv_indices[i];
      if (isValid(index))
        atomicAdd(gradients + index, element);
    });

  // Updates pervious_verlocity and pervious_accumulated_verlocity
  // Can be overlap with communication
  for (auto it = offset_map.begin(); it != offset_map.end(); it++)
  {
    SizeT gradient_start_chunk = std::get<0>(*it);
    SizeT num_gradients_chunk  = std::get<1>(*it);
    size_t offset              = std::get<2>(*it);

    T* pervious_verlocity
      = (T*)(state.pervious_verlocity + offset);
    T* pervious_accumulated_verlocity
      = (T*)(state.pervious_accumulated_verlocity + offset);

    loop_kernel <<<grid_size, block_size, 0, stream>>>(num_gradients_chunk,
      [threshold, gradient_start_chunk, verlocity, pervious_verlocity,
       accumulated_verlocity, pervious_accumulated_verlocity]
      __device__ (const SizeT &i)
      {
        auto v = accumulated_verlocity[i + gradient_start_chunk];
        if (abs(v) > threshold[0])
        {
          pervious_verlocity[i] = 0;
          pervious_accumulated_verlocity[i] = 0;
        } else {
          pervious_verlocity[i] = verlocity[i];
          pervious_accumulated_verlocity[i] = v;
        }
      });
  }
  return retval;
}

// Entry warper function
cudaError_t GradientAllReduce(
  ncclDataType_t  gradient_type, // type of gradient
  void           *gradients,     // GPU pointer to the gradients
  uint64_t        num_gradients, // number of gradients
  std::vector<std::tuple<uint64_t, uint64_t, size_t> >
                 &offset_map,    // <start, length, offset> mappings for
                                 // continous chunks of gradients
  DgcConfig      &config,        // DGC configuration
  DgcState       &state)         // DGC running states
{
  typedef uint32_t SizeT;
  cudaError_t retval = cudaSuccess;

  switch (gradient_type)
  {
  case ncclFloat32:
    retval = GradientAllReduce <float, SizeT> (
      (float*)gradients, (SizeT)num_gradients, offset_map, config, state);
    break;

  case ncclFloat64:
    retval = GradientAllReduce<double, SizeT> (
      (double*)gradients, (SizeT)num_gradients, offset_map, config, state);
    break;

  case ncclInt32:
    retval = GradientAllReduce<int32_t, SizeT> (
      (int32_t*)gradients, (SizeT)num_gradients, offset_map, config, state);
    break;

  case ncclInt64:
    retval = GradientAllReduce<int64_t, SizeT> (
      (int64_t*)gradients, (SizeT)num_gradients, offset_map, config, state);
    break;

  default:
    break;
  }
  return retval;
}

} // end of namespace dgc
} // end of namespace horovod
