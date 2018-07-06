// DGC functions
// by Yuechao Pan
// for NVIDIA

#pragma once

#include <algorithm>
#include <curand_kernel.h>

namespace dgc {

// ****************************
// Kernels
// ****************************

__gloal__
void rand_init_kernel(
  unsigned int  seed,
  curandState  *rand_states)
  {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, rand_states + id);
  } // end of rand_init_kernel

template <typename T, typename SizeT>
__global__
void assign_kernel(
  T     *elements,
  SizeT  num_elements,
  T      val)
  {
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;

    while (i < num_elements)
    {
      elements[i] = val;
      i += STRIDE;
    }
  }

template <typename T, typename SizeT>
__global__
void sample_kernel(
  T           *elements,
  SizeT        num_elements,
  T           *samples,
  SizeT        num_samples,
  curandState *rand_states)
  {
    const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    const SizeT thread_id = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;

    SizeT i = thread_id;
    while (i < num_samples)
    {
      SizeT pos = curand_uniform(rand_states[thread_id]) * num_elements;
      if (pos >= num_elements)
        pos -= num_elements;
      samples[i] = elements[pos];

      i += STRIDE;
    }
  }

template <typename T>
__global__
void threshold_kernel(
  T      *elements,
  SizeT   num_elements,
  double  top_ratio,
  T      *threshold)
  {
    threshold[0] = elements[top_ratio * num_elements];
  }

template <typename T, typename SizeT>
__global__
void select_kernel(
  T      *elements,
  SizeT   num_elements,
  T      *threshold,
  SizeT   target_num,
  T      *selected_elements,
  SizeT  *selected_indices,
  SizeT  *selected_count)
{
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  SizeT block_input_start = (SizeT)blockDim.x * blockIdx.x;
  __shared__ SizeT s_block_output_count, s_block_output_start;

  if (threadId.x == 0)
    s_block_output_count = 0;
  __syncthreads();

  while (block_input_start < num_elements)
  {
    SizeT thread_input  = block_input_start + threadIdx.x;
    SizeT thread_output = 0;
    bool thread_to_select = false;
    T element = 0;
    if (thread_pos < num_elements)
    {
      element = elements[thread_pos];
      if (!(element < threshold))
      {
        thread_to_select = true;
        thread_output = atomicAdd(&s_block_output_count, (SizeT)1);
      }
    }
    __syncthreads();
    // TODO: if double atomicAdd is performance bottleneck,
    //       change to block scan
    if (threadIdx.x == 0 && s_block_output_count != 0)
    {
      s_block_output_start = atomicAdd(selected_count, s_block_output_count);
      s_block_output_count = 0;
    }
    __syncthreads();
    thread_output += s_block_output_start;
    if (thread_to_select && thread_output < target_num)
    {
      selected_elements[thread_output] = element;
      selected_indices [thread_output] = thread_input;
    }

    block_input_start += STRIDE;
  }
}

template <typename T, typename SizeT>
__global__
void pad_kernel(
  T     *selected_elements,
  SizeT *selected_indices,
  SizeT  target_num,
  SizeT *selected_count)
{
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  SizeT i = selected_count[0] + (SizeT)blockIdx.x * blockIdx.x + threadIdx.x;

  while (i < target_num)
  {
    selected_elements[i] = PreDefinedValues<T    >::InvalidValue;
    selected_indices [i] = PreDefinedValues<SizeT>::InvalidValue;
    i += STRIDE;
  }
}

template <typename T, typename SizeT>
__global__
void uppack_kernel(
  T     *recv_elements,
  SizeT *recv_indices,
  SizeT  recv_count,
  T     *global_elements)
{
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  SizeT i = (SizeT)blockIdx.x * blockIdx.x + threadIdx.x;

  while (i < recv_count)
  {
    T     element = recv_elements[i];
    SizeT index   = recv_indices [i];
    atomicAdd(global_elements + index, element);
    i += STRIDE;
  }
}

// ****************************
// Memory management
// ****************************

enum
{
  Default,
  Host,
  Managed,
  Raw,
} Malloc_t;

template <typename T>
cudaError_t Free(
  T* &ptr,
  Malloc_t malloc_type = Malloc_t::Default)
{
  cudaError_t retval = cudaSuccess;
  if (ptr == NULL)
    return retval;

  if (malloc_type == Host)
    retval = cudaFreeHost(ptr);
  else if (malloc_type == Default || malloc_type == Managed)
    retval = cudaFree(ptr);
  else if (malloc_type == Raw)
    free(ptr);

  if (retval)
    return retval;

  ptr = NULL;
  return retval;
}

template <typename T>
cudaError_t Malloc(
  T* &ptr,
  size_t size,
  Malloc_t malloc_type = Malloc_t::Default,
  unsigned int flags = cudaMemAttachGlobal)
{
  cudaError_t retval = cudaSuccess;

  size_t size = target * sizeof(T);
  if (malloc_type == Default)
    retval = cudaMalloc(&ptr, size);
  else if (malloc_type == Host)
    retval = cudaMallocHost(&ptr, size);
  else if (malloc_type == Managed)
    retval = cudaMallocManaged(&ptr, size, flags);
  else if (malloc_type == Raw)
    ptr = malloc(size);

  return retval;
}

template <typename T, typename SizeT>
cudaError_t GarenteeAllocation(
  T*    &ptr,
  SizeT &allocated,
  SizeT  target,
  Malloc_t malloc_type = Malloc_t::Default,
  unsigned int flags = cudaMemAttachGlobal)
{
  cudaError_t retval = cudaSuccess;

  if (allocated >= target)
    return retval;

  retval = Free(ptr, malloc_type);
  if (retval)
    return retval;

  retval = Malloc(ptr, target, malloc_type, flags);
  if (retval)
    return retval;

  allocated = target;
  return retval;
}

// ****************************
// Other functions
// ****************************

template <typename T, typename SizeT, typename Compare>
cudaError_t Sort(
  T           *elements,
  SizeT        num_elements,
  Compare      compare,
  MallocType   malloc_type = MallocType::Default,
  char       **temp_storage = NULL,
  size_t      *temp_storaget_bytes = NULL,
  Malloc_t malloc_type = Malloc_t::Default,
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

template <typename T, typename SizeT, typename Compare>
cudaError_t Sort(
  T      *elements,
  SizeT   num_elements,
  MallocType malloc_type = MallocType::Default)
{
  return Sort(elements, num_elements,
    __host__ __device__ [](T a, T b){ return a < b;},
    malloc_type);
}

} // end of namespace dgc
