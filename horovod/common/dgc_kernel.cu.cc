// DGC device kernel implementations
// by Yuechao Pan
// for NVIDIA

#pragma once

#include "device_intrinsics.cuh"

namespace horovod {
namespace dgc {

__global__
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
    SizeT pos = curand_uniform(rand_states + thread_id) * num_elements;
    if (pos >= num_elements)
      pos -= num_elements;
    samples[i] = elements[pos];

    i += STRIDE;
  }
}

template <typename T, typename SizeT>
__global__
void threshold_kernel(
  T      *elements,
  SizeT   num_elements,
  double  top_ratio,
  float  *threshold)
{
  threshold[0] = elements[top_ratio * num_elements];
}

template <typename T, typename IndexT, typename SizeT, typename CounterT>
__global__
void select_kernel(
  T      *elements,
  SizeT   num_elements,
  float  *threshold_,
  SizeT   target_num,
  T      *selected_elements,
  IndexT *selected_indices,
  CounterT *selected_count)
{
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  SizeT block_input_start = (SizeT)blockDim.x * blockIdx.x;
  __shared__ SizeT s_block_output_count, s_block_output_start;
  float threshold = threshold_[0];

  if (threadIdx.x == 0)
    s_block_output_count = 0;
  __syncthreads();

  while (block_input_start < num_elements)
  {
    SizeT thread_input  = block_input_start + threadIdx.x;
    SizeT thread_output = 0;
    bool thread_to_select = false;
    T element = 0;
    if (thread_input < num_elements)
    {
      element = elements[thread_input];
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
      s_block_output_start = atomicAdd(selected_count, (CounterT)s_block_output_count);
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

template <typename T, typename IndexT, typename SizeT, typename CounterT>
__global__
void pad_kernel(
  T     *selected_elements,
  IndexT *selected_indices,
  SizeT  target_num,
  CounterT *selected_count)
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

template <typename SizeT, typename OpT>
__global__
void loop_kernel(
  SizeT loop_size,
  OpT   op)
{
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  SizeT i = (SizeT)blockIdx.x * blockIdx.x + threadIdx.x;

  while (i < loop_size)
  {
    op(i);
    i += STRIDE;
  }
}

} // end of namespace dgc
} // end of namespace horovod
