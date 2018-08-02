// DGC device kernel implementations
// by Yuechao Pan
// for NVIDIA

#pragma once

#include "device_intrinsics.cuh"

namespace horovod {
namespace dgc {

template <typename T>
__global__
void L2norm_kernel(
  T           *gradients,
  uint32_t    *layer_starts,
  int          num_layers,
  T           *layer_square_sums)
{
  //uint64_t block_start = layer_offsets[num_layers] / gridDim.x * blockIdx.x;
  const uint32_t block_end
    = (blockIdx.x + 1 == gridDim.x) ? layer_starts[num_layers] :
    (layer_starts[num_layers] / gridDim.x * (blockIdx.x + 1));
  //if (blockIdx.x == 0)
  //  block_start = 0;
  //if (blockIdx.x +1 == gridDim.x)
  //  block_end = layer_offsets[num_layers];

  __shared__ uint32_t s_block_current;
  __shared__ T s_sum;
  __shared__ int s_block_layer;

  if (threadIdx.x == 0)
  {
    s_sum = 0;
    s_block_current = layer_starts[num_layers] / gridDim.x * blockIdx.x;
    if (blockIdx.x == 0)
      s_block_current = 0;
    s_block_layer = binarySearch(layer_starts, 0, num_layers, s_block_current);
  }
  __syncthreads();

  while (s_block_current < block_end)
  {
    T t_sum = 0;
    uint32_t layer_end = layer_starts[s_block_layer + 1];
    uint32_t thread_current = s_block_current + threadIdx.x;
    while (thread_current < block_end && thread_current < layer_end)
    {
      auto gradient = gradients[thread_current];
      t_sum += gradient * gradient;
      thread_current += blockDim.x;
    }

    atomicAdd(&s_sum, t_sum);
    __syncthreads();
    if (threadIdx.x == 0)
    {
      atomicAdd(layer_square_sums + s_block_layer, s_sum);
      s_block_layer ++;
      s_sum = 0;
      s_block_current = layer_end;
    }
    __syncthreads();
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
    //if (i < 10)
    //  printf("Selecting elements[%d] = %f for pos %d\n",
    //    pos, elements[pos], i);
    auto element = elements[pos];
    if (!isfinite(element * 1.0f))
      element = 0;
    samples[i] = abs(element);

    i += STRIDE;
  }
}

template <typename T, typename SizeT>
__global__
void sample_kernel2(
  T           *elements,
  SizeT        num_elements,
  SizeT       *layer_starts,
  int          num_layers,
  SizeT       *samp_starts,
  T           *samples,
  curandState *rand_states)
{
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  const SizeT thread_id = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
  const SizeT num_samples = samp_starts[num_layers];

  SizeT i = thread_id;
  while (i < num_samples)
  {
    int layer = binarySearch(samp_starts, 0, num_layers, i);
    if (layer < 0 || layer >= num_layers ||
      samp_starts[layer] > i || samp_starts[layer + 1] <= i)
    {
      printf("#layers = %d, i = %ld, layer = %d, samp_starts = %ld, %ld, %ld\n",
        num_layers, (long)i, layer, layer < 1 ? (long)-1 : (long)samp_starts[layer - 1],
        layer < 0 || layer >= num_layers ? (long)-1 : (long)samp_starts[layer],
        layer + 1 >= num_layers ? (long)-1 : (long)samp_starts[layer + 1]);
      break;
    }
    auto layer_start = layer_starts[layer];
    auto layer_end   = layer_starts[layer + 1];
    auto samp_start  = samp_starts [layer];
    auto samp_end    = samp_starts [layer + 1];
    auto layer_size  = layer_end - layer_start;

    T element;
    if (samp_end - samp_start == layer_end - layer_start)
    {
      element = elements[layer_start + i - samp_start];
    } else {
      SizeT pos = curand_uniform(rand_states + thread_id) * layer_size;
      if (pos >= layer_size)
        pos -= layer_size;
      if (pos + layer_start < 0 || pos + layer_start >= num_elements)
      {
        printf("i = %ld, pos = %ld, layer_start = %ld, layer = %d, layer_size = %ld, "
          "#elements = %ld, layer_end = %ld\n",
          (long)i, (long)pos, (long)layer_start, layer, (long)layer_size,
          (long)num_elements, (long)layer_end);
        break;
      } else 
        element = elements[layer_start + pos];
    }
    if (!isfinite(element * 1.0f))
      element = 0;
    samples[i] = abs(element);
    
    i += STRIDE;
  }
}

template <typename T, typename IndexT, typename SizeT, typename CounterT>
__global__
void select_kernel(
  T      *elements,
  SizeT   num_elements,
  int     global_num_gpus,
  float  *threshold_,
  SizeT   target_num,
  T      *selected_elements,
  IndexT *selected_indices,
  CounterT *selected_count,
  float  *max_gradient)
{
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  SizeT block_input_start = (SizeT)blockDim.x * blockIdx.x;
  __shared__ SizeT s_block_output_count, s_block_output_start;
  float threshold = threshold_[0];
  float t_max_gradient = 0;
  __shared__ float s_max_gradient;

  if (threadIdx.x == 0)
  {
    s_block_output_count = 0;
    s_max_gradient = 0;
    //if (blockIdx.x == 0)
    //  printf("threadhold = %f, #elements = %lld\n", threshold, (long long)num_elements);
  }
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
      if (isfinite(element * 1.0f))
      {
        auto abs_element = abs(element);
        if (!(abs_element < threshold))
        {
          thread_to_select = true;
          thread_output = atomicAdd(&s_block_output_count, (SizeT)1);
        }

        if (t_max_gradient < abs_element)
          t_max_gradient = abs_element;
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
      selected_elements[thread_output] = element; // / global_num_gpus;
      selected_indices [thread_output] = thread_input;
    }

    block_input_start += STRIDE;
  }

  atomicMax(&s_max_gradient, t_max_gradient);
  __syncthreads();

  if (threadIdx.x == 0)
  {
    atomicMax(max_gradient, s_max_gradient);
    //printf("(%d, %d) s_max_gradient = %f\n",
    //  blockIdx.x, threadIdx.x, s_max_gradient);
  }
}

template <typename T, typename IndexT, typename SizeT, typename CounterT>
__global__
void select_kernel2(
  T      *elements,
  SizeT   num_elements,
  int     global_num_gpus,
  float  *threshold_,
  SizeT   target_num,
  T      *selected_elements,
  IndexT *selected_indices,
  CounterT *selected_count)
{
  static const int num_local_slots = 4;
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  __shared__ bool s_to_continue;
  __shared__ int s_block_output_count;
  __shared__ SizeT s_block_output_start;
  const T threshold = threshold_[0];
  T      thread_elements[num_local_slots];
  IndexT thread_indices [num_local_slots];

  if (threadIdx.x == 0)
  {
    s_to_continue = true;
    s_block_output_count = 0;
    if (blockIdx.x == 0)
      printf("threadhold = %f, #elements = %lld\n", threshold,
        (long long)num_elements);
  }
  __syncthreads();

  SizeT thread_pos = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
  int thread_num_output = 0;
  while (s_to_continue)
  {
    while (thread_pos < num_elements &&
           thread_num_output < num_local_slots)
    {
      T element = elements[thread_pos];
      //T element = 0;
      if ((abs(element) > threshold))
      {
        thread_elements[thread_num_output] = element;
        thread_indices [thread_num_output] = thread_pos;
        thread_num_output ++;
      }
      thread_pos += STRIDE;
    }

    int thread_output_start = 0;
    if (thread_num_output != 0)
      atomicAdd(&s_block_output_count, thread_num_output);
    __syncthreads();

    if (threadIdx.x == 0)
    {
      if (s_block_output_count != 0)
      {
        s_block_output_start =
          atomicAdd(selected_count, (CounterT)s_block_output_count);
        s_block_output_count = 0;
        if (s_block_output_start >= target_num)
          s_to_continue = false;
      } else {
        s_to_continue = false;
      }
    }
    __syncthreads();

    IndexT output_pos = s_block_output_start + thread_output_start;
    for (int i = 0; i < thread_num_output; i++)
    {
      if (output_pos >= target_num)
        break;
      selected_elements[output_pos] = thread_elements[i];// / global_num_gpus;
      selected_indices [output_pos] = thread_indices [i];
      output_pos ++;
    }
    thread_num_output = 0;

    //if (thread_pos < num_elements)
    //  s_to_continue = true;
    //__syncthreads();
  }
}

template <typename T, typename IndexT, typename SizeT, typename CounterT>
__global__
void select_kernel3(
  T      *elements,
  //SizeT   num_elements,
  int     global_num_gpus,
  float  *thresholds,
  SizeT  *layer_starts,
  int     num_layers,
  SizeT   target_num,
  T      *selected_elements,
  IndexT *selected_indices,
  CounterT *selected_count,
  float  *max_gradient)
{
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  const SizeT num_elements = layer_starts[num_layers];
  SizeT block_input_start = (SizeT)blockDim.x * blockIdx.x;
  __shared__ SizeT s_block_output_count, s_block_output_start;
  //float threshold = threshold_[0];
  float t_max_gradient = 0;
  __shared__ float s_max_gradient;

  if (threadIdx.x == 0)
  {
    s_block_output_count = 0;
    s_max_gradient = 0;
    //if (blockIdx.x == 0)
    //  printf("threadhold = %f, #elements = %lld\n", threshold, (long long)num_elements);
  }
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
      int layer = binarySearch(layer_starts, 0, num_layers, thread_input);
      float threshold = thresholds[layer];
      if (isfinite(element * 1.0f))
      {
        auto abs_element = abs(element);
        if (!(abs_element < threshold))
        {
          thread_to_select = true;
          thread_output = atomicAdd(&s_block_output_count, (SizeT)1);
        }

        if (t_max_gradient < abs_element)
          t_max_gradient = abs_element;
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
      selected_elements[thread_output] = element;// / global_num_gpus;
      selected_indices [thread_output] = thread_input;
    }

    block_input_start += STRIDE;
  }

  atomicMax(&s_max_gradient, t_max_gradient);
  __syncthreads();

  if (threadIdx.x == 0)
  {
    atomicMax(max_gradient, s_max_gradient);
    //printf("(%d, %d) s_max_gradient = %f\n",
    //  blockIdx.x, threadIdx.x, s_max_gradient);
  }
}

template <typename T, typename IndexT, typename SizeT, typename CounterT>
__global__
void pad_kernel(
  T     *selected_elements,
  IndexT *selected_indices,
  SizeT  target_num,
  CounterT *selected_count,
  float  *max_gradient)
{
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  SizeT i = selected_count[0] + (SizeT)blockDim.x * blockIdx.x + threadIdx.x;

  //if (blockIdx.x == 0 && threadIdx.x == 0)
  //  printf("#selected = %ld, target = %ld, max_gradient = %f\n",
  //    (long long)i, (long long)target_num, max_gradient[0]);

  while (i < target_num)
  {
    selected_elements[i] = PreDefinedValues<T    >::InvalidValue;
    selected_indices [i] = PreDefinedValues<SizeT>::InvalidValue;
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
  SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;

  while (i < loop_size)
  {
    op(i);
    i += STRIDE;
  }
}

} // end of namespace dgc
} // end of namespace horovod
