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
//#include <thrust/sort.h>
//#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <mpi.h>
#include "dgc.h"
#include "dgc_kernel.cu.cc"

namespace horovod {
namespace dgc {

#define GUARD_CU2(op_name, op)                                                 \
{                                                                              \
  do {                                                                         \
    retval = (op);                                                             \
    if (retval != cudaSuccess) {                                               \
      std::string error_message = std::string(__FILE__) + std::string(":")     \
        + std::to_string(__LINE__) + std::string("(")                          \
        + std::string(op_name) + std::string(") failed: ")                     \
        + cudaGetErrorString(retval);                                          \
      fprintf(stderr, "%s\n", error_message.c_str());                          \
      fflush(stderr);                                                          \
      return retval;                                                           \
    }                                                                          \
  } while (false);                                                             \
}

#define GUARD_CU(op)                                                           \
{                                                                              \
  do {                                                                         \
    retval = (op);                                                             \
    if (retval != cudaSuccess) {                                               \
      std::string error_message = std::string(__FILE__) + std::string(":")     \
        + std::to_string(__LINE__) + std::string(" failed: ")                  \
        + cudaGetErrorString(retval);                                          \
      fprintf(stderr, "%s\n", error_message.c_str());                          \
      fflush(stderr);                                                          \
      return retval;                                                           \
    }                                                                          \
  } while (false);                                                             \
}

#define GUARD_NCCL2(op_name, op)                                               \
{                                                                              \
  do {                                                                         \
    auto nccl_result = (op);                                                   \
    if (nccl_result != ncclSuccess) {                                          \
      std::string error_message = std::string(__FILE__) + std::string(":")     \
        + std::to_string(__LINE__) + std::string("(")                          \
        + std::string(op_name) + std::string(") failed: ")                     \
        + ncclGetErrorString(nccl_result);                                     \
      fprintf(stderr, "%s\n", error_message.c_str());                          \
      fflush(stderr);                                                          \
      return cudaErrorUnknown;                                                 \
    }                                                                          \
  } while (false);                                                             \
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

  printf("Freeing @ %p\n", ptr);
  if (malloc_type == Host) {
    GUARD_CU2("cudaFreeHost",
      cudaFreeHost(ptr));
  } else if (malloc_type == Default || malloc_type == Managed) {
    GUARD_CU2("cudaFree",
      cudaFree(ptr));
  } else if (malloc_type == Raw)
    free(ptr);

  printf("Freed @ %p\n", ptr);
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
  printf("Allocating %ld x %ld bytes on %s\n", target, sizeof(T),
     malloc_type == Default ? "Default" :
    (malloc_type == Host    ? "Host" :
    (malloc_type == Managed ? "Managed" : "Raw")));

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

  printf("Allocated %ld x %ld bytes @ %p\n", target, sizeof(T), ptr);
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

void DgcConfig::Set(std::string key, std::string value)
{
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
  {
    if (value == "True")
      local_gradient_clipping = true;
    else if (value == "False")
      local_gradient_clipping = false;
  }

  else if (key == "dgc_clipping_threshold")
    clipping_threshold = std::stof(value);

  else if (key == "dgc_use_allreduce")
  {
    if (value == "True")
      use_allReduce = true;
    else if (value == "False")
      use_allReduce = false;
  }

  else if (key == "dgc_use_hierarchical_allreduce")
  {
    if (value == "True")
      use_hierarchical_allreduce = true;
    else if (value == "False")
      use_hierarchical_allreduce = false;
  }

  else if (key == "momentum")
    momentum = std::stof(value);

  else if (key == "num_examples_per_epoch")
    num_examples_per_epoch = std::stoi(value);

  else if (key == "batch_size")
    batch_size_per_gpu = std::stoi(value);

  //printf("%s = %s\n", key.c_str(), value.c_str());
}

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

  // Not using thrust for now;
  // if sort becomes performance bottleneck, change to cub
  // Note: thrust::sort hit a bug that produced illegal memory access
  //thrust::sort(thrust::cuda::par.on(stream),
  //  elements, elements + num_elements, compare);

  // Cub sorting
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
  GUARD_CU(cub::DeviceRadixSort::SortKeys(
    (char*)NULL, required_bytes,
    elements, elements,
    num_elements, 0, sizeof(T) * 8, stream));

  GUARD_CU(GarenteeAllocation(temp_storage[0],
    temp_storage_bytes[0], required_bytes, malloc_type, flags));
  //GUARD_CU2("cudaDeviceSynchronize",
  //  cudaDeviceSynchronize());

  GUARD_CU(cub::DeviceRadixSort::SortKeys(
    temp_storage[0], temp_storage_bytes[0],
    elements, elements,
    num_elements, 0, sizeof(T) * 8, stream));

  if (temp_storage_allocated)
  {
    GUARD_CU(Free(temp_storage[0], malloc_type));
    free(temp_storage);
    free(temp_storage_bytes);
    temp_storage = NULL;
    temp_storage_bytes = NULL;
    temp_storage_allocated = false;
  }

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

  if (malloc_type == Raw)
  {
    for (int i = 0; i < num_segments; i++)
      std::sort(elements + seg_starts[i], elements + seg_starts[i+1], compare);
    return retval;
  }

  // Not using thrust for now;
  // if sort becomes performance bottleneck, change to cub
  // Note: thrust::sort hit a bug that produced illegal memory access
  //thrust::sort(thrust::cuda::par.on(stream),
  //  elements, elements + num_elements, compare);

  // Cub sorting
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
  GUARD_CU(cub::DeviceSegmentedRadixSort::SortKeys(
    (char*)NULL, required_bytes,
    elements, elements, num_elements,
    num_segments, seg_starts, seg_starts + 1,
    0, sizeof(T) * 8, stream));

  GUARD_CU(GarenteeAllocation(temp_storage[0],
    temp_storage_bytes[0], required_bytes, malloc_type, flags));
  //GUARD_CU2("cudaDeviceSynchronize",
  //  cudaDeviceSynchronize());

  GUARD_CU(cub::DeviceSegmentedRadixSort::SortKeys(
    temp_storage[0], temp_storage_bytes[0],
    elements, elements, num_elements,
    num_segments, seg_starts, seg_starts + 1,
    0, sizeof(T) * 8, stream));

  if (temp_storage_allocated)
  {
    GUARD_CU(Free(temp_storage[0], malloc_type));
    free(temp_storage);
    free(temp_storage_bytes);
    temp_storage = NULL;
    temp_storage_bytes = NULL;
    temp_storage_allocated = false;
  }

  return retval;
}

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

template <typename T>
cudaError_t ClipGradient(
  T          *gradients,
  //uint64_t   *layer_offsets,
  //int         num_layers,
  std::vector<std::pair<std::string, uint64_t> > &layers,
             // <name, #elements> of layers
  DgcConfig  &config,
  DgcState   &state)
{
  cudaError_t retval = cudaSuccess;

  // skip first step, because total number of layers are unknown
  if (state.step == 0)
    return retval;

  int num_layers = layers.size();
  GUARD_CU(GarenteeAllocation(state.temp_storage, state.temp_storage_bytes,
    sizeof(T) * 2 * num_layers + sizeof(uint32_t) * (num_layers + 1)));
  GUARD_CU(GarenteeAllocation(state.h_layer_starts, state.h_layer_starts_allocated,
    num_layers + 1, Malloc_t::Host));
  uint32_t start_counter = 0;
  for (int i = 0; i < num_layers; i++)
  {
    state.h_layer_starts[i] = start_counter;
    start_counter += layers[i].second;
  }
  state.h_layer_starts[num_layers] = start_counter;

  T* sums         = (T*)(state.temp_storage);
  T* coefficients = (T*)(state.temp_storage + sizeof(T) * num_layers);
  uint32_t* layer_starts = (uint32_t*)(state.temp_storage + sizeof(T) * 2 * num_layers);
  auto stream     = config.stream;
  int  grid_size  = config.grid_size;
  int  block_size = config.block_size;
  auto clipping_threshold = config.clipping_threshold;

  GUARD_CU(Memset(sums, 0, num_layers, Malloc_t::Default, stream));
  GUARD_CU2("cudaMemcpyAsync",
    cudaMemcpyAsync(layer_starts, state.h_layer_starts, sizeof(uint32_t) * (num_layers + 1),
      cudaMemcpyHostToDevice, stream));

  // loop_kernel<<<grid_size, block_size, 0, stream>>>(layer_offsets[num_layers],
  //   [offsets, sums, gradients, num_layers] __device__ (const uint64_t &i)
  //   {
  //     int layer = binarySearch(offsets, 0, num_layers, i);
  //     //if (i < offsets[layer] || i >= offsets[layer + 1])
  //     //  printf("offset mismatch: i = %ld, layer = %d, offsets = %ld, %ld, %ld\n",
  //     //      i, layer, layer > 0 ? offsets[layer -1] : -1,
  //     //      offsets[layer], layer < num_layers ? offsets[layer + 1] : -1);
  //
  //     auto gradient = gradients[i];
  //     atomicAdd(sums + layer, gradient * gradient);
  //   });
  L2norm_kernel<<<grid_size, block_size, 0, stream>>>(
    gradients, layer_starts, num_layers, sums);

  int total_num_layers = state.layer_offset_bytes.size();
  uint64_t total_num_gradients = state.offset_byte_counter / sizeof(T);

  loop_kernel<<<grid_size, block_size, 0, stream>>>(num_layers,
    [sums, coefficients, total_num_layers, total_num_gradients,
    clipping_threshold, layer_starts]
    __device__ (const int &layer)
    {
      coefficients[layer] = clipping_threshold /
        // (sqrt(sums[layer] * total_num_gradients / (offsets[layer + 1] - offsets[layer])) + 1e-6);
        // (sqrt(sums[layer]) + 1e-6);
        (sqrt(sums[layer]) * total_num_layers + 1e-6);
        //(sqrt(sums[layer]) * total_num_gradients / (offsets[layer + 1] - offsets[layer]) + 1e-6);
      //printf("Layer %3d: L2 norm = %3.6f, #gradients = %6ld, coef = %3.6f\n",
      //  layer, sqrt(sums[layer]), (long)(offsets[layer+1] - offsets[layer]),
      //  coefficients[layer]);
    });

  loop_kernel<<<grid_size, block_size, 0, stream>>>(start_counter,
    [layer_starts, gradients, coefficients, num_layers] __device__ (const uint32_t &i)
    {
      int layer = binarySearch(layer_starts, 0, num_layers, i);
      auto coefficient = coefficients[layer];
      if (coefficient < 1)
        gradients[i] *= coefficient;
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
  //SizeT num_samples  = 0;
  auto  block_size   = config.block_size;
  auto  grid_size    = config.grid_size;
  auto  stream       = config.stream;
  int   num_layers   = layers.size();
  SizeT num_gradients = 0;

  //GUARD_CU2("cudaStreamSynchronize before",
  //  cudaStreamSynchronize(stream));

  if (config.local_gradient_clipping)
    GUARD_CU(ClipGradient(input_gradients, layers, config, state));
  //GUARD_CU2("cudaStreamSynchronize after clipping",
  //  cudaStreamSynchronize(stream));

  GUARD_CU(GarenteeAllocation(state.h_layer_starts, state.h_layer_starts_allocated,
    num_layers + 1, Malloc_t::Host));
  // find which step is currently in and look for unallocated layers
  std::vector<std::pair<std::string, uint64_t> > layers_to_allocate;
  SizeT num_gradients_to_allocate = 0;
  for (auto &layer : layers)
  {
    auto name = layer.first;
    num_gradients += layer.second;
    // finds step number
    auto counter_it = state.step_counters.find(name);
    if (counter_it == state.step_counters.end())
      state.step_counters[name] = 0;
    else {
      auto step = counter_it -> second;
      counter_it -> second ++;
      if (state.step < step)
        state.step = step;
    }

    auto offset_it = state.layer_offset_bytes.find(name);
    if (offset_it == state.layer_offset_bytes.end()) {
      layers_to_allocate.push_back(layer);
      num_gradients_to_allocate += layer.second;
    }
  } // end of for layers

  // allocate new layers
  if (num_gradients_to_allocate > 0) {
    GUARD_CU(GarenteeAllocation(state.pervious_verlocity,
      state.pervious_verlocity_allocated,
      state.offset_byte_counter + sizeof(T) * num_gradients_to_allocate,
      Malloc_t::Default, cudaMemAttachGlobal, stream, true, true));
    GUARD_CU(GarenteeAllocation(state.pervious_accumulated_verlocity,
      state.pervious_accumulated_verlocity_allocated,
      state.offset_byte_counter + sizeof(T) * num_gradients_to_allocate,
      Malloc_t::Default, cudaMemAttachGlobal, stream, true, true));

    for (auto& layer : layers_to_allocate) {
      state.layer_offset_bytes[layer.first] = state.offset_byte_counter;
      state.offset_byte_counter += layer.second * sizeof(T);
    }
  }

  // find continous layers as chunks
  // <start, size, offset> of chunks
  std::vector<std::tuple<SizeT, SizeT, size_t> > chunks;
  size_t chunk_offset_bytes = state.layer_offset_bytes[layers.begin() -> first];
  SizeT  layer_start = 0;
  SizeT  chunk_start = 0;
  SizeT  chunk_size  = 0;
  for (int i = 0; i < num_layers; i++) {
    auto &layer = layers[i];
    state.h_layer_starts[i] = layer_start;
    if (chunk_offset_bytes + chunk_size * sizeof(T) !=
      state.layer_offset_bytes[layer.first]) {
      // mismatch
      chunks.push_back(std::make_tuple(
        chunk_start, chunk_size, chunk_offset_bytes));
      chunk_size  = 0;
      chunk_start = layer_start;
      chunk_offset_bytes = state.layer_offset_bytes[layer.first];
    }

    chunk_size  += layer.second;
    layer_start += layer.second;
  } // end of for layers
  state.h_layer_starts[num_layers] = layer_start;
  if (chunk_size != 0)
    chunks.push_back(std::make_tuple(
      chunk_start, chunk_size, chunk_offset_bytes));

  auto &layer_starts = state.layer_starts;
  GUARD_CU(GarenteeAllocation(state.layer_starts,
    state.layer_starts_allocated, num_layers + 1));
  GUARD_CU2("cudaMemcpyAsync",
    cudaMemcpyAsync(state.layer_starts, state.h_layer_starts,
      sizeof(uint32_t) * (num_layers + 1), cudaMemcpyHostToDevice, stream));

  // Memory allocation and type conversion
  GUARD_CU(GarenteeAllocation(state.verlocity,
    state.verlocity_allocated, num_gradients * sizeof(T)));
  GUARD_CU(GarenteeAllocation(state.accumulated_verlocity,
    state.accumulated_verlocity_allocated, num_gradients * sizeof(T)));
  T* verlocity = (T*)(state.verlocity);
  T* accumulated_verlocity = (T*)(state.accumulated_verlocity);

  // momentum correction by chunks
  for (auto& chunk : chunks)
  {
    SizeT chunk_start = std::get<0>(chunk);
    SizeT chunk_size  = std::get<1>(chunk);
    size_t chunk_offset = std::get<2>(chunk);

    T* pervious_verlocity
      = (T*)(state.pervious_verlocity + chunk_offset);
    T* pervious_accumulated_verlocity
      = (T*)(state.pervious_accumulated_verlocity + chunk_offset);
    auto &momentum = config.momentum;

    //printf("input_gradients = %p, gradient_chunk = [%ld, %ld), "
    //  "pervious_verlocity = %p, verlocity = %p, "
    //  "pervious_accumulated_verlocity = %p, accumulated_verlocity = %p\n",
    //  input_gradients, gradient_start_chunk,
    //  gradient_start_chunk + num_gradients_chunk,
    //  pervious_verlocity, verlocity,
    //  pervious_accumulated_verlocity, accumulated_verlocity);

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
  //GUARD_CU2("cudaStreamSynchronize after momentum correction",
  //  cudaStreamSynchronize(stream));

  // Sampling
  auto &samp_starts = state.samp_starts;
  GUARD_CU(GarenteeAllocation(state.samp_starts, state.samp_starts_allocated,
    num_layers + 1));
  GUARD_CU(GarenteeAllocation(state.h_samp_starts, state.h_samp_starts_allocated,
    num_layers + 1, Malloc_t::Host));
  uint32_t samp_counter = 0;
  for (int i = 0; i < num_layers; i++)
  {
    auto &layer = layers[i];
    state.h_samp_starts[i] = samp_counter;

    uint32_t num_samples = 0;
    if (config.sampling_rate < 1 &&
        layer.second > config.min_sampling_num) {

      num_samples = layer.second * config.sampling_rate;
      if (num_samples < config.min_sampling_num)
        num_samples = config.min_sampling_num;
      if (num_samples > layer.second)
        num_samples = layer.second;
    }

    else { // no sampling
      num_samples = layer.second;
      //GUARD_CU(GarenteeAllocation(state.samp_data, state.samp_allocated,
      //  num_samples * sizeof(T)));

      //GUARD_CU2("cudaMemcpyAsync",
      //  cudaMemcpyAsync(state.samp_data, gradients,
      //    sizeof(T) * num_samples, cudaMemcpyDeviceToDevice, stream));
      //T* samp_data = (T*)(state.samp_data);
      //loop_kernel<<<grid_size, block_size, 0, stream>>>(num_samples,
      //  [samp_data, accumulated_verlocity] __device__ (const SizeT &i){
      //    samp_data[i] = abs(accumulated_verlocity[i]);
      //  });
    }
    samp_counter += num_samples;
  }
  state.h_samp_starts[num_layers] = samp_counter;
  GUARD_CU2("cudaMemcpyAsync",
    cudaMemcpyAsync(state.samp_starts, state.h_samp_starts,
      sizeof(uint32_t) * (num_layers + 1), cudaMemcpyHostToDevice, stream));

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

  GUARD_CU(GarenteeAllocation(state.samp_data, state.samp_allocated,
    samp_counter * sizeof(T)));
  T* samp_data = (T*)(state.samp_data);

  //sample_kernel <T, SizeT>
  //  <<<grid_size, block_size, 0, stream>>>(
  //  accumulated_verlocity, num_gradients,
  //  samp_data, num_samples,
  //  state.rand_states);
  sample_kernel2<<<grid_size, block_size, 0, stream>>>(
    accumulated_verlocity, num_gradients,
    state.layer_starts, num_layers,
    state.samp_starts, samp_data, state.rand_states);

  //GUARD_CU2("cudaStreamSynchronize after sampling",
  //  cudaStreamSynchronize(stream));
  //GUARD_CU2("cudaDeviceSynchronize before Sort",
  //  cudaDeviceSynchronize());

  // Sort the samples
  //GUARD_CU(Sort(samp_data, num_samples, stream, Malloc_t::Default,
  //  &(state.temp_storage), &(state.temp_storage_bytes)));
  GUARD_CU(SegSort(samp_data, samp_counter, state.samp_starts, num_layers,
    stream, Malloc_t::Default, &(state.temp_storage), &(state.temp_storage_bytes)));
  //GUARD_CU2("cudaDeviceSynchronize after Sort",
  //  cudaDeviceSynchronize());
  //GUARD_CU2("cudaStreamSynchronize after Sort",
  //  cudaStreamSynchronize(stream));

  // Determine the threshold
  uint64_t num_examples_per_step = config.batch_size_per_gpu * config.global_num_gpus;
  uint64_t steps_per_epoch = config.num_examples_per_epoch / num_examples_per_step;
  if (steps_per_epoch * num_examples_per_step < config.num_examples_per_epoch)
    steps_per_epoch ++;
  uint64_t epoch    = state.step * 1.0 / steps_per_epoch;
  double sparsity   = config.final_sparsity;
  if (epoch < config.warmup_epochs) {
    sparsity = config.init_sparsity * exp(
      log(config.final_sparsity / config.init_sparsity)
      / (config.warmup_epochs - 1) * epoch);
    //if (epoch * steps_per_epoch == state.step)
    //  printf("Epoch %ld, Step %ld, sparsity = %lf\n",
    //    epoch, state.step, sparsity);
  }
  SizeT  target_num = num_gradients * (1 - sparsity);
  //auto &threshold = state.gradient_threshold;
  //if (threshold == NULL) {
  //  GUARD_CU(Malloc(threshold, 1));
  //}

  //loop_kernel<<<1, 1, 0, stream>>>((SizeT)1,
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
  GUARD_CU(GarenteeAllocation(thresholds, state.thresholds_allocated, num_layers));

  loop_kernel<<<grid_size, block_size, 0, stream>>>(num_layers,
    [thresholds, samp_data, samp_starts, sparsity]
    __device__ (const int &layer){
      auto samp_start = samp_starts[layer];
      auto samp_end   = samp_starts[layer + 1];
      auto samp_size  = samp_end - samp_start;
      SizeT pos = samp_size * sparsity;
      if (pos >= samp_size)
        pos = samp_size;
      thresholds[layer] = samp_data[samp_start + pos];
    });

  if (config.use_allReduce) {
    // use allReduce on mask to communicate

    SizeT num_masks = num_gradients / 32;
    if (num_masks * 32 < num_gradients)
      num_masks ++;

    auto mask_allocated_ = state.mask_allocated;
    GUARD_CU(GarenteeAllocation(state.send_masks  , mask_allocated_, num_masks));
    mask_allocated_ = state.mask_allocated;
    GUARD_CU(GarenteeAllocation(state.h_send_masks, mask_allocated_, num_masks, Malloc_t::Host));
    mask_allocated_ = state.mask_allocated;
    GUARD_CU(GarenteeAllocation(state.h_recv_masks, mask_allocated_, num_masks, Malloc_t::Host));
    mask_allocated_ = state.mask_allocated;
    GUARD_CU(GarenteeAllocation(state.recv_masks  , mask_allocated_, num_masks));
    if (state.mask_allocated < num_masks)
      state.mask_allocated = num_masks;

    auto &mask_counters = state.mask_counters;
    auto &mask_offsets  = state.mask_offsets;
    GUARD_CU(GarenteeAllocation(
        mask_counters, state.mask_counters_allocated, (num_masks + 1)));
    GUARD_CU(GarenteeAllocation(
        mask_offsets , state.mask_offsets_allocated , (num_masks + 1)));

    size_t required_bytes = 0;
    GUARD_CU(cub::DeviceScan::InclusiveSum(
      (char*)NULL, required_bytes,
      mask_counters, mask_counters, num_masks));
    GUARD_CU(GarenteeAllocation(
      state.temp_storage, state.temp_storage_bytes, required_bytes));

    if (state.h_num_gradients_to_communicate == NULL)
        GUARD_CU(Malloc(state.h_num_gradients_to_communicate, 1, Malloc_t::Host));

    //GUARD_CU2("cudaStreamSynchronize after allocation",
    //  cudaStreamSynchronize(stream));

    auto &send_masks = state.send_masks;
    auto &recv_masks = state.recv_masks;
    loop_kernel<<<grid_size, block_size, 0, stream>>>(num_masks,
      [send_masks, num_gradients, thresholds, layer_starts, num_layers,
      accumulated_verlocity]
      __device__ (const SizeT &i)
      {
        uint32_t mask = 0;
        SizeT offset = i * 32;
        int end_j = 32, j = 0;
        if (offset + end_j > num_gradients)
          end_j = num_gradients - offset;
        while (j < end_j)
        {
          auto pos = j + offset;
          T element = accumulated_verlocity[pos];
          int layer = binarySearch(layer_starts, 0, num_layers, pos);
          if (!isfinite(element * 1.0f))
          {
            j ++;
            continue;
          }

          if (!(abs(element) < thresholds[layer]))
          {
            mask |= (((uint32_t)1) << j);
          }
          j++;
        }
        send_masks[i] = mask;
      });

    GUARD_CU2("cudaMemcpyAsync",
      cudaMemcpyAsync(state.h_send_masks, send_masks, sizeof(uint32_t) * num_masks,
        cudaMemcpyDeviceToHost, stream));
    GUARD_CU2("cudaStreamSynchronize after mask",
      cudaStreamSynchronize(stream));

    MPI_Allreduce(state.h_send_masks, state.h_recv_masks, (int)num_masks,
      PreDefinedValues<uint32_t>::getMpiDataType(), MPI_BOR,
      config.use_hierarchical_allreduce ? config.cross_comm : config.mpi_comm);

    GUARD_CU2("cudaMemcpyAsync",
      cudaMemcpyAsync(recv_masks, state.h_recv_masks, sizeof(uint32_t) * num_masks,
        cudaMemcpyHostToDevice, stream));

    loop_kernel<<<grid_size, block_size, 0, stream>>>(num_masks,
      [recv_masks, mask_counters] __device__ (const SizeT &i)
      {
        mask_counters[i] = __popc(recv_masks[i]);
      });

    GUARD_CU(Memset(mask_offsets, 0, 1, Malloc_t::Default, stream));
    GUARD_CU(cub::DeviceScan::InclusiveSum(
      state.temp_storage, required_bytes,
      mask_counters, mask_offsets + 1, num_masks, stream));

    GUARD_CU2("cudaMemcpyAsync",
      cudaMemcpyAsync(state.h_num_gradients_to_communicate,
        mask_offsets + num_masks, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    GUARD_CU2("cudaStreamSynchronize after InclusiveSum",
      cudaStreamSynchronize(stream));

    auto num_gradients_comm = state.h_num_gradients_to_communicate[0];
    if (config.global_gpu_rank == 0)
      printf("%d #gradients to comm = %ld, #gradients = %ld, rate = %f\n",
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

    T* send_data = (T*)(state.send_data);
    T* recv_data = (T*)(state.recv_data);
    auto global_num_gpus = config.global_num_gpus;
    loop_kernel<<<grid_size, block_size, 0, stream>>>(num_masks,
      [recv_masks, mask_offsets, send_data, global_num_gpus,
      num_gradients, accumulated_verlocity]
      __device__ (const SizeT &i)
      {
        uint32_t mask = recv_masks[i];
        if (mask == 0)
          return;

        SizeT offset = i * 32, output_offset = mask_offsets[i];
        int end_j = 32, j = 0, output_count = 0;
        if (offset + end_j > num_gradients)
          end_j = num_gradients - offset;
        while (j < end_j)
        {
          if ((mask & (((uint32_t)1) << j)) == 0)
          {
            j ++;
            continue;
          }
          T element = accumulated_verlocity[j + offset];
          if (!isfinite(element * 1.0f))
            element = 0;

          send_data[output_offset + output_count] = element; // / global_num_gpus;
          output_count ++;
          j++;
        }
      });

    //GUARD_CU2("cudaStreamSynchronize after send_data forming",
    //  cudaStreamSynchronize(stream));

    GUARD_NCCL2("ncclAllReduce",
      ncclAllReduce(send_data   , (void*)recv_data,
        (size_t)num_gradients_comm,
        PreDefinedValues<T>::NCCLDataType, ncclSum,
        config.use_hierarchical_allreduce ? config.nccl_cross_comm :
          config.nccl_comm, stream));

    GUARD_CU(Memset(output_gradients, 0, num_gradients, Malloc_t::Default, stream));
    loop_kernel<<<grid_size, block_size, 0, stream>>>(num_masks,
      [recv_masks, mask_offsets, recv_data, output_gradients, num_gradients]
      __device__ (const SizeT &i)
      {
        uint32_t mask = recv_masks[i];
        if (mask == 0)
          return;

        SizeT offset = i * 32, output_offset = mask_offsets[i];
        int end_j = 32, j = 0, output_count = 0;
        if (offset + end_j > num_gradients)
          end_j = num_gradients - offset;
        while (j < end_j)
        {
          if ((mask & (((uint32_t)1) << j)) == 0)
          {
            j ++;
            continue;
          }

          output_gradients[j + offset] = recv_data[output_offset + output_count];
          output_count ++;
          j++;
        }
      });

    //GUARD_CU2("cudaStreamSynchronize after output_gradient calculation",
    //  cudaStreamSynchronize(stream));

    // Updates pervious_verlocity and pervious_accumulated_verlocity
    // Can be overlap with communication
    for (auto& chunk : chunks)
    {
      SizeT  chunk_start  = std::get<0>(chunk);
      SizeT  chunk_size   = std::get<1>(chunk);
      size_t chunk_offset = std::get<2>(chunk);

      T* pervious_verlocity
        = (T*)(state.pervious_verlocity + chunk_offset);
      T* pervious_accumulated_verlocity
        = (T*)(state.pervious_accumulated_verlocity + chunk_offset);

      loop_kernel <<<grid_size, block_size, 0, stream>>>(chunk_size,
        [recv_masks, chunk_start, chunk_size,
         verlocity, pervious_verlocity,
         accumulated_verlocity, pervious_accumulated_verlocity]
        __device__ (const SizeT &i)
        {
          //if (i == 0)
          //  printf("gradient [%ld...%ld) \n",
          //    (long)gradient_start_chunk,
          //    (long)(gradient_start_chunk + num_gradients_chunk));
          auto gradient_pos = i + chunk_start;
          auto mask_pos = gradient_pos / 32;
          auto mask = recv_masks[mask_pos];
          auto mask_offset = (gradient_pos & ((uint32_t)31));

          if ((mask & (((uint32_t)1) << mask_offset)) != 0)
          {
            pervious_verlocity[i] = 0;
            pervious_accumulated_verlocity[i] = 0;
          } else {
            pervious_verlocity[i] = verlocity[gradient_pos];
            pervious_accumulated_verlocity[i] = accumulated_verlocity[gradient_pos];
          }
        });
    }
  } // end of if (use_allReduce)

  else {
    // use allGather to communicate
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
    if (state.max_gradient == NULL) {
      GUARD_CU(Malloc(state.max_gradient, 1));
    }
    //loop_kernel <<<1, 1, 0, stream>>>((SizeT)1,
    //  [send_counter] __device__ (const SizeT &i)
    //  {
    //    send_counter[0] = 0;
    //  });
    GUARD_CU(Memset(send_counter, 0, 1, Malloc_t::Default, stream));
    GUARD_CU(Memset(state.max_gradient, 0, 1, Malloc_t::Default, stream));

    T* send_data = (T*)(state.send_data);
    // select at most target_num gradients
    //select_kernel
    //  <<<grid_size, block_size, 0, stream>>>
    //  (accumulated_verlocity, num_gradients, config.global_num_gpus,
    //  threshold, target_num, send_data, send_indices, send_counter,
    //  state.max_gradient);
    select_kernel3
      <<<grid_size, block_size, 0, stream>>>
      (accumulated_verlocity, config.global_num_gpus,
      thresholds, layer_starts, num_layers, target_num,
      send_data, send_indices, send_counter, state.max_gradient);

    // pad if num_slected < target_num
    pad_kernel
      <<<grid_size, block_size, 0, stream>>>
      ((T*)send_data, send_indices, target_num, send_counter, state.max_gradient);

    // Reallocate if not enough
    SizeT recv_count      = target_num * config.global_num_gpus;
    auto &recv_allocated  = state.recv_allocated;
    auto  recv_allocated_ = state.recv_allocated * sizeof(T);
    //auto &recv_data       = state.recv_data;
    auto &recv_indices    = state.recv_indices;

    //printf("recv_count = %lld\n", (long long)recv_count);
    GUARD_CU(GarenteeAllocation(
        state.recv_data, recv_allocated_, recv_count * sizeof(T)));
    GUARD_CU(GarenteeAllocation(
        recv_indices, recv_allocated, recv_count));

    //GUARD_CU2("cudaStreamSynchronize after send data forming",
    //    cudaStreamSynchronize(stream));

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
    //GUARD_CU2("cudaStreamSynchronize after AllGather",
    //    cudaStreamSynchronize(stream));

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
    GUARD_CU(Memset(output_gradients, 0, num_gradients, Malloc_t::Default, stream));

    // Unpack recv data
    loop_kernel <<<grid_size, block_size, 0, stream>>>(recv_count,
      [recv_data, recv_indices, output_gradients] __device__ (const SizeT &i)
      {
        T     element = recv_data   [i];
        SizeT index   = recv_indices[i];
        if (isValid(index))
          atomicAdd(output_gradients + index, element);
      });

    // Updates pervious_verlocity and pervious_accumulated_verlocity
    // Can be overlap with communication
    for (auto &chunk : chunks)
    {
      SizeT  chunk_start  = std::get<0>(chunk);
      SizeT  chunk_size   = std::get<1>(chunk);
      size_t chunk_offset = std::get<2>(chunk);

      T* pervious_verlocity
        = (T*)(state.pervious_verlocity + chunk_offset);
      T* pervious_accumulated_verlocity
        = (T*)(state.pervious_accumulated_verlocity + chunk_offset);

      loop_kernel <<<grid_size, block_size, 0, stream>>>(chunk_size,
        [thresholds, chunk_start, chunk_size,
        verlocity, pervious_verlocity,
        accumulated_verlocity, pervious_accumulated_verlocity,
        layer_starts, num_layers]
        __device__ (const SizeT &i)
        {
          //if (i == 0)
          //  printf("gradient [%ld...%ld) \n",
          //    (long)gradient_start_chunk,
          //    (long)(gradient_start_chunk + num_gradients_chunk));
          auto gradient_pos = i + chunk_start;
          auto v = accumulated_verlocity[gradient_pos];
          int layer = binarySearch(layer_starts, 0, num_layers, gradient_pos);
          if (isfinite(v * 1.0f) && abs(v) > thresholds[layer])
          {
            pervious_verlocity[i] = 0;
            pervious_accumulated_verlocity[i] = 0;
          } else {
            pervious_verlocity[i] = verlocity[gradient_pos];
            pervious_accumulated_verlocity[i] = v;
          }
        });
    }
  }

  //GUARD_CU2("cudaStreamSynchronize after",
  //  cudaStreamSynchronize(stream));

  return retval;
}

// Entry warper function
cudaError_t ClipGradient(
  ncclDataType_t  gradient_type, // type of gradient
  void           *gradients,     // GPU pointer to the gradients
  //uint64_t       *layer_offsets, // gradient layer offsets, on host
  //int             num_layers,    // The number of layers in the gradients
  std::vector<std::pair<std::string, uint64_t> > &layers,
                                // <name, #elements> of layers
  DgcConfig      &config,        // DGC configuration
  DgcState       &state)         // DGC running states
{
  typedef uint32_t SizeT;
  cudaError_t retval = cudaSuccess;

  switch (gradient_type)
  {
  case ncclFloat32:
    retval = ClipGradient <float> (
      //(float*)gradients, layer_offsets, num_layers, config, state);
      (float*)gradients, layers, config, state);
    break;

  case ncclFloat64:
    retval = ClipGradient <double> (
      //(double*)gradients, layer_offsets, num_layers, config, state);
      (double*)gradients, layers, config, state);
    break;

  case ncclInt32:
    retval = ClipGradient <int32_t> (
      //(int32_t*)gradients, layer_offsets, num_layers, config, state);
      (int32_t*)gradients, layers, config, state);
    break;

  case ncclInt64:
    retval = ClipGradient <int64_t> (
      //(int64_t*)gradients, layer_offsets, num_layers, config, state);
      (int64_t*)gradients, layers, config, state);
    break;

  default:
    break;
  }
  return retval;
}

cudaError_t GradientAllReduce(
  ncclDataType_t  gradient_type, // type of gradient
  void           *input_gradients, // GPU pointer to the input graients
  void           *output_gradients,// GPU pointer to the output gradients
  //uint64_t        num_gradients, // number of gradients
  //std::vector<std::tuple<uint64_t, uint64_t, size_t> >
  //               &offset_map,    // <start, length, offset> mappings for
                                 // continous chunks of gradients
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

    MPI_Bcast((void*)&nccl_cross_id, sizeof(nccl_cross_id), MPI_BYTE, 0,
      config.cross_comm);

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

    MPI_Bcast((void*)&nccl_local_id, sizeof(nccl_local_id), MPI_BYTE, 0,
      config.local_comm);

    ncclComm_t new_nccl_comm2;
    GUARD_NCCL2("ncclCommInitRank",
      ncclCommInitRank(&new_nccl_comm2, config.local_num_gpus,
        nccl_local_id, config.local_gpu_rank));
    config.nccl_local_comm = new_nccl_comm2;

    MPI_Barrier(config.mpi_comm);
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
    switch (gradient_type)
    {
    case ncclFloat32:
      retval = GradientAllReduce <float, SizeT> (
        (float*)input_gradients, (float*)output_gradients,
        //(SizeT)num_gradients, offset_map, config, state);
        layers, config, state);
      break;

    case ncclFloat64:
      retval = GradientAllReduce<double, SizeT> (
        (double*)input_gradients, (double*)output_gradients,
        //(SizeT)num_gradients, offset_map, config, state);
        layers, config, state);
      break;

    case ncclInt32:
      retval = GradientAllReduce<int32_t, SizeT> (
        (int32_t*)input_gradients, (int32_t*)output_gradients,
        //(SizeT)num_gradients, offset_map, config, state);
        layers, config, state);
      break;

    case ncclInt64:
      retval = GradientAllReduce<int64_t, SizeT> (
        (int64_t*)input_gradients, (int64_t*)output_gradients,
        //(SizeT)num_gradients, offset_map, config, state);
        layers, config, state);
      break;

    default:
      break;
    }
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
