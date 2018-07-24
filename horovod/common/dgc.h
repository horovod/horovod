// DGC definations
// by Yuechao Pan
// for NVIDIA

#pragma once

#include <map>
#include <math.h>
#include <nccl.h>
#include <curand_kernel.h>

namespace horovod {
namespace dgc {

template <typename T>
struct PreDefinedValues{};

template <>
struct PreDefinedValues<float>
{
  static const ncclDataType_t NCCLDataType = ncclFloat32;
  constexpr static const float InvalidValue = NAN;
};

template <>
struct PreDefinedValues<double>
{
  static const ncclDataType_t NCCLDataType = ncclFloat64;
  constexpr static const double InvalidValue = NAN;
};

template <>
struct PreDefinedValues<int32_t>
{
  static const ncclDataType_t NCCLDataType = ncclInt32;
  static const int32_t AllZeros = (int32_t)0;
  static const int32_t AllOnes  = ~AllZeros;
  static const int32_t InvalidValue = AllOnes;
};

template <>
struct PreDefinedValues<uint32_t>
{
  static const ncclDataType_t NCCLDataType = ncclUint32;
  static const uint32_t AllZeros = (uint32_t)0;
  static const uint32_t AllOnes  = ~AllZeros;
  static const uint32_t InvalidValue = AllOnes;
};

template <>
struct PreDefinedValues<int64_t>
{
  static const ncclDataType_t NCCLDataType = ncclInt64;
  static const int64_t AllZeros = (int64_t)0;
  static const int64_t AllOnes  = ~AllZeros;
  static const int64_t InvalidValue = AllOnes;
};

template <>
struct PreDefinedValues<uint64_t>
{
  static const ncclDataType_t NCCLDataType = ncclUint64;
  static const uint64_t AllZeros = (uint64_t)0;
  static const uint64_t AllOnes  = ~AllZeros;
  static const uint64_t InvalidValue = AllOnes;
};

template <typename T>
__device__ __host__ __forceinline__
bool isValid(const T &val)
{
    return (val != PreDefinedValues<T>::InvalidValue);
}

template <>
__device__ __host__ __forceinline__
bool isValid(const float &val)
{
    return (!isnan(val));
}

template <>
__device__ __host__ __forceinline__
bool isValid(const double &val)
{
    return (!isnan(val));
}

template <>
__device__ __host__ __forceinline__
bool isValid(const long double &val)
{
    return (!isnan(val));
}

struct DgcConfig {
  // The number of warmup epoches for DGC.
  // DGC communication will use a gradient sparsity, which starts from
  // init_sparsity in the first epoch, and exponentially increases to
  // final_sparsity after warmup epoches.
  double warmup_epochs = 5.0;

  // Each epoch has (num_examples_per_epoch / (global_num_gpus * batch_size_per_gpu)
  // steps
  int num_examples_per_epoch = 1000000;
  int batch_size_per_gpu = 32;

  // Initial gradient sparsity for DGC.
  double init_sparsity = 0.75;

  // Final gradient sparsity for DGC, after the warmup epoches.
  double final_sparsity = 0.999;

  // Sampling rate for top-k selection in DGC
  double sampling_rate = 0.01;

  // dgc rand seed
  unsigned int rand_seed = 2800;

  // dgc grid and block sizes
  int grid_size = 32;
  int block_size = 512;

  // stream DGC works on
  cudaStream_t stream = 0;

  // number of GPUs in all nodes
  int global_num_gpus = 1;

  // NCCL communication handle
  ncclComm_t nccl_comm;

  // whether DgcConfig has been configured
  bool configured = false;

  // the minimum number of elements to trigger sampling
  uint64_t min_sampling_num = 4000;

  // Momentum
  float momentum = 0.9;

  // function to set indivual configuration
  void Set(std::string key, std::string value);
};

struct DgcState {

  // States for curand, one for each GPU thread
  curandState *rand_states     = NULL;

  // Verlocity
  char     *verlocity          = NULL;
  uint64_t  verlocity_allocated = 0;

  // Past verlocity
  char     *pervious_verlocity = NULL;
  uint64_t  pervious_verlocity_allocated = 0;

  // Accumulated verlociy
  char     *accumulated_verlocity = NULL;
  uint64_t  accumulated_verlocity_allocated = 0;

  char     *pervious_accumulated_verlocity = NULL;
  uint64_t  pervious_accumulated_verlocity_allocated = 0;

  // Sample counter
  uint64_t *samp_counter       = NULL;

  // Sample data, raw data in chars; need to convert type before using
  char     *samp_data          = NULL;
  uint64_t  samp_allocated     = 0;

  // Gradient selection threshold
  float    *gradient_threshold = NULL;

  // Counter for gradient selection
  uint64_t *send_counter       = NULL;

  // Number of allocated elements for selected data
  uint64_t  send_allocated     = 0;

  // Memory for selected gradient and indices
  char     *send_data          = NULL;
  uint32_t *send_indices       = NULL;

  // Number of allocated elements for recv data
  uint64_t  recv_allocated     = 0;

  // Memory for recved gradient and indices
  char     *recv_data          = NULL;
  uint32_t *recv_indices       = NULL;

  // Number of allocated elements for global gradients
  uint64_t  global_allocated   = 0;

  // Global gradients
  char     *global_gradients   = NULL;

  // Tensor offset address book
  std::map<std::string, size_t  > tensor_offsets;

  // Per-tensor step counter
  std::map<std::string, uint64_t> step_counters;

  // Step number
  uint64_t step = 0;
  
  // Epoch number
  double epoch = 0;

  // Counter for adding new tensor to the end of memory space
  size_t offset_counter = 0;
};

// Entry warper function
cudaError_t GradientAllReduce(
  ncclDataType_t  element_type, // type of element
  void           *elements,     // GPU pointer to the elements
  uint64_t        num_elements, // number of elements
  std::vector<std::tuple<uint64_t, uint64_t, size_t> >
                 &offset_map,   // <start, length, offset> mappings for
                                // continous chunks of gradients
  DgcConfig      &config,       // DGC configuration
  DgcState       &state);       // DGC running states

} // end of namespace dgc
} // end of namespace horovod
