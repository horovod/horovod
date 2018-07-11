// DGC definations
// by Yuechao Pan
// for NVIDIA

#include <nccl.h>

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

struct DgcConfig {
  // The number of warmup epoches for DGC.
  // DGC communication will use a gradient sparsity, which starts from
  // init_sparsity in the first epoch, and exponentially increases to
  // final_sparsity after warmup epoches.
  int warmup_epoches = 5;

  // Initial gradient sparsity for DGC.
  double init_sparsity = 0.75;

  // Final gradient sparsity for DGC, after the warmup epoches.
  double final_sparsity = 0.999;

  // Sampling rate for top-k selection in DGC
  double sampling_rate = 0.01;

  // dgc rand seed
  unsigned int rand_seed = 2800;

  // dgc grid and block sizes
  int grid_size = 4;
  int block_size = 256;

  // stream DGC works on
  cudaStream_t stream = 0;

  // number of GPUs in all nodes
  int global_num_gpus = 1;

  // NCCL communication handle
  ncclComm_t nccl_comm;
};

struct DgcState {

  // States for curand, one for each GPU thread
  curandState *rand_states = NULL;

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

};

// Entry warper function
template <typename SizeT>
cudaError_t GradientAllReduce(
  ncclDataType_t  element_type, // type of element
  void           *elements,     // GPU pointer to the elements
  SizeT           num_elements, // number of elements
  DgcConfig      &config,       // DGC configuration
  DgcState       &state);       // DGC running states

} // end of namespace dgc
} // end of namespace horovod
