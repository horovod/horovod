#pragma once

#define xstr(s) cstr(s)
#define cstr(s) #s

#if HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
using gpuError_t = cudaError_t;
using gpuEvent_t = cudaEvent_t;
using gpuStream_t = cudaStream_t;
using gpuPointerAttributes = cudaPointerAttributes;
#define gpuStreamNonBlocking  cudaStreamNonBlocking
#define gpuEventCreateWithFlags cudaEventCreateWithFlags
#define gpuEventCreate cudaEventCreate
#define gpuEventDisableTiming cudaEventDisableTiming
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuStreamWaitEvent cudaStreamWaitEvent
#define gpuSuccess cudaSuccess
#define gpuGetDevice cudaGetDevice
#define gpuSetDevice cudaSetDevice
#define gpuGetErrorString cudaGetErrorString
#define gpuEventQuery cudaEventQuery
#define gpuErrorNotReady cudaErrorNotReady
#define gpuDeviceGetStreamPriorityRange cudaDeviceGetStreamPriorityRange
#define gpuStreamCreateWithPriority cudaStreamCreateWithPriority
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuPointerGetAttributes cudaPointerGetAttributes
#define gpuMalloc cudaMalloc
#define gpuEventDestroy cudaEventDestroy
#define HVD_GPU_CHECK(x)                                                                    \
  do {                                                                                      \
    cudaError_t cuda_result = (x);                                                          \
    if (cuda_result != cudaSuccess) {                                                       \
      throw std::logic_error(std::string("GPU Error:") + cudaGetErrorString(cuda_result));  \
    }                                                                                       \
  } while (0)
#elif HAVE_ROCM
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
using gpuError_t = hipError_t;
using gpuEvent_t = hipEvent_t;
using gpuStream_t = hipStream_t;
using gpuPointerAttributes = hipPointerAttribute_t;
#define gpuStreamNonBlocking  hipStreamNonBlocking
#define gpuEventCreateWithFlags hipEventCreateWithFlags
#define gpuEventCreate hipEventCreate
#define gpuSuccess hipSuccess
#define gpuGetDevice hipGetDevice
#define gpuSetDevice hipSetDevice
#define gpuGetErrorString hipGetErrorString
#define gpuEventDisableTiming hipEventDisableTiming
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuStreamWaitEvent hipStreamWaitEvent
#define gpuEventQuery hipEventQuery
#define gpuErrorNotReady hipErrorNotReady
#define gpuDeviceGetStreamPriorityRange hipDeviceGetStreamPriorityRange
#define gpuStreamCreateWithPriority hipStreamCreateWithPriority
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuPointerGetAttributes hipPointerGetAttributes
#define gpuMalloc hipMalloc
#define gpuEventDestroy hipEventDestroy
#define HVD_GPU_CHECK(x)                                                                  \
  do {                                                                                    \
    hipError_t hip_result = (x);                                                          \
    if (hip_result != hipSuccess) {                                                       \
      throw std::logic_error(std::string("GPU Error:") + hipGetErrorString(hip_result));  \
    }                                                                                     \
  } while (0)
#endif
