// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef HOROVOD_TORCH_TENSOR_UTIL_H
#define HOROVOD_TORCH_TENSOR_UTIL_H

#include <TH/TH.h>
#include <cassert>

#if HAVE_CUDA
#include <THC/THC.h>
#endif

#include "../common/common.h"
#include "cuda_util.h"

#if HAVE_CUDA
extern THCState* state;
#endif

namespace horovod {
namespace torch {

using namespace horovod::common;

// TH<xxx>Tensor are all aliased to THTensor as of PyTorch 0.4.1, so we need
// an additional template parameter to distinguish between them.
class TensorUtil {
public:
  template <class T, MPIDataType DT>
  static const TensorShape GetShape(T* tensor);
  template <class T, MPIDataType DT> static const void* GetData(T* tensor);
  template <class T, MPIDataType DT> static int64_t GetSize(T* tensor);
  template <class T, MPIDataType DT> static int GetDevice(T* tensor);

  template <class T, MPIDataType DT> static T* New(int device);
  template <class T, MPIDataType DT> static void Free(T* tensor);
  template <class T, MPIDataType DT>
  static void ResizeNd(T* tensor, int nDimension, int64_t* size,
                       int64_t* stride);
  template <class T, MPIDataType DT> static void Copy(T* output, T* tensor);
  template <class T, MPIDataType DT>
  static void DivideTensorInPlace(T* tensor, int value);

#if HAVE_CUDA
  template <class T, class TC, MPIDataType DT>
  static void CopyCPUToCuda(T* cpu, TC* cuda);
  template <class TC, class T, MPIDataType DT>
  static void AsyncCopyCudaToCPU(TC* cuda, T* cpu);
#endif
};

#define TENSOR_UTIL_DEFINE_TYPE_H(THTensor, HorovodType)                       \
  template <>                                                                  \
  const TensorShape TensorUtil::GetShape<THTensor, HorovodType>(THTensor *     \
                                                                tensor);       \
  template <>                                                                  \
  const void* TensorUtil::GetData<THTensor, HorovodType>(THTensor * tensor);   \
  template <>                                                                  \
  int64_t TensorUtil::GetSize<THTensor, HorovodType>(THTensor * tensor);       \
  template <>                                                                  \
  int TensorUtil::GetDevice<THTensor, HorovodType>(THTensor * tensor);         \
                                                                               \
  template <> THTensor* TensorUtil::New<THTensor, HorovodType>(int device);    \
  template <> void TensorUtil::Free<THTensor, HorovodType>(THTensor * tensor); \
  template <>                                                                  \
  void TensorUtil::ResizeNd<THTensor, HorovodType>(                            \
      THTensor * tensor, int nDimension, int64_t* size, int64_t* stride);      \
  template <>                                                                  \
  void TensorUtil::Copy<THTensor, HorovodType>(THTensor * output,              \
                                               THTensor * tensor);             \
  template <>                                                                  \
  void TensorUtil::DivideTensorInPlace<THTensor, HorovodType>(                 \
      THTensor * tensor, int value);

#define TENSOR_UTIL_DEFINE_CUDA_TYPE_H(THCTensor, THTensor, HorovodType)       \
  TENSOR_UTIL_DEFINE_TYPE_H(THCTensor, HorovodType)                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::CopyCPUToCuda<THTensor, THCTensor, HorovodType>(            \
      THTensor * cpu, THCTensor * cuda);                                       \
  template <>                                                                  \
  void TensorUtil::AsyncCopyCudaToCPU<THCTensor, THTensor, HorovodType>(       \
      THCTensor * cuda, THTensor * cpu);

#define TENSOR_UTIL_DEFINE_TYPE(THTensor, THStorage, HorovodType)              \
  template <>                                                                  \
  const TensorShape TensorUtil::GetShape<THTensor, HorovodType>(THTensor *     \
                                                                tensor) {      \
    TensorShape shape;                                                         \
    for (int idx = 0; idx < THTensor##_nDimension(tensor); idx++) {            \
      shape.AddDim(THTensor##_size(tensor, idx));                              \
    }                                                                          \
    return shape;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  const void* TensorUtil::GetData<THTensor, HorovodType>(THTensor * tensor) {  \
    return THTensor##_data(tensor);                                            \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  int64_t TensorUtil::GetSize<THTensor, HorovodType>(THTensor * tensor) {      \
    return (int64_t)(THStorage##_size(THTensor##_storage(tensor)) *            \
                     THStorage##_elementSize());                               \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  int TensorUtil::GetDevice<THTensor, HorovodType>(THTensor * tensor) {        \
    return CPU_DEVICE_ID;                                                      \
  }                                                                            \
                                                                               \
  template <> THTensor* TensorUtil::New<THTensor, HorovodType>(int device) {   \
    assert(device == CPU_DEVICE_ID);                                           \
    return THTensor##_new();                                                   \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::Free<THTensor, HorovodType>(THTensor * tensor) {            \
    THTensor##_free(tensor);                                                   \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::ResizeNd<THTensor, HorovodType>(                            \
      THTensor * tensor, int nDimension, int64_t* size, int64_t* stride) {     \
    THTensor##_resizeNd(tensor, nDimension, size, stride);                     \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::Copy<THTensor, HorovodType>(THTensor * output,              \
                                               THTensor * tensor) {            \
    THTensor##_copy(output, tensor);                                           \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::DivideTensorInPlace<THTensor, HorovodType>(                 \
      THTensor * tensor, int value) {                                          \
    THTensor##_div(tensor, tensor, value);                                     \
  }

#define TENSOR_UTIL_DEFINE_CUDA_TYPE(THCTensor, THTensor, THCStorage,          \
                                     HorovodType)                              \
  template <>                                                                  \
  const TensorShape TensorUtil::GetShape<THCTensor, HorovodType>(THCTensor *   \
                                                                 tensor) {     \
    TensorShape shape;                                                         \
    for (int idx = 0; idx < THCTensor##_nDimension(state, tensor); idx++) {    \
      shape.AddDim(THCTensor##_size(state, tensor, idx));                      \
    }                                                                          \
    return shape;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  const void* TensorUtil::GetData<THCTensor, HorovodType>(THCTensor *          \
                                                          tensor) {            \
    return THCTensor##_data(state, tensor);                                    \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  int64_t TensorUtil::GetSize<THCTensor, HorovodType>(THCTensor * tensor) {    \
    return (int64_t)(                                                          \
        THCStorage##_size(state, THCTensor##_storage(state, tensor)) *         \
        THCStorage##_elementSize(state));                                      \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  int TensorUtil::GetDevice<THCTensor, HorovodType>(THCTensor * tensor) {      \
    return THCTensor##_getDevice(state, tensor);                               \
  }                                                                            \
                                                                               \
  template <> THCTensor* TensorUtil::New<THCTensor, HorovodType>(int device) { \
    with_device device_context(device);                                        \
    return THCTensor##_new(state);                                             \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::Free<THCTensor, HorovodType>(THCTensor * tensor) {          \
    THCTensor##_free(state, tensor);                                           \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::ResizeNd<THCTensor, HorovodType>(                           \
      THCTensor * tensor, int nDimension, int64_t* size, int64_t* stride) {    \
    with_device device_context(THCTensor##_getDevice(state, tensor));          \
    THCTensor##_resizeNd(state, tensor, nDimension, size, stride);             \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::Copy<THCTensor, HorovodType>(THCTensor * output,            \
                                                THCTensor * tensor) {          \
    with_device device_context(THCTensor##_getDevice(state, output));          \
    THCTensor##_copy(state, output, tensor);                                   \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::DivideTensorInPlace<THCTensor, HorovodType>(                \
      THCTensor * tensor, int value) {                                         \
    with_device device_context(THCTensor##_getDevice(state, tensor));          \
    THCTensor##_div(state, tensor, tensor, value);                             \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::CopyCPUToCuda<THTensor, THCTensor, HorovodType>(            \
      THTensor * cpu, THCTensor * cuda) {                                      \
    with_device device_context(THCTensor##_getDevice(state, cuda));            \
    THLongStorage* size = THTensor##_newSizeOf(cpu);                           \
    if (!THCTensor##_isSize(state, cuda, size)) {                              \
      THCTensor##_resize(state, cuda, size, NULL);                             \
    }                                                                          \
    THLongStorage_free(size);                                                  \
    THCTensor##_copyCPU(state, cuda, cpu);                                     \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::AsyncCopyCudaToCPU<THCTensor, THTensor, HorovodType>(       \
      THCTensor * cuda, THTensor * cpu) {                                      \
    with_device device_context(THCTensor##_getDevice(state, cuda));            \
    THLongStorage* size = THCTensor##_newSizeOf(state, cuda);                  \
    if (!THTensor##_isSize(cpu, size)) {                                       \
      THTensor##_resize(cpu, size, NULL);                                      \
    }                                                                          \
    THLongStorage_free(size);                                                  \
    THTensor##_copyAsyncCuda(state, cpu, cuda);                                \
  }

TENSOR_UTIL_DEFINE_TYPE_H(THByteTensor, MPIDataType::HOROVOD_UINT8)
TENSOR_UTIL_DEFINE_TYPE_H(THCharTensor, MPIDataType::HOROVOD_INT8)
TENSOR_UTIL_DEFINE_TYPE_H(THShortTensor, MPIDataType::HOROVOD_INT16)
TENSOR_UTIL_DEFINE_TYPE_H(THIntTensor, MPIDataType::HOROVOD_INT32)
TENSOR_UTIL_DEFINE_TYPE_H(THLongTensor, MPIDataType::HOROVOD_INT64)
TENSOR_UTIL_DEFINE_TYPE_H(THFloatTensor, MPIDataType::HOROVOD_FLOAT32)
TENSOR_UTIL_DEFINE_TYPE_H(THDoubleTensor, MPIDataType::HOROVOD_FLOAT64)

#if HAVE_CUDA
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(THCudaByteTensor, THByteTensor,
                               MPIDataType::HOROVOD_UINT8)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(THCudaCharTensor, THCharTensor,
                               MPIDataType::HOROVOD_INT8)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(THCudaShortTensor, THShortTensor,
                               MPIDataType::HOROVOD_INT16)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(THCudaIntTensor, THIntTensor,
                               MPIDataType::HOROVOD_INT32)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(THCudaLongTensor, THLongTensor,
                               MPIDataType::HOROVOD_INT64)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(THCudaTensor, THFloatTensor,
                               MPIDataType::HOROVOD_FLOAT32)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(THCudaDoubleTensor, THDoubleTensor,
                               MPIDataType::HOROVOD_FLOAT64)
#endif

} // namespace torch
} // namespace horovod

#endif // HOROVOD_TORCH_TENSOR_UTIL_H
