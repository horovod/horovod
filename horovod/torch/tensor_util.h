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

class TensorUtil {
public:
  template <class T> static const MPIDataType GetDType();
  template <class T> static const TensorShape GetShape(T* tensor);
  template <class T> static const void* GetData(T* tensor);
  template <class T> static int64_t GetSize(T* tensor);
  template <class T> static int GetDevice(T* tensor);

  template <class T> static T* New(int device);
  template <class T> static void Free(T* tensor);
  template <class T>
  static void ResizeNd(T* tensor, int nDimension, int64_t* size,
                       int64_t* stride);
  template <class T> static void Copy(T* output, T* tensor);
  template <class T> static void DivideTensorInPlace(T* tensor, int value);

#if HAVE_CUDA
  template <class T, class TC> static void CopyCPUToCuda(T* cpu, TC* cuda);
  template <class TC, class T> static void AsyncCopyCudaToCPU(TC* cuda, T* cpu);
#endif
};

#define TENSOR_UTIL_DEFINE_TYPE_H(THTensor)                                    \
  template <> const MPIDataType TensorUtil::GetDType<THTensor>();              \
  template <>                                                                  \
  const TensorShape TensorUtil::GetShape<THTensor>(THTensor * tensor);         \
  template <> const void* TensorUtil::GetData<THTensor>(THTensor * tensor);    \
  template <> int64_t TensorUtil::GetSize<THTensor>(THTensor * tensor);        \
  template <> int TensorUtil::GetDevice<THTensor>(THTensor * tensor);          \
                                                                               \
  template <> THTensor* TensorUtil::New<THTensor>(int device);                 \
  template <> void TensorUtil::Free<THTensor>(THTensor * tensor);              \
  template <>                                                                  \
  void TensorUtil::ResizeNd<THTensor>(THTensor * tensor, int nDimension,       \
                                      int64_t* size, int64_t* stride);         \
  template <>                                                                  \
  void TensorUtil::Copy<THTensor>(THTensor * output, THTensor * tensor);       \
  template <>                                                                  \
  void TensorUtil::DivideTensorInPlace<THTensor>(THTensor * tensor,            \
                                                 int value);

#define TENSOR_UTIL_DEFINE_CUDA_TYPE_H(THCTensor, THTensor)                    \
  TENSOR_UTIL_DEFINE_TYPE_H(THCTensor)                                         \
                                                                               \
  template <>                                                                  \
  void TensorUtil::CopyCPUToCuda<THTensor, THCTensor>(THTensor * cpu,          \
                                                      THCTensor * cuda);       \
  template <>                                                                  \
  void TensorUtil::AsyncCopyCudaToCPU<THCTensor, THTensor>(THCTensor * cuda,   \
                                                           THTensor * cpu);

#define TENSOR_UTIL_DEFINE_TYPE(THTensor, THStorage, HorovodType)              \
  template <> const MPIDataType TensorUtil::GetDType<THTensor>() {             \
    return HorovodType;                                                        \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  const TensorShape TensorUtil::GetShape<THTensor>(THTensor * tensor) {        \
    TensorShape shape;                                                         \
    for (int idx = 0; idx < THTensor##_nDimension(tensor); idx++) {            \
      shape.AddDim(THTensor##_size(tensor, idx));                              \
    }                                                                          \
    return shape;                                                              \
  }                                                                            \
                                                                               \
  template <> const void* TensorUtil::GetData<THTensor>(THTensor * tensor) {   \
    return THTensor##_data(tensor);                               \
  }                                                                            \
                                                                               \
  template <> int64_t TensorUtil::GetSize<THTensor>(THTensor * tensor) {       \
    return (int64_t)(THStorage##_size(tensor->storage) *                       \
                     THStorage##_elementSize());                               \
  }                                                                            \
                                                                               \
  template <> int TensorUtil::GetDevice<THTensor>(THTensor * tensor) {         \
    return CPU_DEVICE_ID;                                                      \
  }                                                                            \
                                                                               \
  template <> THTensor* TensorUtil::New<THTensor>(int device) {                \
    assert(device == CPU_DEVICE_ID);                                           \
    return THTensor##_new();                                                   \
  }                                                                            \
                                                                               \
  template <> void TensorUtil::Free<THTensor>(THTensor * tensor) {             \
    THTensor##_free(tensor);                                                   \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::ResizeNd<THTensor>(THTensor * tensor, int nDimension,       \
                                      int64_t* size, int64_t* stride) {        \
    THTensor##_resizeNd(tensor, nDimension, size, stride);                     \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::Copy<THTensor>(THTensor * output, THTensor * tensor) {      \
    THTensor##_copy(output, tensor);                                           \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::DivideTensorInPlace<THTensor>(THTensor * tensor,            \
                                                 int value) {                  \
    THTensor##_div(tensor, tensor, value);                                     \
  }

#define TENSOR_UTIL_DEFINE_CUDA_TYPE(THCTensor, THTensor, THCStorage,          \
                                     HorovodType)                              \
  template <> const MPIDataType TensorUtil::GetDType<THCTensor>() {            \
    return HorovodType;                                                        \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  const TensorShape TensorUtil::GetShape<THCTensor>(THCTensor * tensor) {      \
    TensorShape shape;                                                         \
    for (int idx = 0; idx < THCTensor##_nDimension(state, tensor); idx++) {    \
      shape.AddDim(THCTensor##_size(state, tensor, idx));                      \
    }                                                                          \
    return shape;                                                              \
  }                                                                            \
                                                                               \
  template <> const void* TensorUtil::GetData<THCTensor>(THCTensor * tensor) { \
    return THCTensor##_data(state, tensor);                       \
  }                                                                            \
                                                                               \
  template <> int64_t TensorUtil::GetSize<THCTensor>(THCTensor * tensor) {     \
    return (int64_t)(THCStorage##_size(state, tensor->storage) *               \
                     THCStorage##_elementSize(state));                         \
  }                                                                            \
                                                                               \
  template <> int TensorUtil::GetDevice<THCTensor>(THCTensor * tensor) {       \
    return THCTensor##_getDevice(state, tensor);                               \
  }                                                                            \
                                                                               \
  template <> THCTensor* TensorUtil::New<THCTensor>(int device) {              \
    with_device device_context(device);                                        \
    return THCTensor##_new(state);                                             \
  }                                                                            \
                                                                               \
  template <> void TensorUtil::Free<THCTensor>(THCTensor * tensor) {           \
    THCTensor##_free(state, tensor);                                           \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::ResizeNd<THCTensor>(THCTensor * tensor, int nDimension,     \
                                       int64_t* size, int64_t* stride) {       \
    with_device device_context(THCTensor##_getDevice(state, tensor));          \
    THCTensor##_resizeNd(state, tensor, nDimension, size, stride);             \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::Copy<THCTensor>(THCTensor * output, THCTensor * tensor) {   \
    with_device device_context(THCTensor##_getDevice(state, output));          \
    THCTensor##_copy(state, output, tensor);                                   \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::DivideTensorInPlace<THCTensor>(THCTensor * tensor,          \
                                                  int value) {                 \
    with_device device_context(THCTensor##_getDevice(state, tensor));          \
    THCTensor##_div(state, tensor, tensor, value);                             \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::CopyCPUToCuda<THTensor, THCTensor>(THTensor * cpu,          \
                                                      THCTensor * cuda) {      \
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
  void TensorUtil::AsyncCopyCudaToCPU<THCTensor, THTensor>(THCTensor * cuda,   \
                                                           THTensor * cpu) {   \
    with_device device_context(THCTensor##_getDevice(state, cuda));            \
    THLongStorage* size = THCTensor##_newSizeOf(state, cuda);                  \
    if (!THTensor##_isSize(cpu, size)) {                                       \
      THTensor##_resize(cpu, size, NULL);                                      \
    }                                                                          \
    THLongStorage_free(size);                                                  \
    THTensor##_copyAsyncCuda(state, cpu, cuda);                                \
  }

TENSOR_UTIL_DEFINE_TYPE_H(THByteTensor)
TENSOR_UTIL_DEFINE_TYPE_H(THCharTensor)
TENSOR_UTIL_DEFINE_TYPE_H(THShortTensor)
TENSOR_UTIL_DEFINE_TYPE_H(THIntTensor)
TENSOR_UTIL_DEFINE_TYPE_H(THLongTensor)
TENSOR_UTIL_DEFINE_TYPE_H(THFloatTensor)
TENSOR_UTIL_DEFINE_TYPE_H(THDoubleTensor)

#if HAVE_CUDA
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(THCudaByteTensor, THByteTensor)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(THCudaCharTensor, THCharTensor)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(THCudaShortTensor, THShortTensor)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(THCudaIntTensor, THIntTensor)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(THCudaLongTensor, THLongTensor)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(THCudaTensor, THFloatTensor)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(THCudaDoubleTensor, THDoubleTensor)
#endif

} // namespace torch
} // namespace horovod

#endif // HOROVOD_TORCH_TENSOR_UTIL_H
