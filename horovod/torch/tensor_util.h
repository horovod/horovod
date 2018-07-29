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
  template <MPIDataType DT, DeviceType Dev, class T>
  static const TensorShape GetShape(T* tensor);
  template <MPIDataType DT, DeviceType Dev, class T>
  static const void* GetData(T* tensor);
  template <MPIDataType DT, DeviceType Dev, class T>
  static int64_t GetSize(T* tensor);
  template <MPIDataType DT, DeviceType Dev, class T>
  static int GetDevice(T* tensor);

  template <MPIDataType DT, DeviceType Dev, class T> static T* New(int device);
  template <MPIDataType DT, DeviceType Dev, class T>
  static void Free(T* tensor);
  template <MPIDataType DT, DeviceType Dev, class T>
  static void ResizeNd(T* tensor, int nDimension, int64_t* size,
                       int64_t* stride);
  template <MPIDataType DT, DeviceType Dev, class T>
  static void Copy(T* output, T* tensor);
  template <MPIDataType DT, DeviceType Dev, class T>
  static void DivideTensorInPlace(T* tensor, int value);

#if HAVE_CUDA
  template <MPIDataType DT, class T, class TC>
  static void CopyCPUToCuda(T* cpu, TC* cuda);
  template <MPIDataType DT, class TC, class T>
  static void AsyncCopyCudaToCPU(TC* cuda, T* cpu);
#endif
};

#define TENSOR_UTIL_DEFINE_TYPE_H(HorovodType, DeviceType, THTensor)           \
  template <>                                                                  \
  const TensorShape TensorUtil::GetShape<HorovodType, DeviceType, THTensor>(   \
      THTensor * tensor);                                                      \
  template <>                                                                  \
  const void* TensorUtil::GetData<HorovodType, DeviceType, THTensor>(          \
      THTensor * tensor);                                                      \
  template <>                                                                  \
  int64_t TensorUtil::GetSize<HorovodType, DeviceType, THTensor>(THTensor *    \
                                                                 tensor);      \
  template <>                                                                  \
  int TensorUtil::GetDevice<HorovodType, DeviceType, THTensor>(THTensor *      \
                                                               tensor);        \
                                                                               \
  template <>                                                                  \
  THTensor* TensorUtil::New<HorovodType, DeviceType, THTensor>(int device);    \
  template <>                                                                  \
  void TensorUtil::Free<HorovodType, DeviceType, THTensor>(THTensor * tensor); \
  template <>                                                                  \
  void TensorUtil::ResizeNd<HorovodType, DeviceType, THTensor>(                \
      THTensor * tensor, int nDimension, int64_t* size, int64_t* stride);      \
  template <>                                                                  \
  void TensorUtil::Copy<HorovodType, DeviceType, THTensor>(THTensor * output,  \
                                                           THTensor * tensor); \
  template <>                                                                  \
  void TensorUtil::DivideTensorInPlace<HorovodType, DeviceType, THTensor>(     \
      THTensor * tensor, int value);

#define TENSOR_UTIL_DEFINE_CPU_TYPE_H(HorovodType, THTensor)                   \
  TENSOR_UTIL_DEFINE_TYPE_H(HorovodType, DeviceType::CPU, THTensor)

#define TENSOR_UTIL_DEFINE_CUDA_TYPE_H(HorovodType, THCTensor, THTensor)       \
  TENSOR_UTIL_DEFINE_TYPE_H(HorovodType, DeviceType::GPU, THCTensor)           \
                                                                               \
  template <>                                                                  \
  void TensorUtil::CopyCPUToCuda<HorovodType, THTensor, THCTensor>(            \
      THTensor * cpu, THCTensor * cuda);                                       \
  template <>                                                                  \
  void TensorUtil::AsyncCopyCudaToCPU<HorovodType, THCTensor, THTensor>(       \
      THCTensor * cuda, THTensor * cpu);

#define TENSOR_UTIL_DEFINE_CPU_TYPE(HorovodType, THTensor, THStorage)          \
  template <>                                                                  \
  const TensorShape TensorUtil::GetShape<HorovodType, DeviceType::CPU,         \
                                         THTensor>(THTensor * tensor) {        \
    TensorShape shape;                                                         \
    for (int idx = 0; idx < THTensor##_nDimension(tensor); idx++) {            \
      shape.AddDim(THTensor##_size(tensor, idx));                              \
    }                                                                          \
    return shape;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  const void* TensorUtil::GetData<HorovodType, DeviceType::CPU, THTensor>(     \
      THTensor * tensor) {                                                     \
    return THTensor##_data(tensor);                                            \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  int64_t TensorUtil::GetSize<HorovodType, DeviceType::CPU, THTensor>(         \
      THTensor * tensor) {                                                     \
    return (int64_t)(THStorage##_size(THTensor##_storage(tensor)) *            \
                     THStorage##_elementSize());                               \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  int TensorUtil::GetDevice<HorovodType, DeviceType::CPU, THTensor>(THTensor * \
                                                                    tensor) {  \
    return CPU_DEVICE_ID;                                                      \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  THTensor* TensorUtil::New<HorovodType, DeviceType::CPU, THTensor>(           \
      int device) {                                                            \
    assert(device == CPU_DEVICE_ID);                                           \
    return THTensor##_new();                                                   \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::Free<HorovodType, DeviceType::CPU, THTensor>(THTensor *     \
                                                                tensor) {      \
    THTensor##_free(tensor);                                                   \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::ResizeNd<HorovodType, DeviceType::CPU, THTensor>(           \
      THTensor * tensor, int nDimension, int64_t* size, int64_t* stride) {     \
    THTensor##_resizeNd(tensor, nDimension, size, stride);                     \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::Copy<HorovodType, DeviceType::CPU, THTensor>(               \
      THTensor * output, THTensor * tensor) {                                  \
    THTensor##_copy(output, tensor);                                           \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void                                                                         \
  TensorUtil::DivideTensorInPlace<HorovodType, DeviceType::CPU, THTensor>(     \
      THTensor * tensor, int value) {                                          \
    THTensor##_div(tensor, tensor, value);                                     \
  }

#define TENSOR_UTIL_DEFINE_CUDA_TYPE(HorovodType, THCTensor, THTensor,         \
                                     THCStorage)                               \
  template <>                                                                  \
  const TensorShape TensorUtil::GetShape<HorovodType, DeviceType::GPU,         \
                                         THCTensor>(THCTensor * tensor) {      \
    TensorShape shape;                                                         \
    for (int idx = 0; idx < THCTensor##_nDimension(state, tensor); idx++) {    \
      shape.AddDim(THCTensor##_size(state, tensor, idx));                      \
    }                                                                          \
    return shape;                                                              \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  const void* TensorUtil::GetData<HorovodType, DeviceType::GPU, THCTensor>(    \
      THCTensor * tensor) {                                                    \
    return THCTensor##_data(state, tensor);                                    \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  int64_t TensorUtil::GetSize<HorovodType, DeviceType::GPU, THCTensor>(        \
      THCTensor * tensor) {                                                    \
    return (int64_t)(                                                          \
        THCStorage##_size(state, THCTensor##_storage(state, tensor)) *         \
        THCStorage##_elementSize(state));                                      \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  int TensorUtil::GetDevice<HorovodType, DeviceType::GPU, THCTensor>(          \
      THCTensor * tensor) {                                                    \
    return THCTensor##_getDevice(state, tensor);                               \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  THCTensor* TensorUtil::New<HorovodType, DeviceType::GPU, THCTensor>(         \
      int device) {                                                            \
    with_device device_context(device);                                        \
    return THCTensor##_new(state);                                             \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::Free<HorovodType, DeviceType::GPU, THCTensor>(THCTensor *   \
                                                                 tensor) {     \
    THCTensor##_free(state, tensor);                                           \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::ResizeNd<HorovodType, DeviceType::GPU, THCTensor>(          \
      THCTensor * tensor, int nDimension, int64_t* size, int64_t* stride) {    \
    with_device device_context(THCTensor##_getDevice(state, tensor));          \
    THCTensor##_resizeNd(state, tensor, nDimension, size, stride);             \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::Copy<HorovodType, DeviceType::GPU, THCTensor>(              \
      THCTensor * output, THCTensor * tensor) {                                \
    with_device device_context(THCTensor##_getDevice(state, output));          \
    THCTensor##_copy(state, output, tensor);                                   \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void                                                                         \
  TensorUtil::DivideTensorInPlace<HorovodType, DeviceType::GPU, THCTensor>(    \
      THCTensor * tensor, int value) {                                         \
    with_device device_context(THCTensor##_getDevice(state, tensor));          \
    THCTensor##_div(state, tensor, tensor, value);                             \
  }                                                                            \
                                                                               \
  template <>                                                                  \
  void TensorUtil::CopyCPUToCuda<HorovodType, THTensor, THCTensor>(            \
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
  void TensorUtil::AsyncCopyCudaToCPU<HorovodType, THCTensor, THTensor>(       \
      THCTensor * cuda, THTensor * cpu) {                                      \
    with_device device_context(THCTensor##_getDevice(state, cuda));            \
    THLongStorage* size = THCTensor##_newSizeOf(state, cuda);                  \
    if (!THTensor##_isSize(cpu, size)) {                                       \
      THTensor##_resize(cpu, size, NULL);                                      \
    }                                                                          \
    THLongStorage_free(size);                                                  \
    THTensor##_copyAsyncCuda(state, cpu, cuda);                                \
  }

TENSOR_UTIL_DEFINE_CPU_TYPE_H(MPIDataType::HOROVOD_UINT8, THByteTensor)
TENSOR_UTIL_DEFINE_CPU_TYPE_H(MPIDataType::HOROVOD_INT8, THCharTensor)
TENSOR_UTIL_DEFINE_CPU_TYPE_H(MPIDataType::HOROVOD_INT16, THShortTensor)
TENSOR_UTIL_DEFINE_CPU_TYPE_H(MPIDataType::HOROVOD_INT32, THIntTensor)
TENSOR_UTIL_DEFINE_CPU_TYPE_H(MPIDataType::HOROVOD_INT64, THLongTensor)
TENSOR_UTIL_DEFINE_CPU_TYPE_H(MPIDataType::HOROVOD_FLOAT32, THFloatTensor)
TENSOR_UTIL_DEFINE_CPU_TYPE_H(MPIDataType::HOROVOD_FLOAT64, THDoubleTensor)

#if HAVE_CUDA
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(MPIDataType::HOROVOD_UINT8, THCudaByteTensor,
                               THByteTensor)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(MPIDataType::HOROVOD_INT8, THCudaCharTensor,
                               THCharTensor)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(MPIDataType::HOROVOD_INT16, THCudaShortTensor,
                               THShortTensor)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(MPIDataType::HOROVOD_INT32, THCudaIntTensor,
                               THIntTensor)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(MPIDataType::HOROVOD_INT64, THCudaLongTensor,
                               THLongTensor)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(MPIDataType::HOROVOD_FLOAT32, THCudaTensor,
                               THFloatTensor)
TENSOR_UTIL_DEFINE_CUDA_TYPE_H(MPIDataType::HOROVOD_FLOAT64, THCudaDoubleTensor,
                               THDoubleTensor)
#endif

} // namespace torch
} // namespace horovod

#endif // HOROVOD_TORCH_TENSOR_UTIL_H
