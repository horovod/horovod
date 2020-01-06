// Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

#ifndef HOROVOD_MXNET_TENSOR_UTIL_H
#define HOROVOD_MXNET_TENSOR_UTIL_H

#include <cassert>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>

#include "../common/common.h"
#include "cuda_util.h"
#include "util.h"

namespace horovod {
namespace mxnet {

using namespace horovod::common;
using namespace ::mxnet;

class TensorUtil {
public:
  static const DataType GetDType(NDArray* tensor);
  static const TensorShape GetShape(NDArray* tensor);
  static const void* GetData(NDArray* tensor);
  static int64_t GetSize(NDArray* tensor);
  static int GetDevice(NDArray* tensor);

  static void ResizeNd(NDArray* tensor, int ndim, int64_t* dims);
  static void Copy(NDArray* output, NDArray* tensor);
  static void DivideTensorInPlace(NDArray* tensor, int value);

#if HAVE_CUDA
  static void AsyncCopyCPUToCuda(NDArray* cpu, NDArray* cuda);
  static void AsyncCopyCudaToCPU(NDArray* cuda, NDArray* cpu);
#endif

private:
  static const size_t kFloat32Size = 4;
  static const size_t kFloat64Size = 8;
  static const size_t kFloat16Size = 2;
  static const size_t kUInt8Size = 1;
  static const size_t kInt32Size = 4;
  static const size_t kInt8Size = 1;
  static const size_t kInt64Size = 8;
};

} // namespace mxnet
} // namespace horovod

#endif // HOROVOD_MXNET_TENSOR_UTIL_H
