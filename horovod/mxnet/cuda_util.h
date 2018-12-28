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

#ifndef HOROVOD_MXNET_CUDA_UTIL_H
#define HOROVOD_MXNET_CUDA_UTIL_H

namespace horovod {
namespace mxnet {

class with_device {
public:
  with_device(int device);
  ~with_device();

private:
  int restore_device_;
};

} // namespace mxnet
} // namespace horovod

#endif // HOROVOD_MXNET_CUDA_UTIL_H
