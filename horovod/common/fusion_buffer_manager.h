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

#ifndef HOROVOD_FUSION_BUFFER_MANAGER_H
#define HOROVOD_FUSION_BUFFER_MANAGER_H

#include <iostream>
#include <unordered_map>

#include "common.h"
#include "hashes.h"
#include "operations.h"

namespace horovod {
namespace common {

class FusionBufferManager {
public:
  FusionBufferManager(int64_t default_threshold);
  void SetInitialThreshold(int64_t threshold);
  Status InitializeBuffer(int device, std::shared_ptr<OpContext> context,
                          std::function<void()> on_start_init,
                          std::function<void()> on_end_init);
  std::shared_ptr<PersistentBuffer>& GetBuffer(int device, Framework framework);
  inline int64_t GetThreshold() { return threshold_; };

private:
  int64_t threshold_;

  // Memory buffers for Tensor Fusion.  They are keyed off device ID and
  // framework, and all are allocated tensor_fusion_threshold bytes if
  // initialized.
  std::unordered_map<std::tuple<int, Framework>,
      std::shared_ptr<PersistentBuffer>>
      tensor_fusion_buffers_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_FUSION_BUFFER_MANAGER_H
