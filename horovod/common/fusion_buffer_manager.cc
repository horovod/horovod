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

#include "fusion_buffer_manager.h"

namespace horovod {
namespace common {

FusionBufferManager::FusionBufferManager(int64_t default_threshold) : threshold_(default_threshold) {}

void FusionBufferManager::SetInitialThreshold(int64_t threshold) {
  threshold_ = threshold;
}

Status FusionBufferManager::InitializeBuffer(int device, std::shared_ptr<OpContext> context,
                                             std::function<void()> on_start_init,
                                             std::function<void()> on_end_init) {
  auto& buffer = GetBuffer(device, context->framework());
  if (buffer == nullptr) {
    on_start_init();

    // Lazily allocate persistent buffer for Tensor Fusion and keep it
    // forever per device.
    Status status = context->AllocatePersistent(threshold_, &buffer);
    if (status.ok()) {
      on_end_init();
    }

    return status;
  }

  return Status::OK();
}

std::shared_ptr<PersistentBuffer>& FusionBufferManager::GetBuffer(int device, Framework framework) {
  return tensor_fusion_buffers_[std::make_tuple(device, framework)];
}

} // namespace common
} // namespace horovod