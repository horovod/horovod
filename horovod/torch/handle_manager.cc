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

#include "handle_manager.h"

namespace horovod {
namespace torch {

int HandleManager::AllocateHandle() {
  int handle = last_handle_.fetch_add(1) + 1;
  std::lock_guard<std::mutex> guard(mutex_);
  results_[handle] = nullptr;
  return handle;
}

void HandleManager::MarkDone(int handle, const Status& status) {
  std::lock_guard<std::mutex> guard(mutex_);
  results_[handle] = std::make_shared<Status>(status);
}

bool HandleManager::PollHandle(int handle) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (results_.find(handle) == results_.end()) {
    throw std::invalid_argument("Handle " + std::to_string(handle) +
                                " was not created or has been cleared.");
  }
  return results_[handle] != nullptr;
}

std::shared_ptr<Status> HandleManager::ReleaseHandle(int handle) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (results_.find(handle) == results_.end()) {
    throw std::invalid_argument("Handle " + std::to_string(handle) +
        " was not created or has been cleared.");
  }
  auto status = results_[handle];
  results_.erase(handle);
  return status;
}

} // namespace torch
} // namespace horovod