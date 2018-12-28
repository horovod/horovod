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

#include "handle_manager.h"

namespace horovod {
namespace mxnet {

typedef ::mxnet::Engine::CallbackOnComplete Callback;

int HandleManager::AllocateHandle(Callback cb) {
  int handle = last_handle_.fetch_add(1) + 1;
  std::lock_guard<std::mutex> guard(mutex_);
  results_[handle] = nullptr;
  callbacks_[handle] = std::make_shared<Callback>(cb);
  return handle;
}

void HandleManager::MarkDone(int handle, const Status& status) {
  std::lock_guard<std::mutex> guard(mutex_);
  results_[handle] = std::make_shared<Status>(status);
}

void HandleManager::ExecuteCallback(int handle) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (callbacks_.find(handle) == callbacks_.end()) {
    return;
  }
  auto cb_ptr = callbacks_[handle];
  lock.unlock();
  if (cb_ptr != nullptr) {
    (*cb_ptr)();
  }
}

} // namespace mxnet
} // namespace horovod
