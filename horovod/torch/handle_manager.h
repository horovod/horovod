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

#ifndef HOROVOD_TORCH_HANDLE_MANAGER_H
#define HOROVOD_TORCH_HANDLE_MANAGER_H

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "../common/common.h"

namespace horovod {
namespace torch {

using namespace horovod::common;

class HandleManager {
public:
  int AllocateHandle();
  void MarkDone(int handle, const Status& status);
  bool PollHandle(int handle);
  std::shared_ptr<Status> ReleaseHandle(int handle);

private:
  std::atomic_int last_handle_;
  std::unordered_map<int, std::shared_ptr<Status>> results_;
  std::mutex mutex_;
};

} // namespace torch
} // namespace horovod

#endif // HOROVOD_TORCH_HANDLE_MANAGER_H
