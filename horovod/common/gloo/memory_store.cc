// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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
// ============================================================================

#include "memory_store.h"

#include <chrono>
#include <thread>

#include "gloo/common/error.h"

namespace horovod {
namespace common {

void MemoryStore::set(const std::string& key, const std::vector<char>& data) {
  map_[key] = data;
}

std::vector<char> MemoryStore::get(const std::string& key) {
  return map_[key];
}

void MemoryStore::wait(const std::vector<std::string>& keys) {
  for (auto& key : keys) {
    while (map_.find(key) == map_.end()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

void MemoryStore::wait(const std::vector<std::string>& keys,
                       const std::chrono::milliseconds& timeout) {
  const auto start = std::chrono::steady_clock::now();
  for (auto& key : keys) {
    while (map_.find(key) == map_.end()) {
      auto now = std::chrono::steady_clock::now();
      const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start);
      if (timeout != gloo::kNoTimeout && elapsed > timeout) {
        GLOO_THROW_IO_EXCEPTION(GLOO_ERROR_MSG("Wait timeout for key(s): ",
                                               ::gloo::MakeString(keys)));
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

void MemoryStore::Finalize() {
  map_.clear();
}

} // namespace common
} // namespace horovod

