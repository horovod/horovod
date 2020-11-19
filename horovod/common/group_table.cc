// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "group_table.h"

#include <assert.h>

namespace horovod {
namespace common {

int32_t GroupTable::GetGroupIDFromTensorName(const std::string& tensor_name) const {
  std::lock_guard<std::mutex> guard(mutex_);
  auto it = tensor_name_to_id_.find(tensor_name);
  if (it != tensor_name_to_id_.end())
    return it->second;
  else {
    return NULL_GROUP_ID;
  }
}

const std::vector<std::string>& GroupTable::GetGroupTensorNames(int32_t group_id) const {
  std::lock_guard<std::mutex> guard(mutex_);
  return id_to_tensor_names_.at(group_id);
}

bool GroupTable::empty(void) const {
  std::lock_guard<std::mutex> guard(mutex_);
  return tensor_name_to_id_.empty();
}

int32_t GroupTable::RegisterGroup(std::vector<std::string>&& tensor_names) {
  std::lock_guard<std::mutex> guard(mutex_);

  int32_t group_id;
  if (!free_ids_.empty()) {
    // Reuse old group_id
    group_id = free_ids_.front();
    free_ids_.pop();
  } else {
    // Create a new group_id
    group_id = next_group_id_++;
  }

  for (auto& name : tensor_names) {
    tensor_name_to_id_.emplace(name, group_id);
  }
  id_to_tensor_names_.emplace(group_id, std::move(tensor_names));

  return group_id;
}

void GroupTable::DeregisterGroups(const std::vector<std::string>& tensor_names) {
  std::lock_guard<std::mutex> guard(mutex_);

  for (auto& name : tensor_names) {
    auto it = tensor_name_to_id_.find(name);
    if (it != tensor_name_to_id_.end()) {
      auto group_id = it->second;
      for (auto& entry : id_to_tensor_names_[group_id]) {
        tensor_name_to_id_.erase(entry);
      }
      id_to_tensor_names_.erase(group_id);

      free_ids_.push(group_id);
    }
  }
}

} // namespace common
} // namespace horovod
