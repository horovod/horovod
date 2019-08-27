// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

int32_t GroupTable::GetGroupSize(int32_t group_id) const {
  assert(registered_groups_.size() > group_id);
  std::lock_guard<std::mutex> guard(mutex_);
  return registered_groups_[group_id].size;
}

int32_t GroupTable::GetGroupIDFromName(std::string group_name) const {
  std::lock_guard<std::mutex> guard(mutex_);
  return group_name_to_id_.at(group_name);
}

bool GroupTable::IsNameRegistered(std::string group_name) const {
  std::lock_guard<std::mutex> guard(mutex_);
  return group_name_to_id_.find(group_name) != group_name_to_id_.end();
}

bool GroupTable::IsIDRegistered(int32_t group_id) const {
  std::lock_guard<std::mutex> guard(mutex_);
  return (uint32_t) registered_groups_.size() > group_id;
}

bool GroupTable::empty(void) const {
  std::lock_guard<std::mutex> guard(mutex_);
  return registered_groups_.empty();
}

int32_t GroupTable::AddEntry(std::string name, int32_t size) {
  std::lock_guard<std::mutex> guard(mutex_);

  GroupTableEntry e;
  int32_t group_id = next_group_id_;
  e.id = group_id;
  e.size= size;
  registered_groups_.push_back(std::move(e));
  group_name_to_id_[name] = group_id;

  next_group_id_++;

  return group_id;
}

} // namespace common
} // namespace horovod
