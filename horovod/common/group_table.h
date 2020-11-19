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

#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

#ifndef HOROVOD_GROUP_TABLE_H
#define HOROVOD_GROUP_TABLE_H

// Null group ID used when no group is assigned.
#define NULL_GROUP_ID -1

namespace horovod {
namespace common {

class GroupTable {
public:
  GroupTable() = default;
  GroupTable(const GroupTable&) = delete;

  int32_t GetGroupIDFromTensorName(const std::string& tensor_name) const;
  const std::vector<std::string>& GetGroupTensorNames(int32_t group_id) const;
  bool empty(void) const;

  int32_t RegisterGroup(std::vector<std::string>&& tensor_names);
  void DeregisterGroups(const std::vector<std::string>& tensor_names);

private:
  std::unordered_map<std::string, int32_t> tensor_name_to_id_;
  std::unordered_map<int32_t, std::vector<std::string>> id_to_tensor_names_;

  // Queue of ids that can be reused
  std::queue<int32_t> free_ids_;

  // Next available group id (increases each time a group is added)
  int32_t next_group_id_ = 0;

  mutable std::mutex mutex_;

};

} // namespace common
} // namespace horovod

#endif // HOROVOD_GROUP_TABLE_H
