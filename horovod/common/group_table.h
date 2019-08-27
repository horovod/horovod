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

#include <mutex>
#include <unordered_map>
#include <vector>

#ifndef HOROVOD_GROUP_TABLE_H
#define HOROVOD_GROUP_TABLE_H
namespace horovod {
namespace common {

// Table storing information on groups for allreduce grouping.
struct GroupTableEntry {
  int32_t id;
  int32_t size;
  // Note: We can add other data to this struct to enable more error checking
  // if desired.
};

class GroupTable {
public:
  GroupTable() = default;
  GroupTable(const GroupTable&) = delete;

  int32_t GetGroupSize(int32_t group_id) const;
  int32_t GetGroupIDFromName(std::string group_name) const;
  bool IsNameRegistered(std::string group_name) const;
  bool IsIDRegistered(int32_t group_id) const;
  bool empty(void) const;

  int32_t AddEntry(std::string name, int32_t size);

private:
  std::vector<GroupTableEntry> registered_groups_;

  // Map to obtain assigned group id value from names
  std::unordered_map<std::string, int32_t> group_name_to_id_;

  // Next available group id (increases each time a group is added)
  int32_t next_group_id_ = 0;

  mutable std::mutex mutex_;

};

} // namespace common
} // namespace horovod

#endif // HOROVOD_GROUP_TABLE_H
