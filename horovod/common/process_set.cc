#include <algorithm>

#include "process_set.h"

namespace horovod {
namespace common {

int32_t ProcessSetTable::RegisterProcessSet(
    const std::vector<int32_t>& global_ranks) {
  int32_t id;
  if (!free_ids_.empty()) {
    id = free_ids_.front();
    free_ids_.pop();
  } else {
    id = next_id_++;
  }

  // emplace (id, ProcessSet()) without requiring a copy constructor (would be
  // nicer in C++17 with try_emplace)
  id_to_process_set_.emplace(std::piecewise_construct,
                             std::forward_as_tuple(id),
                             std::forward_as_tuple());   // TODO: ranks, MPI_Context
  ids_.push_back(id);

  return id;
}

void ProcessSetTable::DeregisterProcessSet(int32_t process_set_id) {
  auto map_it = id_to_process_set_.find(process_set_id);
  if (map_it != id_to_process_set_.end()) {
    id_to_process_set_.erase(map_it);

    auto ids_it = std::find(ids_.begin(), ids_.end(), process_set_id);
    if (ids_it != ids_.end()) {
      ids_.erase(ids_it);
    }

    free_ids_.push(process_set_id);
  }
}

} // common
} // horovod
