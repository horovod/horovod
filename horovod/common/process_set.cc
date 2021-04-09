#include <algorithm>
#include <utility>

#include "controller.h"
#include "process_set.h"

namespace horovod {
namespace common {

ProcessSet::ProcessSet(std::vector<int> global_ranks)
    : registered_global_ranks_(std::move(global_ranks)) {}

bool ProcessSet::IsCurrentProcessIncluded() const {
#if HAVE_MPI
  return mpi_comms.Get(CommunicatorType::GLOBAL) != MPI_COMM_NULL;
#endif // HAVE_MPI
  return true;
}

#if HAVE_MPI
void ProcessSet::Initialize(const MPIContext& mpi_context) {
  if (initialization_done) {
    return;
  }
  LOG(TRACE) << "Initializing new process set.";
  assert(controller != nullptr);
  mpi_comms.Initialize(mpi_context, registered_global_ranks_);
  if (IsCurrentProcessIncluded()) {
    controller->Initialize();
  }
  initialization_done = true;
}
#endif // HAVE_MPI

void ProcessSet::Finalize(const Status& status) {
  tensor_queue.FinalizeTensorQueue(status);
#if HAVE_MPI
  mpi_comms.Finalize();
#endif // HAVE_MPI
}

ProcessSetTable::ProcessSetTable() {
  auto process_set_id = RegisterProcessSet();
  assert(process_set_id == 0);
}

#if HAVE_MPI
void ProcessSetTable::Initialize(const MPIContext& mpi_context) {
  std::lock_guard<std::recursive_mutex> guard(mutex_);
  assert(next_id_ == 1);  // exactly one process set is registered
  Get(0).Initialize(mpi_context);
}

void ProcessSetTable::InitializeRegisteredIfReady(const MPIContext& mpi_context) {
  std::lock_guard<std::recursive_mutex> guard(mutex_);

  int locally_registered_count = ids_.size();
  auto& global_controller = *Get(0).controller;
  auto registered_counts = std::vector<int>(global_controller.GetSize());
  global_controller.AllgatherInt(locally_registered_count, registered_counts);
  if (std::any_of(registered_counts.begin(), registered_counts.end(),
                  [locally_registered_count](int reg_count) {
                    return reg_count != locally_registered_count;
                  })) {
    // Do not initialize newly added process sets until every process has
    // registered them.
    return;
  }

  for (auto id: Ids()) {
    Get(id).Initialize(mpi_context);
  }
}
#endif // HAVE_MPI

void ProcessSetTable::Finalize(const Status& status) {
  std::lock_guard<std::recursive_mutex> guard(mutex_);
  std::vector<int32_t> ids_copy(ids_.begin(), ids_.end());
  for (auto id: ids_copy) {
    id_to_process_set_[id].Finalize(status);
    DeregisterProcessSet(id);
  }
}

int32_t ProcessSetTable::RegisterProcessSet(const std::vector<int>& global_ranks) {
  std::lock_guard<std::recursive_mutex> guard(mutex_);

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
                             std::forward_as_tuple(global_ranks));
  ids_.push_back(id);

  return id;
}

void ProcessSetTable::DeregisterProcessSet(int32_t process_set_id) {
  std::lock_guard<std::recursive_mutex> guard(mutex_);
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
std::vector<int32_t> ProcessSetTable::Ids() const {
  std::lock_guard<std::recursive_mutex> guard(mutex_);
  return ids_;
}

ProcessSet& ProcessSetTable::Get(int32_t id) {
  std::lock_guard<std::recursive_mutex> guard(mutex_);
  return id_to_process_set_.at(id);
}

void ProcessSetTable::MarkProcessSetForRemoval(int32_t process_set_id) {
  std::lock_guard<std::recursive_mutex> guard(mutex_);
  assert(id_to_be_removed_ == NO_PENDING_REMOVAL);
  id_to_be_removed_ = process_set_id;
}

bool ProcessSetTable::ProcessSetHasJustBeenRemoved() {
  std::lock_guard<std::recursive_mutex> guard(mutex_);
  if (id_to_be_removed_ == SUCCESSFUL_REMOVAL) {
    id_to_be_removed_ = NO_PENDING_REMOVAL;
    return true;
  }
  return false;
}

void ProcessSetTable::RemoveMarkedProcessSetIfReady() {
  std::lock_guard<std::recursive_mutex> guard(mutex_);

  auto& global_controller = *Get(0).controller;
  auto ids_marked_on_all_ranks = std::vector<int>(global_controller.GetSize());
  global_controller.AllgatherInt(id_to_be_removed_, ids_marked_on_all_ranks);
  if (std::any_of(
          ids_marked_on_all_ranks.begin(), ids_marked_on_all_ranks.end(),
          [this](int other_id) { return other_id != id_to_be_removed_; })) {
    // Do not remove marked process set until every process has marked the same.
    return;
  }
  if (id_to_be_removed_ == NO_PENDING_REMOVAL ||
      id_to_be_removed_ == SUCCESSFUL_REMOVAL) {
    return;
  }

  id_to_process_set_[id_to_be_removed_].Finalize(
      Status::Aborted("Process set has been removed"));
  DeregisterProcessSet(id_to_be_removed_);

  id_to_be_removed_ = SUCCESSFUL_REMOVAL;
}

} // common
} // horovod
