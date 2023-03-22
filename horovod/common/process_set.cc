#include <algorithm>
#include <utility>

#if HAVE_GLOO
#include "gloo/allgather.h"
#include "gloo/allreduce.h"
#include "gloo/math.h"
#endif // HAVE_GLOO

#include "controller.h"
#include "process_set.h"

namespace horovod {
namespace common {

ProcessSet::ProcessSet(std::vector<int> global_ranks)
    : registered_global_ranks(std::move(global_ranks)) {}

bool ProcessSet::IsCurrentProcessIncluded() const {
  assert(initialization_done);
  return controller->IsInitialized();
}

namespace {

std::string RanksString(const std::vector<int>& ranks) {
  std::ostringstream oss_ranks;
  oss_ranks << "[";
  std::copy(ranks.begin(), ranks.end(),
            std::ostream_iterator<int>(oss_ranks, ","));
  oss_ranks << "]";
  return oss_ranks.str();
}

} // namespace

#if HAVE_MPI
bool ProcessSet::Initialize(const MPIContext& global_mpi_context) {
  if (initialization_done) {
    return false;
  }
  LOG(TRACE) << "Initializing new process set with MPI: "
             << RanksString(registered_global_ranks);
  assert(controller != nullptr);
  int size;
  MPI_Comm_size(global_mpi_context.global_comm, &size);
  if (!registered_global_ranks.empty()) {
    // Verify that each process has registered the same set of processes.
    std::vector<int> buf(size);
    assert((int)registered_global_ranks.size() <= size);
    auto len = static_cast<int>(registered_global_ranks.size());
    MPI_Allgather(&len, 1, MPI_INT, buf.data(), 1, MPI_INT,
                  global_mpi_context.global_comm);
    if (std::any_of(buf.begin(), buf.end(), [len](int other_len) {
          return len != other_len;
        })) {
      throw std::logic_error("Attempted to register process set with "
                             "mismatching size on different ranks");
    }
    for (auto reduction_op : {MPI_MAX, MPI_MIN}) {
      buf.resize(len);
      MPI_Allreduce(registered_global_ranks.data(), buf.data(), len, MPI_INT,
                    reduction_op, global_mpi_context.global_comm);
      if (registered_global_ranks != buf) {
        throw std::logic_error("Attempted to register process set with "
                               "mismatching values on different ranks");
      }
    }
  }
  mpi_context.InitializeForProcessSet(global_mpi_context,
                                      registered_global_ranks);
  if (mpi_context.GetMPICommunicator(Communicator::GLOBAL) != MPI_COMM_NULL) {
    // The running process is part of this process set.
    controller->Initialize();
  }
  if (registered_global_ranks.empty()) {
    registered_global_ranks.resize(size);
    std::iota(registered_global_ranks.begin(), registered_global_ranks.end(), 0);
  }
  initialization_done = true;
  return true;
}
#endif // HAVE_MPI

#if HAVE_GLOO
bool ProcessSet::Initialize(const GlooContext& global_gloo_context) {
  if (initialization_done) {
    return false;
  }
  LOG(TRACE) << "Initializing new process set with Gloo: "
             << RanksString(registered_global_ranks);
  assert(controller != nullptr);
  assert(global_gloo_context.ctx != nullptr);
  int size = global_gloo_context.ctx->size;
  if (!registered_global_ranks.empty()) {
    // Verify that each process has registered the same set of processes.
    std::vector<int> buf(size);
    assert((int)registered_global_ranks.size() <= size);
    auto len = static_cast<int>(registered_global_ranks.size());
    {
      gloo::AllgatherOptions opts(global_gloo_context.ctx);
      opts.setInput(&len, 1);
      opts.setOutput(buf.data(), size);
      gloo::allgather(opts);
    }
    if (std::any_of(buf.begin(), buf.end(),
                    [len](int other_len) { return len != other_len; })) {
      throw std::logic_error("Attempted to register process set with "
                             "mismatching size on different ranks");
    }

    for (auto marker : {'>', '<'}) {
      buf.resize(len);
      {
        gloo::AllreduceOptions opts(global_gloo_context.ctx);
        opts.setInput(registered_global_ranks.data(), len);
        opts.setOutput(buf.data(), len);
        if (marker == '>') {
          opts.setReduceFunction(
              static_cast<void (*)(void*, const void*, const void*, size_t)>(
                  &gloo::max<int>));
        } else if (marker == '<') {
          opts.setReduceFunction(
              static_cast<void (*)(void*, const void*, const void*, size_t)>(
                  &gloo::min<int>));
        }
        gloo::allreduce(opts);
      }
      if (registered_global_ranks != buf) {
        LOG(TRACE) << "buf: " << RanksString(buf);
        throw std::logic_error("Attempted to register process set with "
                               "mismatching values on different ranks");
      }
    }
  }
  gloo_context.InitializeForProcessSet(global_gloo_context,
                                       registered_global_ranks);
  if (gloo_context.ctx != nullptr) {
    controller->Initialize();
  }
  if (registered_global_ranks.empty()) {
    registered_global_ranks.resize(size);
    std::iota(registered_global_ranks.begin(), registered_global_ranks.end(), 0);
  }
  initialization_done = true;
  return true;
}
#endif // HAVE_GLOO

void ProcessSet::Finalize(const Status& status) {
  tensor_queue.FinalizeTensorQueue(status);
#if HAVE_MPI
  mpi_context.FinalizeWithoutEnv();
#endif // HAVE_MPI
#if HAVE_GLOO
  gloo_context.Finalize();
#endif // HAVE_GLOO
  // Clear the registered_global_ranks vector so the global process set can be
  // properly re-initialized with elastic Horovod.
  registered_global_ranks.clear();
  initialization_done = false;
}

Status ProcessSetTable::PROCESS_SET_HAS_BEEN_REMOVED =
    Status::Aborted("Process set has been removed");

ProcessSetTable::ProcessSetTable() {
  auto process_set_id = RegisterProcessSet();
  assert(process_set_id == 0);
}

template<class Context>
void ProcessSetTable::Initialize_(const Context& global_context) {
  std::lock_guard<std::recursive_mutex> guard(mutex);
  assert(next_id_ == 1);  // exactly one process set is registered
  Get(0).Initialize(global_context);
}

template <class Context>
int32_t ProcessSetTable::InitializeRegisteredAndRemoveMarkedIfReady_(
    const Context& global_context, const Status& removal_status) {
  std::lock_guard<std::recursive_mutex> guard(mutex);

  auto locally_registered_count = (int)ids_.size();
  std::array<int, 2> pair_to_transmit{locally_registered_count,
                                      id_to_be_removed_};

  auto& global_controller = *Get(0).controller;
  auto recv_buffer = std::vector<int>(2 * global_controller.GetSize());
  global_controller.Allgather2Ints(pair_to_transmit, recv_buffer);

  bool registered_count_agreement = true;
  bool id_to_be_removed_agreement = true;
  for (int i_locally_registered_count = 0;
       i_locally_registered_count < 2 * global_controller.GetSize() &&
       (registered_count_agreement || id_to_be_removed_agreement);
       i_locally_registered_count += 2) {
    auto other_reg_count = recv_buffer[i_locally_registered_count];
    auto other_marked_id = recv_buffer[i_locally_registered_count + 1];
    if (other_reg_count != locally_registered_count) {
      registered_count_agreement = false;
    }
    if (other_marked_id != id_to_be_removed_) {
      id_to_be_removed_agreement = false;
    }
  }

  // 1) Initialize registered process sets if all processes are ready to do so:
  int32_t initialized_count = 0;
  if (registered_count_agreement) {
    for (auto id : Ids()) {
      bool newly_registered = Get(id).Initialize(global_context);
      if (newly_registered) {
        ++initialized_count;
        LOG(TRACE, global_controller.GetRank())
            << "Initialized process set with id " << id;
      }
    }
  }

  // 2) Finalize and deregister a process set marked for removal by all processess:
  if (id_to_be_removed_agreement) {
    if (id_to_be_removed_ == NO_PENDING_REMOVAL ||
        id_to_be_removed_ == SUCCESSFUL_REMOVAL) {
      // do nothing
    } else {
      id_to_process_set_[id_to_be_removed_].Finalize(removal_status);
      DeregisterProcessSet(id_to_be_removed_);
      LOG(TRACE, global_controller.GetRank())
          << "Removed process set with id " << id_to_be_removed_;

      id_to_be_removed_ = SUCCESSFUL_REMOVAL;
    }
  }

  // Return count from 1)
  return initialized_count;
}

template <class Context>
void ProcessSetTable::Finalize_(const Context& context, const Status& status) {
  std::lock_guard<std::recursive_mutex> guard(mutex);
  std::vector<int32_t> ids_copy(ids_.begin(), ids_.end());
  for (auto id : ids_copy) {
    if (id == 0) {
      // We will still need the global process set to negotiate removing any
      // other process set.
      continue;
    }
    LOG(TRACE, Get(0).controller->GetRank())
        << "Finalizing ProcessSetTable, process set id: " << id;
    MarkProcessSetForRemoval(id);
    // Block until all processes have been able to finalize and deregister
    // this process set.
    while (true) {
      InitializeRegisteredAndRemoveMarkedIfReady_(context, status);
      if (ProcessSetHasJustBeenRemoved()) {
        break;
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }
  }
  // The process set hosting the global controller needs to remain in the
  // table to allow a future re-initialization of Horovod (it must still
  // be finalized now and re-initialized then).
  // We don't need to negotiate the removal of this global process set.
  LOG(TRACE, Get(0).controller->GetRank())
      << "Finalizing ProcessSetTable, global process set id 0";
  id_to_process_set_[0].Finalize(status);

  next_id_ = 1;
  while (!free_ids_.empty()) free_ids_.pop();  // Clear queue to be sure.
  assert(ids_.size() == 1);
  assert(id_to_process_set_.size() == 1);
}

#if HAVE_MPI
void ProcessSetTable::Initialize(const MPIContext& global_mpi_context) {
  Initialize_(global_mpi_context);
}

int32_t ProcessSetTable::InitializeRegisteredAndRemoveMarkedIfReady(
    const MPIContext& global_mpi_context, const Status& removal_status) {
  return InitializeRegisteredAndRemoveMarkedIfReady_(global_mpi_context,
                                                     removal_status);
}

void ProcessSetTable::Finalize(const MPIContext& global_mpi_context,
                               const Status& status) {
  Finalize_(global_mpi_context, status);
}
#endif // HAVE_MPI

#if HAVE_GLOO
void ProcessSetTable::Initialize(const GlooContext& global_gloo_context) {
  Initialize_(global_gloo_context);
}

int32_t ProcessSetTable::InitializeRegisteredAndRemoveMarkedIfReady(
    const GlooContext& global_gloo_context, const Status& status) {
  return InitializeRegisteredAndRemoveMarkedIfReady_(global_gloo_context,
                                                     status);
}

void ProcessSetTable::Finalize(const GlooContext& global_gloo_context,
                               const Status& status) {
  Finalize_(global_gloo_context, status);
}
#endif // HAVE_GLOO

int32_t ProcessSetTable::RegisterProcessSet(std::vector<int> global_ranks) {
  std::lock_guard<std::recursive_mutex> guard(mutex);

  if (!global_ranks.empty() && Contains(0)) {
    // We are registering a potentially non-global process set and we have
    // already registered the global process set 0. -> Check global_ranks
    std::sort(global_ranks.begin(), global_ranks.end());
    auto dup_it = std::adjacent_find(global_ranks.begin(), global_ranks.end());
    if (dup_it != global_ranks.end()) {
      throw std::logic_error(
          "Tried to register process set with duplicate rank: " +
          std::to_string(*dup_it));
    }
    for (auto rk : global_ranks) {
      if (rk < 0 || rk >= Get(0).controller->GetSize()) {
        throw std::logic_error(
            "Tried to register process set with invalid rank: " +
            std::to_string(rk));
      }
    }
  }

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
  std::lock_guard<std::recursive_mutex> guard(mutex);
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
  std::lock_guard<std::recursive_mutex> guard(mutex);
  return ids_;
}

bool ProcessSetTable::Contains(int32_t id) const {
  std::lock_guard<std::recursive_mutex> guard(mutex);
  return id_to_process_set_.find(id) != id_to_process_set_.end();
}

ProcessSet& ProcessSetTable::Get(int32_t id) {
  std::lock_guard<std::recursive_mutex> guard(mutex);
  return id_to_process_set_.at(id);
}

int32_t ProcessSetTable::FindId(const std::vector<int32_t>& ranks) {
  std::lock_guard<std::recursive_mutex> guard(mutex);
  for (auto id : Ids()) {
    if (Get(id).registered_global_ranks == ranks) {
      return id;
    }
  }
  return -1;
}

void ProcessSetTable::MarkProcessSetForRemoval(int32_t process_set_id) {
  if (process_set_id == 0) {
    throw std::logic_error("Attempted to remove global process set with id 0");
  }
  std::lock_guard<std::recursive_mutex> guard(mutex);
  assert(id_to_be_removed_ == NO_PENDING_REMOVAL);
  id_to_be_removed_ = process_set_id;
}

bool ProcessSetTable::ProcessSetHasJustBeenRemoved() {
  std::lock_guard<std::recursive_mutex> guard(mutex);
  if (id_to_be_removed_ == SUCCESSFUL_REMOVAL) {
    id_to_be_removed_ = NO_PENDING_REMOVAL;
    return true;
  }
  return false;
}

} // common
} // horovod
