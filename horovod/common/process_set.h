#ifndef HOROVOD_PROCESS_SET_H
#define HOROVOD_PROCESS_SET_H

#include <atomic>
#include <list>
#include <queue>
#include <unordered_map>

#include "response_cache.h"
#include "tensor_queue.h"

#if HAVE_MPI
#include "mpi/mpi_context.h"
#endif // HAVE_MPI

#if HAVE_GLOO
#include "gloo/gloo_context.h"
#endif // HAVE_GLOO

// Forward declaration
class Controller;

namespace horovod {
namespace common {

struct ProcessSet {
  std::shared_ptr<Controller> controller;

  TensorQueue tensor_queue;

  // LRU cache of Responses
  ResponseCache response_cache;

  // Information on registered groups.
  GroupTable group_table;

  // If this is empty before initialization, all Horovod
  // processes will belong to this set. After initialization this always
  // enumerates all ranks belonging to the proces set.
  std::vector<int> registered_global_ranks;

  std::atomic_bool initialization_done{false};

  // Number of ranks that did Join()
  int joined_size = 0;

  // Last global rank that did Join()
  int32_t last_joined_rank = -1;

  // If a rank is Joined, AllReduce uses temporary 0 tensors for it.
  bool joined = false;

  // Pointer to shared buffer for allgather
  void* shared_buffer = nullptr;

  // Current shared buffer size
  int64_t shared_buffer_size = 0;

#if HAVE_MPI
  MPIContext mpi_context;

  // Before calling Initialize, controller and registered_ranks should be set.
  // Returns true if it newly initializes the process set, false if it was
  // already intialized before.
  bool Initialize(const MPIContext& global_mpi_context);
#endif // HAVE_MPI

#if HAVE_GLOO
  GlooContext gloo_context;

  bool Initialize(const GlooContext& global_gloo_context);
#endif // HAVE_GLOO

  // Finalize tensor queue and communicators.
  void Finalize(const Status& status);

  bool IsCurrentProcessIncluded() const;

  // If an empty vector is passed, all Horovod processes will be part of this
  // process set.
  explicit ProcessSet(std::vector<int> global_ranks = {});

  ProcessSet(const ProcessSet&) = delete;
};

/*
const Status *PROCESS_SET_HAS_BEEN_REMOVED =
    &Status::Aborted("Process set has been removed");
*/

class ProcessSetTable {
public:
  ProcessSetTable();
  ProcessSetTable(const ProcessSetTable&) = delete;

  static Status PROCESS_SET_HAS_BEEN_REMOVED;

  // Initialize table and the global process set with id 0, to be called in
  // background thread.
#if HAVE_MPI
  void Initialize(const MPIContext& global_mpi_context);
#endif
#if HAVE_GLOO
  void Initialize(const GlooContext& global_gloo_context);
#endif

  // To be called in the background thread: 1) Initialize any process sets
  // that have been registered by all processes.
  // 2) Deregister a process set (just one) that has been marked for removal by
  // all processes.
  // Returns the number of newly initialized process sets (may be zero) from 1).
#if HAVE_MPI
  int32_t InitializeRegisteredAndRemoveMarkedIfReady(
      const MPIContext& global_mpi_context,
      const Status& removal_status = PROCESS_SET_HAS_BEEN_REMOVED);
#endif
#if HAVE_GLOO
  int32_t InitializeRegisteredAndRemoveMarkedIfReady(
      const GlooContext& global_gloo_context,
      const Status& removal_status = PROCESS_SET_HAS_BEEN_REMOVED);
#endif

  // Finalize tensor queues and communicators and remove all process sets.
#if HAVE_MPI
  void Finalize(const MPIContext& global_mpi_context, const Status& status);
#endif
#if HAVE_GLOO
  void Finalize(const GlooContext& global_gloo_context, const Status& status);
#endif

  int32_t RegisterProcessSet(std::vector<int> global_ranks = {});

  std::vector<int32_t> Ids() const; // Returns copy to be threadsafe

  bool Contains(int32_t id) const;

  ProcessSet& Get(int32_t id);

  // Returns -1 if no process set with these ranks has been registered.
  int32_t FindId(const std::vector<int32_t>& ranks);

  void MarkProcessSetForRemoval(int32_t process_set_id);

  bool ProcessSetHasJustBeenRemoved();

  // Guard access to the table by this mutex
  mutable std::recursive_mutex mutex;

private:
  void DeregisterProcessSet(int32_t process_set_id);

  // Context can be either MPIContext or GlooContext for the following:
  template <class Context> void Initialize_(const Context& context);
  template <class Context>
  int32_t
  InitializeRegisteredAndRemoveMarkedIfReady_(const Context& context,
                                              const Status& removal_status);
  template <class Context>
  void Finalize_(const Context& context, const Status& removal_status);

  std::unordered_map<int32_t, ProcessSet> id_to_process_set_;

  // Tracks ids by insertion order
  std::vector<int32_t> ids_;

  // Queue of ids that can be reused
  std::queue<int32_t> free_ids_;

  // Next available id (increases when a process set is added and no id is reused)
  int32_t next_id_ = 0;

  static constexpr int32_t NO_PENDING_REMOVAL = -1;
  static constexpr int32_t SUCCESSFUL_REMOVAL = -2;
  int32_t id_to_be_removed_ = NO_PENDING_REMOVAL;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_PROCESS_SET_H
