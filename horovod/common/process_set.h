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

  // If this is empty before initialization, but after registration, all Horovod
  // processes belong to this set. After initialization this always enumerates
  // all ranks belonging to the proces set.
  std::vector<int> registered_global_ranks;

  std::atomic_bool initialization_done{false};

  // Number of ranks that did Join()
  int joined_size = 0;

  // If a rank is Joined, AllReduce uses temporary 0 tensors for it.
  bool joined = false;

  // Pointer to shared buffer for allgather
  void* shared_buffer = nullptr;

  // Current shared buffer size
  int64_t shared_buffer_size = 0;

#if HAVE_MPI
  MPIContext mpi_context;

  // Before calling Initialize the controller must be populated.
  // TODO: doc
  void Initialize(const MPIContext& global_mpi_context);
#endif // HAVE_MPI

#if HAVE_GLOO
  void Initialize(const GlooContext& gloo_context);
#endif // HAVE_GLOO

  // Finalize tensor queue and communicators.
  void Finalize(const Status& status);

  bool IsCurrentProcessIncluded() const;

  // If an empty vector is passed, all Horovod processes will be part of this
  // process set.
  explicit ProcessSet(std::vector<int> global_ranks = {});

  ProcessSet(const ProcessSet&) = delete;
};

// TODO: Make more explicit that id=0 (the global process set) is special? Prevent removal?

class ProcessSetTable {
public:
  ProcessSetTable();
  ProcessSetTable(const ProcessSetTable&) = delete;

#if HAVE_MPI
  // TODO: doc
  void Initialize(const MPIContext& global_mpi_context);

  void InitializeRegisteredIfReady(const MPIContext& global_mpi_context);
#endif // HAVE_MPI

#if HAVE_GLOO
  void Initialize(const GlooContext& gloo_context);

  // void InitializeRegisteredIfReady(const GlooContext& gloo_context);
#endif // HAVE_GLOO

  // Finalize tensor queues and communicators and remove all process sets.
  void Finalize(const Status& status);

  int32_t RegisterProcessSet(std::vector<int> global_ranks = {});

  std::vector<int32_t> Ids() const; // Returns copy to be threadsafe

  bool Contains(int32_t id) const;

  ProcessSet& Get(int32_t id);

  void MarkProcessSetForRemoval(int32_t process_set_id);

  bool ProcessSetHasJustBeenRemoved();

  void RemoveMarkedProcessSetIfReady();

  // Guard access to the table by this mutex
  mutable std::recursive_mutex mutex;

private:
  void DeregisterProcessSet(int32_t process_set_id);

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
