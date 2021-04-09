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

  // If empty, all Horovod processes belong to this set. (Then ranks are stored
  // explicitly in controller).
  std::vector<int> registered_global_ranks_;

  std::atomic_bool initialization_done{false};

  // Number of ranks that did Join()
  int joined_size = 0;

  // If a rank is Joined, AllReduce uses temporary 0 tensors for it.
  bool joined = false;

#if HAVE_MPI
  MPICommunicators mpi_comms;

  // Before calling Initialize the controller must be populated.
  // TODO: doc
  void Initialize(const MPIContext& mpi_context);
#endif // HAVE_MPI

#if HAVE_GLOO
  void Initialize(const GlooContext& gloo_context,
                  const std::vector<int>& global_ranks = {});
#endif // HAVE_GLOO

  // Finalize tensor queue and communicators.
  void Finalize(const Status& status);

  bool IsCurrentProcessIncluded() const;

  // If an empty vector is passed, all Horovod processes will be part of this
  // process set.
  explicit ProcessSet(std::vector<int> global_ranks = {});

  ProcessSet(const ProcessSet&) = delete;
};

// TODO: Special handling for id=0 (global process set)?

class ProcessSetTable {
public:
  ProcessSetTable();
  ProcessSetTable(const ProcessSetTable&) = delete;

#if HAVE_MPI
  // TODO: doc
  void Initialize(const MPIContext& mpi_context);

  void InitializeRegisteredIfReady(const MPIContext& mpi_context);
#endif // HAVE_MPI

#if HAVE_GLOO
  void Initialize(const GlooContext& gloo_context);

  void InitializeRegisteredIfReady(const GlooContext& gloo_context);
#endif // HAVE_GLOO

  // Finalize tensor queues and communicators and deregister process sets.
  void Finalize(const Status& status);

  int32_t RegisterProcessSet(const std::vector<int>& global_ranks = {});

  void DeregisterProcessSet(int32_t process_set_id);

  std::vector<int32_t> Ids() const; // Returns copy to be threadsafe

  ProcessSet& Get(int32_t id) { return id_to_process_set_.at(id); }

private:
  std::unordered_map<int32_t, ProcessSet> id_to_process_set_;

  // Tracks ids by insertion order
  std::vector<int32_t> ids_;

  // Queue of ids that can be reused
  std::queue<int32_t> free_ids_;

  // Next available id (increases when a process set is added and no id is reused)
  int32_t next_id_ = 0;

  // Guard access to this table by this mutex
  mutable std::recursive_mutex mutex_;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_PROCESS_SET_H
