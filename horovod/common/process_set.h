#ifndef HOROVOD_PROCESS_SET_H
#define HOROVOD_PROCESS_SET_H

#include <list>
#include <unordered_map>
#include <queue>

#include "response_cache.h"
#include "tensor_queue.h"

#if HAVE_MPI
#include "mpi/mpi_context.h"
#endif // HAVE_MPI

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

  // Number of ranks that did Join()
  int joined_size = 0;

  // If a rank is Joined, AllReduce uses temporary 0 tensors for it.
  bool joined = false;

#if HAVE_MPI
  MPICommunicators mpi_comms;
#endif // HAVE_MPI

  // TODO: Initialize, Finalize

  bool IsCurrentProcessIncluded() const;

  ProcessSet() = default;
  ProcessSet(const ProcessSet&) = delete;
};

// TODO: Special handling for id=0 (global process set)?

class ProcessSetTable {
public:
  ProcessSetTable() = default;
  ProcessSetTable(const ProcessSetTable&) = delete;

  int32_t RegisterProcessSet();
  void DeregisterProcessSet(int32_t process_set_id);

  // TODO: thread safe?
  const std::list<int32_t>& Ids() const { return ids_; }
  ProcessSet& Get(int32_t id) { return id_to_process_set_.at(id); }
  bool Empty() const { return id_to_process_set_.empty(); }

private:
  std::unordered_map<int32_t, ProcessSet> id_to_process_set_;

  // Tracks ids by insertion order
  std::list<int32_t> ids_;

  // Queue of ids that can be reused
  std::queue<int32_t> free_ids_;

  // Next available id (increases when a process set is added and no id is reused)
  int32_t next_id_ = 0;

  // TODO: mutex?
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_PROCESS_SET_H
