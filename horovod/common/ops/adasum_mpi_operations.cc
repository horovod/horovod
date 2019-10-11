//TODO license
#include "adasum_mpi_operations.h"

namespace horovod {
namespace common {
AdasumMPIAllreduceOp::AdasumMPIAllreduceOp(MPIContext* mpi_context, HorovodGlobalState* global_state)
    : AllreduceOp(global_state), AdasumMPIP2pOp(mpi_context) {}

bool AdasumMPIAllreduceOp::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

Status AdasumMPIAllreduceOp::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  if(entries.empty()) {
      return Status::OK();
  }
  auto& first_entry = entries[0];

  void* buffer_data;
  size_t buffer_len;

  // Copy memory into the fusion buffer.
  auto& timeline = global_state_->timeline;
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    const void* fused_input_data;
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);
    timeline.ActivityEndAll(entries);
  } else {
    buffer_data = (void*) first_entry.output->data();
    buffer_len = (size_t) first_entry.output->size();
    if (first_entry.tensor->data() != first_entry.output->data()) {
      std::memcpy(buffer_data, (void*)first_entry.tensor->data(), buffer_len);
    }
  }

  // Do allreduce.
  timeline.ActivityStartAll(entries, MPI_ALLREDUCE);
	std::vector<int> tensor_counts;
  for (auto& e : entries) {
    tensor_counts.push_back(e.tensor->shape().num_elements());
  }
  std::unique_ptr<char[]> recv_buffer = std::unique_ptr<char[]>(new char[buffer_len]);
  DispatchFusedAllreduce(buffer_data, recv_buffer.get(), tensor_counts,
                    1, // start_level
                    mpi_context_->GetMPICommunicator(Communicator::GLOBAL),
                    0, // tag
                    world_reduction_comms_,
                    first_entry.tensor->dtype());
  timeline.ActivityEndAll(entries);

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(buffer_data, entries);
    timeline.ActivityEndAll(entries);
  }

  return Status::OK();
}
} // namespace common
} // namespace horovod
