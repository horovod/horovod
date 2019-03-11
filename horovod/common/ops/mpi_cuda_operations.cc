// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
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

#include "mpi_cuda_operations.h"

namespace horovod {
namespace common {

MPI_CUDAAllreduce::MPI_CUDAAllreduce(MPIContext* mpi_context,
                                     CUDAContext* cuda_context,
                                     HorovodGlobalState* global_state)
    : CUDAAllreduce(cuda_context, global_state),
      mpi_context_(mpi_context) {}

Status MPI_CUDAAllreduce::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& first_entry = entries[0];

  InitCUDA(entries);

  void* buffer_data;
  size_t buffer_len;
  int64_t num_elements = NumElements(entries);

  // Copy memory into the fusion buffer.
  auto& timeline = global_state_->timeline;
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    const void* fused_input_data;
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);

    auto cuda_result = cudaStreamSynchronize(cuda_context_->streams[entries[0].device]);
    cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);

    timeline.ActivityEndAll(entries);
  } else {
    buffer_data = (void*) first_entry.output->data();
    buffer_len = (size_t) first_entry.output->size();
  }

  // Do allreduce.
  timeline.ActivityStartAll(entries, MPI_ALLREDUCE);
  const void* sendbuf = entries.size() > 1 || first_entry.tensor->data() == first_entry.output->data()
                        ? MPI_IN_PLACE : first_entry.tensor->data();
  int op = MPI_Allreduce(sendbuf, buffer_data,
                         (int) num_elements,
                         mpi_context_->GetMPIDataType(first_entry.tensor),
                         mpi_context_->GetMPISumOp(first_entry.tensor->dtype()),
                         mpi_context_->GetMPICommunicator(Communicator::GLOBAL));
  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Allreduce failed, see MPI output for details.");
  }
  timeline.ActivityEndAll(entries);

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(buffer_data, entries);

    auto cuda_result = cudaStreamSynchronize(cuda_context_->streams[entries[0].device]);
    cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);

    timeline.ActivityEndAll(entries);
  }

  return Status::OK();
}

} // namespace common
} // namespace horovod
