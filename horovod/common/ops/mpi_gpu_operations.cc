// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
// Modifications copyright (C) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "mpi_gpu_operations.h"
#include "../mpi/mpi_context.h"

#if HAVE_CUDA
#include "cuda/cuda_kernels.h"
#endif
#if HAVE_ROCM
#include "rocm/hip_kernels.h"
#endif

namespace horovod {
namespace common {

MPI_GPUAllreduce::MPI_GPUAllreduce(GPUContext* gpu_context,
                                   HorovodGlobalState* global_state)
    : GPUAllreduce(gpu_context, global_state) {}

Status MPI_GPUAllreduce::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& first_entry = entries[0];
  auto& process_set =
      global_state_->process_set_table.Get(first_entry.process_set_id);
  const auto& mpi_context = process_set.mpi_context;

  gpu_op_context_.InitGPU(entries);

  WaitForData(entries);

  MPI_Op mpiOp = mpi_context.GetMPISumOp(first_entry.tensor->dtype());
  double prescale_factor = response.prescale_factor();
  double postscale_factor = response.postscale_factor();

  if (response.reduce_op() == ReduceOp::AVERAGE) {
    mpiOp = mpi_context.GetMPISumOp(first_entry.tensor->dtype());
    // Averaging happens via postscale_factor
    postscale_factor /= process_set.controller->GetSize();
  } else if (response.reduce_op() == ReduceOp::SUM) {
    mpiOp = mpi_context.GetMPISumOp(first_entry.tensor->dtype());
  } else if (response.reduce_op() == ReduceOp::MIN) {
    mpiOp = mpi_context.GetMPIMinOp(first_entry.tensor->dtype());
  } else if (response.reduce_op() == ReduceOp::MAX) {
    mpiOp = mpi_context.GetMPIMaxOp(first_entry.tensor->dtype());
  } else if (response.reduce_op() == ReduceOp::PRODUCT) {
    mpiOp = mpi_context.GetMPIProdOp(first_entry.tensor->dtype());
  } else {
    throw std::logic_error("Reduction op type not supported.");
  }

  const void* fused_input_data;
  void* buffer_data;
  size_t buffer_len;

  // Copy memory into the fusion buffer.
  auto& timeline = global_state_->timeline;
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);

    gpu_context_->StreamSynchronize(gpu_context_->streams[global_state_->current_nccl_stream][entries[0].device]);

    timeline.ActivityEndAll(entries);
  } else {
    fused_input_data = first_entry.tensor->data();
    buffer_data = (void*) first_entry.output->data();
    buffer_len = (size_t) first_entry.output->size();
  }

  int64_t num_elements = buffer_len / DataType_Size(first_entry.tensor->dtype());

  if (prescale_factor != 1.0) {
    // Execute prescaling op
    ScaleBuffer(prescale_factor, entries, fused_input_data, buffer_data, num_elements);
    fused_input_data = buffer_data; // for unfused, scale is done out of place
  }

  // Do allreduce.
  timeline.ActivityStartAll(entries, MPI_ALLREDUCE);
  const void* sendbuf = entries.size() > 1 || fused_input_data == buffer_data
                        ? MPI_IN_PLACE : fused_input_data;
  int op =
      MPI_Allreduce(sendbuf, buffer_data, (int)num_elements,
                    mpi_context.GetMPIDataType(first_entry.tensor),
                    mpiOp,
                    mpi_context.GetMPICommunicator(Communicator::GLOBAL));
  if (op != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Allreduce failed, see MPI output for details.");
  }
  timeline.ActivityEndAll(entries);

  if (postscale_factor != 1.0) {
    // Execute postscaling op
    ScaleBuffer(postscale_factor, entries, buffer_data, buffer_data, num_elements);
  }

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(buffer_data, entries);

    gpu_context_->StreamSynchronize(gpu_context_->streams[global_state_->current_nccl_stream][entries[0].device]);

    timeline.ActivityEndAll(entries);
  }

  return Status::OK();
}

MPI_GPUAllgather::MPI_GPUAllgather(GPUContext* gpu_context,
                                   HorovodGlobalState* global_state)
    : GPUAllgather(gpu_context, global_state) {}

Status MPI_GPUAllgather::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  assert(!entries.empty());
  auto& first_entry = entries[0];
  auto& process_set =
      global_state_->process_set_table.Get(first_entry.process_set_id);
  const auto& mpi_context = process_set.mpi_context;

  auto& timeline = global_state_->timeline;

  gpu_op_context_.InitGPU(entries);

  WaitForData(entries);

  // Sizes of subcomponents of each entry from all ranks
  auto** entry_component_sizes = new int64_t* [entries.size()];

  // Offset of each subcomponent of every entry in the final buffer after
  // allgatherv
  auto** entry_component_offsets = new int64_t* [entries.size()];

  int global_size = process_set.controller->GetSize();
  int rank = process_set.controller->GetRank();

  auto* recvcounts = new int[global_size]();
  auto* displcmnts = new int[global_size]();

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    entry_component_sizes[ec] = new int64_t[global_size]();
    entry_component_offsets[ec] = new int64_t[global_size]();
  }

  timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);
  Status status = AllocateOutput(entries, response, entry_component_sizes);
  if (!status.ok()) {
    for (size_t ec = 0; ec < entries.size(); ++ec) {
      delete[] entry_component_sizes[ec];
      delete[] entry_component_offsets[ec];
    }
    delete[] entry_component_sizes;
    delete[] entry_component_offsets;
    delete[] recvcounts;
    delete[] displcmnts;
    return status;
  }
  timeline.ActivityEndAll(entries);

  int element_size = mpi_context.GetMPITypeSize(first_entry.tensor->dtype());
  int padding_elements = 1;
  if (entries.size() > 1) {
    assert(BATCHED_D2D_PADDING % element_size == 0);
    padding_elements = BATCHED_D2D_PADDING / element_size;
  }

  SetRecvcounts(entry_component_sizes, entries.size(), global_size, recvcounts,
                padding_elements);
  SetDisplacements(recvcounts, displcmnts, global_size);
  SetEntryComponentOffsets(entry_component_sizes, recvcounts, entries.size(),
                           global_size, entry_component_offsets);

  const void* sendbuf = nullptr;
  void* buffer_data;
  int sendcount = recvcounts[rank];  // includes rank padding if appropriate

  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    MemcpyInFusionBuffer(entries, displcmnts, element_size, buffer_data);

    gpu_context_->StreamSynchronize(gpu_context_->streams[global_state_->current_nccl_stream][entries[0].device]);

    timeline.ActivityEndAll(entries);
  } else {
    sendbuf = first_entry.tensor->data();
    buffer_data = (void*) first_entry.output->data();
  }

  global_state_->timeline.ActivityStartAll(entries, MPI_ALLGATHER);
  auto dtype = mpi_context.GetMPIDataType(first_entry.tensor->dtype());
  int op = MPI_Allgatherv(sendbuf != nullptr ? sendbuf : MPI_IN_PLACE,
                          sendcount,
                          dtype,
                          buffer_data,
                          recvcounts,
                          displcmnts,
                          dtype,
                          mpi_context.GetMPICommunicator(Communicator::GLOBAL));
  if (op != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Allgatherv failed, see MPI output for details.");
  }
  global_state_->timeline.ActivityEndAll(entries);

  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(entry_component_offsets, entry_component_sizes,
                          buffer_data, element_size, entries);

    gpu_context_->StreamSynchronize(gpu_context_->streams[global_state_->current_nccl_stream][entries[0].device]);

    timeline.ActivityEndAll(entries);
  }

  delete[] recvcounts;
  delete[] displcmnts;

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    delete[] entry_component_sizes[ec];
    delete[] entry_component_offsets[ec];
  }
  delete[] entry_component_sizes;
  delete[] entry_component_offsets;

  return Status::OK();
}

MPI_GPUAlltoall::MPI_GPUAlltoall(GPUContext* gpu_context,
                                 HorovodGlobalState* global_state)
    : GPUAlltoall(gpu_context, global_state) {}

Status MPI_GPUAlltoall::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  assert(entries.size() == 1);

  gpu_op_context_.InitGPU(entries);

  WaitForData(entries);

  auto e = entries[0];
  auto& process_set = global_state_->process_set_table.Get(e.process_set_id);
  const auto& mpi_context = process_set.mpi_context;

  std::vector<int32_t> sdispls, rdispls;
  std::vector<int32_t> sendcounts, recvcounts;
  Status status = PrepareOutputAndParams(e, sdispls, rdispls, sendcounts, recvcounts);
  if (!status.ok()) {
    return status;
  }

  const void* sendbuf = e.tensor->data();
  void* buffer_data = (void*) e.output->data();
  global_state_->timeline.ActivityStartAll(entries, MPI_ALLTOALL);

  int op =
      MPI_Alltoallv(sendbuf, sendcounts.data(), sdispls.data(),
                    mpi_context.GetMPIDataType(e.tensor->dtype()), buffer_data,
                    recvcounts.data(), rdispls.data(),
                    mpi_context.GetMPIDataType(e.output->dtype()),
                    mpi_context.GetMPICommunicator(Communicator::GLOBAL));
  if (op != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Alltoallv failed, see MPI output for details.");
  }
  global_state_->timeline.ActivityEndAll(entries);

  return Status::OK();
}

MPI_GPUReducescatter::MPI_GPUReducescatter(GPUContext* gpu_context,
                                           HorovodGlobalState* global_state)
    : GPUReducescatter(gpu_context, global_state) {}

Status MPI_GPUReducescatter::Execute(std::vector<TensorTableEntry>& entries,
                                     const Response& response) {
  assert(!entries.empty());
  auto& first_entry = entries[0];
  auto& process_set =
      global_state_->process_set_table.Get(first_entry.process_set_id);
  const auto& mpi_context = process_set.mpi_context;

  auto& timeline = global_state_->timeline;

  gpu_op_context_.InitGPU(entries);

  WaitForData(entries);

  double prescale_factor = response.prescale_factor();
  double postscale_factor = response.postscale_factor();

  void* fusion_buffer = nullptr;
  const void* sendbuf = nullptr;
  void* recvbuf = nullptr;

  int global_rank = process_set.controller->GetRank();
  int global_size = process_set.controller->GetSize();
  auto output_shapes = ComputeOutputShapes(entries, global_size);
  std::vector<int> recvcounts = ComputeReceiveCounts(output_shapes);

  timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);
  Status status = AllocateOutput(entries, output_shapes[global_rank]);
  if (!status.ok()) {
    return status;
  }
  timeline.ActivityEndAll(entries);

  // Copy memory into the fusion buffer. Execute prescaling op if necessary.
  if (entries.size() > 1 || prescale_factor != 1.0) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    int element_size = mpi_context.GetMPITypeSize(first_entry.tensor->dtype());
    size_t buffer_len;
    MemcpyInFusionBuffer(entries, output_shapes, element_size, fusion_buffer,
                         buffer_len);
    if (prescale_factor != 1.0) {
      // Prescale in place on the fusion buffer.
      int64_t num_elements =
          buffer_len / DataType_Size(first_entry.tensor->dtype());
      ScaleBuffer(prescale_factor, entries, fusion_buffer, fusion_buffer,
                  num_elements);
      if (entries.size() == 1) {
        // Unfused prescaled: Send from fusion buffer, receive at output tensor
        sendbuf = fusion_buffer;
        recvbuf = (void*)first_entry.output->data();
      }
    }
    if (entries.size() > 1) {
      // Fused: MPI_Reduce_scatter in place
      sendbuf = nullptr;
      recvbuf = fusion_buffer;
    }
    gpu_context_->StreamSynchronize(
        gpu_context_
            ->streams[global_state_->current_nccl_stream][first_entry.device]);

    timeline.ActivityEndAll(entries);
  } else {
    // Unfused without prescaling
    sendbuf = first_entry.tensor->data();
    recvbuf = (void*)first_entry.output->data();
  }

  // Do reducescatter.
  timeline.ActivityStartAll(entries, MPI_REDUCESCATTER);
  int op = MPI_Reduce_scatter(
      sendbuf != nullptr ? sendbuf : MPI_IN_PLACE, recvbuf,
      recvcounts.data(), mpi_context.GetMPIDataType(first_entry.tensor),
      mpi_context.GetMPISumOp(first_entry.tensor->dtype()),
      mpi_context.GetMPICommunicator(Communicator::GLOBAL));
  if (op != MPI_SUCCESS) {
    throw std::runtime_error(
        "MPI_Reduce_scatter failed, see MPI output for details.");
  }
  timeline.ActivityEndAll(entries);

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(fusion_buffer, entries);

    gpu_context_->StreamSynchronize(
        gpu_context_
            ->streams[global_state_->current_nccl_stream][first_entry.device]);

    timeline.ActivityEndAll(entries);
  }
  if (postscale_factor != 1.0) {
    // Execute postscaling ops
    for (auto& e : entries) {
      ScaleBuffer(postscale_factor, entries, e.output->data(),
                  const_cast<void*>(e.output->data()),
                  e.output->shape().num_elements());
    }
  }

  return Status::OK();
}

} // namespace common
} // namespace horovod
