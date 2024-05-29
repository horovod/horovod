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

#include "gpu_operations.h"
#if HAVE_CUDA
#include "cuda/cuda_kernels.h"
#endif
#if HAVE_ROCM
#include "rocm/hip_kernels.h"
#endif

#include <thread>

namespace horovod {
namespace common {

GPUOpContext::GPUOpContext(GPUContext* context,
                           HorovodGlobalState* global_state)
    : gpu_context_(context), global_state_(global_state) {}

void GPUOpContext::InitGPU(const std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  gpu_context_->SetDevice(first_entry.device);

  // Ensure stream is in the map before executing reduction.
  gpuStream_t& stream =
      gpu_context_
          ->streams[global_state_->current_nccl_stream][first_entry.device];
  if (stream == nullptr) {
    gpu_context_->StreamCreate(&stream);
  }
}

void GPUOpContext::InitGPUQueue(const std::vector<TensorTableEntry>& entries,
                                const Response& response) {
  event_queue = std::queue<std::pair<std::string, Event>>();
  stream =
      &gpu_context_
           ->streams[global_state_->current_nccl_stream][entries[0].device];

  if (global_state_->timeline.Initialized()) {
    gpu_context_->RecordEvent(event_queue, QUEUE, *stream);
  }
}

Status GPUOpContext::FinalizeGPUQueue(
    std::vector<TensorTableEntry>& entries, bool free_host_buffer /*= true*/,
    const std::function<void()>& error_check_callback) {
  // Use completion marker via event because it's faster than
  // blocking gpuStreamSynchronize() in this thread.
  if (!global_state_->enable_async_completion) {
    gpu_context_->RecordEvent(event_queue, "", *stream);
  }

  auto& first_entry = entries[0];
  void* cpu_buffer = host_buffer;
  auto& evt_queue = event_queue;
  auto& timeline = global_state_->timeline;
  auto& gpu_context = gpu_context_;
  auto& global_state = global_state_;

  // Claim a std::shared_ptr to the fusion buffer to prevent its memory from
  // being reclaimed during finalization.
  auto fusion_buffer = global_state_->fusion_buffer.GetBuffer(
      first_entry.device, first_entry.context->framework(),
      global_state_->current_nccl_stream);

  bool elastic = global_state_->elastic_enabled;
  bool enable_async_completion = global_state_->enable_async_completion;
  auto current_stream = *stream;
  gpu_context_->finalizer_thread_pool.execute(
      [entries, first_entry, cpu_buffer, fusion_buffer, free_host_buffer,
       evt_queue, &timeline, &gpu_context, error_check_callback, elastic,
       enable_async_completion, current_stream, &global_state]() mutable {
        gpu_context->SetDevice(first_entry.device);

        Event event;
        bool gpu_evt_failed = false;
        std::string gpu_evt_err_msg;
        if (!enable_async_completion || timeline.Initialized()) {
          // If timeline is enabled, wait for events on CPU for accurate
          // timings.
          if (elastic) {
            try {
              gpu_context->WaitForEventsElastic(evt_queue, entries, timeline,
                                                error_check_callback);
            } catch (std::exception& e) {
              // notify background loop to exit and reinit rather than just
              // aborting the program
              global_state->shut_down = true;
              gpu_evt_failed = true;
              gpu_evt_err_msg = e.what();
            }
          } else {
            gpu_context->WaitForEvents(evt_queue, entries, timeline,
                                       error_check_callback);
          }
        } else {
          gpu_context->ClearEvents(evt_queue, entries, timeline,
                                   error_check_callback, elastic);
          event = gpu_context->RecordEvent(current_stream);
        }

        if (free_host_buffer && cpu_buffer != nullptr) {
          free(cpu_buffer);
        }

        Status status;
        if (gpu_evt_failed) {
          status = Status::UnknownError(gpu_evt_err_msg);
        } else {
          status = Status::OK();
          status.event = event;
        }

        for (auto& e : entries) {
          timeline.End(e.tensor_name, e.output);
          e.FinishWithCallback(status);
        }
        if (enable_async_completion) {
          gpu_context->ReleaseEvent(event);
        }
      });

  // Update current stream
  global_state_->current_nccl_stream =
      (global_state_->current_nccl_stream + 1) %
      global_state_->num_nccl_streams;

  return Status::InProgress();
}

GPUAllreduce::GPUAllreduce(GPUContext* context,
                           HorovodGlobalState* global_state)
    : AllreduceOp(global_state), gpu_context_(context),
      gpu_op_context_(context, global_state) {}

bool GPUAllreduce::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}

void GPUAllreduce::MemcpyInFusionBuffer(
    const std::vector<TensorTableEntry>& entries, const void*& fused_input_data,
    void*& buffer_data, size_t& buffer_len) {
  // Access the fusion buffer.
  auto& first_entry = entries[0];
  auto buffer = global_state_->fusion_buffer.GetBuffer(
      first_entry.device, first_entry.context->framework(),
      global_state_->current_nccl_stream);
  buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));

  if (global_state_->batch_d2d_memcopies) {
    int64_t offset = 0;
    int idx = 0;
    int count = 0;

    BatchedD2DParams d2d_params;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;

      // Set input/output pointers and sizes
      d2d_params.out[idx % BATCHED_D2D_CAPACITY] = buffer_data_at_offset;
      d2d_params.in[idx % BATCHED_D2D_CAPACITY] = (void*)e.tensor->data();
      d2d_params.sizes[idx % BATCHED_D2D_CAPACITY] = e.tensor->size();

      offset +=
          BATCHED_D2D_PADDING *
          ((e.tensor->size() + BATCHED_D2D_PADDING - 1) / BATCHED_D2D_PADDING);
      idx++;
      count++;

      if (idx % BATCHED_D2D_CAPACITY == 0 || idx == (int)entries.size()) {
        // Perform batched d2d memcpy
#if HAVE_CUDA
        BatchedD2DMemcpyCudaImpl(
            d2d_params, count,
            gpu_context_->streams[global_state_->current_nccl_stream]
                                 [first_entry.device]);
#elif HAVE_ROCM
        BatchedD2DMemcpyROCmImpl(
            d2d_params, count,
            gpu_context_->streams[global_state_->current_nccl_stream]
                                 [first_entry.device]);
#endif
        // TODO: https://github.com/horovod/horovod/issues/2230
        // gpu_context_->ErrorCheck("BatchedD2DMemcpyCudaImpl",
        // cudaGetLastError());
        count = 0;
      }
    }
    buffer_len = (size_t)offset;

  } else {
    int64_t offset = 0;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
      MemcpyEntryInFusionBuffer(entries, e, buffer_data_at_offset);
      offset += e.tensor->size();
    }

    buffer_len = (size_t)offset;
  }

  // Set the input data to originate from the buffer.
  fused_input_data = buffer_data;
}

void GPUAllreduce::ScaleMemcpyInFusionBuffer(
    const std::vector<TensorTableEntry>& entries, const void*& fused_input_data,
    void*& buffer_data, size_t& buffer_len, double scale_factor) {
  auto& first_entry = entries[0];
  // Access the fusion buffer.
  auto buffer = global_state_->fusion_buffer.GetBuffer(
      first_entry.device, first_entry.context->framework(),
      global_state_->current_nccl_stream);
  buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));

  if (global_state_->batch_d2d_memcopies) {
    int64_t offset = 0;
    int idx = 0;
    int count = 0;

    BatchedD2DParams d2d_params;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;

      // Set input/output pointers and sizes
      d2d_params.out[idx % BATCHED_D2D_CAPACITY] = buffer_data_at_offset;
      d2d_params.in[idx % BATCHED_D2D_CAPACITY] = (void*)e.tensor->data();
      d2d_params.sizes[idx % BATCHED_D2D_CAPACITY] = e.tensor->size();

      offset +=
          BATCHED_D2D_PADDING *
          ((e.tensor->size() + BATCHED_D2D_PADDING - 1) / BATCHED_D2D_PADDING);
      idx++;
      count++;

      if (idx % BATCHED_D2D_CAPACITY == 0 || idx == (int)entries.size()) {
        // Perform batched d2d memcpy
#if HAVE_CUDA
        BatchedScaledD2DMemcpyCudaImpl(
            d2d_params, count, scale_factor, first_entry.tensor->dtype(),
            gpu_context_->streams[global_state_->current_nccl_stream]
                                 [first_entry.device]);
#elif HAVE_ROCM
        BatchedScaledD2DMemcpyROCmImpl(
            d2d_params, count, scale_factor, first_entry.tensor->dtype(),
            gpu_context_->streams[global_state_->current_nccl_stream]
                                 [first_entry.device]);
#endif
        // TODO: https://github.com/horovod/horovod/issues/2230
        // gpu_context_->ErrorCheck("BatchedScaledD2DMemcpyCudaImpl",
        // cudaGetLastError());
        count = 0;
      }
    }
    buffer_len = (size_t)offset;

  } else {
    int64_t offset = 0;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
      MemcpyEntryInFusionBuffer(entries, e, buffer_data_at_offset);
      offset += e.tensor->size();
    }

    buffer_len = (size_t)offset;
    int64_t num_elements =
        buffer_len / DataType_Size(first_entry.tensor->dtype());
    if (scale_factor != 1.0) {
      ScaleBuffer(scale_factor, entries, buffer_data, buffer_data,
                  num_elements);
    }
  }

  // Set the input data to originate from the buffer.
  fused_input_data = buffer_data;
}

void GPUAllreduce::MemcpyEntryInFusionBuffer(
    const std::vector<TensorTableEntry>& entries, const TensorTableEntry& e,
    void* buffer_data_at_offset) {
  auto& first_entry = entries[0];
  gpu_context_->MemcpyAsyncD2D(
      buffer_data_at_offset, e.tensor->data(), (size_t)e.tensor->size(),
      gpu_context_
          ->streams[global_state_->current_nccl_stream][first_entry.device]);
}

void GPUAllreduce::MemcpyOutFusionBuffer(
    const void* buffer_data, std::vector<TensorTableEntry>& entries) {
  if (global_state_->batch_d2d_memcopies) {
    int64_t offset = 0;
    int idx = 0;
    int count = 0;

    BatchedD2DParams d2d_params;
    auto& first_entry = entries[0];
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;

      // Set input/output pointers and sizes
      d2d_params.out[idx % BATCHED_D2D_CAPACITY] = (void*)(e.output->data());
      d2d_params.in[idx % BATCHED_D2D_CAPACITY] = buffer_data_at_offset;
      d2d_params.sizes[idx % BATCHED_D2D_CAPACITY] = e.tensor->size();

      offset +=
          BATCHED_D2D_PADDING *
          ((e.tensor->size() + BATCHED_D2D_PADDING - 1) / BATCHED_D2D_PADDING);
      idx++;
      count++;

      if (idx % BATCHED_D2D_CAPACITY == 0 || idx == (int)entries.size()) {
        // Perform batched d2d memcpy
#if HAVE_CUDA
        BatchedD2DMemcpyCudaImpl(
            d2d_params, count,
            gpu_context_->streams[global_state_->current_nccl_stream]
                                 [first_entry.device]);
#elif HAVE_ROCM
        BatchedD2DMemcpyROCmImpl(
            d2d_params, count,
            gpu_context_->streams[global_state_->current_nccl_stream]
                                 [first_entry.device]);
#endif
        // TODO: https://github.com/horovod/horovod/issues/2230
        // gpu_context_->ErrorCheck("BatchedD2DMemcpyCudaImpl",
        // cudaGetLastError());
        count = 0;
      }
    }

  } else {
    int64_t offset = 0;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
      MemcpyEntryOutFusionBuffer(entries, buffer_data_at_offset, e);
      offset += e.tensor->size();
    }
  }
}

void GPUAllreduce::ScaleMemcpyOutFusionBuffer(
    void* buffer_data, size_t buffer_len, double scale_factor,
    std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];

  if (global_state_->batch_d2d_memcopies) {
    int64_t offset = 0;
    int idx = 0;
    int count = 0;

    BatchedD2DParams d2d_params;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;

      // Set input/output pointers and sizes
      d2d_params.out[idx % BATCHED_D2D_CAPACITY] = (void*)(e.output->data());
      d2d_params.in[idx % BATCHED_D2D_CAPACITY] = buffer_data_at_offset;
      d2d_params.sizes[idx % BATCHED_D2D_CAPACITY] = e.tensor->size();

      offset +=
          BATCHED_D2D_PADDING *
          ((e.tensor->size() + BATCHED_D2D_PADDING - 1) / BATCHED_D2D_PADDING);
      idx++;
      count++;

      if (idx % BATCHED_D2D_CAPACITY == 0 || idx == (int)entries.size()) {
        // Perform batched d2d memcpy
#if HAVE_CUDA
        BatchedScaledD2DMemcpyCudaImpl(
            d2d_params, count, scale_factor, first_entry.tensor->dtype(),
            gpu_context_->streams[global_state_->current_nccl_stream]
                                 [first_entry.device]);
#elif HAVE_ROCM
        BatchedScaledD2DMemcpyROCmImpl(
            d2d_params, count, scale_factor, first_entry.tensor->dtype(),
            gpu_context_->streams[global_state_->current_nccl_stream]
                                 [first_entry.device]);
#endif
        // TODO: https://github.com/horovod/horovod/issues/2230
        // gpu_context_->ErrorCheck("BatchedD2DMemcpyCudaImpl",
        // cudaGetLastError());
        count = 0;
      }
    }

  } else {
    int64_t num_elements =
        buffer_len / DataType_Size(first_entry.tensor->dtype());
    if (scale_factor != 1.0) {
      ScaleBuffer(scale_factor, entries, buffer_data, buffer_data,
                  num_elements);
    }

    int64_t offset = 0;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
      MemcpyEntryOutFusionBuffer(entries, buffer_data_at_offset, e);
      offset += e.tensor->size();
    }
  }
}

void GPUAllreduce::MemcpyEntryOutFusionBuffer(
    const std::vector<TensorTableEntry>& entries,
    const void* buffer_data_at_offset, TensorTableEntry& e) {
  auto& first_entry = entries[0];
  gpu_context_->MemcpyAsyncD2D(
      (void*)e.output->data(), buffer_data_at_offset, (size_t)e.tensor->size(),
      gpu_context_
          ->streams[global_state_->current_nccl_stream][first_entry.device]);
}

void GPUAllreduce::ScaleBuffer(double scale_factor,
                               const std::vector<TensorTableEntry>& entries,
                               const void* fused_input_data, void* buffer_data,
                               int64_t num_elements) {
  gpu_context_->ScaleBufferImpl(
      fused_input_data, buffer_data, num_elements, scale_factor,
      entries[0].tensor->dtype(),
      gpu_context_
          ->streams[global_state_->current_nccl_stream][entries[0].device]);
}

GPUAllgather::GPUAllgather(GPUContext* context,
                           HorovodGlobalState* global_state)
    : AllgatherOp(global_state), gpu_context_(context),
      gpu_op_context_(context, global_state) {}

bool GPUAllgather::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}


void GPUAllgather::MemcpyInFusionBuffer(
    const std::vector<TensorTableEntry>& entries, const int* displcmnts,
    int element_size, void*& buffer_data) {
  // Access first entry for retrieving context
  auto& first_entry = entries[0];
  auto buffer = global_state_->fusion_buffer.GetBuffer(
      first_entry.device, first_entry.context->framework(),
      global_state_->current_nccl_stream);
  buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));
  auto& process_set =
      global_state_->process_set_table.Get(first_entry.process_set_id);
  // offset incorporates rank padding via displacements
  int64_t offset = (int64_t)displcmnts[process_set.controller->GetRank()] *
                   (int64_t)element_size;

  if (global_state_->batch_d2d_memcopies) {
    int idx = 0;
    int count = 0;
    BatchedD2DParams d2d_params;
    auto& first_entry = entries[0];
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;

      // accumulate the buffered data
      d2d_params.out[idx % BATCHED_D2D_CAPACITY] = buffer_data_at_offset;
      d2d_params.in[idx % BATCHED_D2D_CAPACITY] = (void*)e.tensor->data();
      d2d_params.sizes[idx % BATCHED_D2D_CAPACITY] = e.tensor->size();


      offset += e.tensor->size();
      idx++;
      count++;

      if (idx % BATCHED_D2D_CAPACITY == 0 || idx == (int)entries.size()) {
        // do the batched d2d memcopies of all the accumulated entries
#if HAVE_CUDA
        BatchedD2DMemcpyCudaImpl(
            d2d_params, count,
            gpu_context_->streams[global_state_->current_nccl_stream]
                                 [first_entry.device]);
#elif HAVE_ROCM
        BatchedD2DMemcpyROCmImpl(
            d2d_params, count,
            gpu_context_->streams[global_state_->current_nccl_stream]
                                 [first_entry.device]);
#endif
        // (kvignesh142): Keeping in sync with all-reduce based implementation
        // TODO: https://github.com/horovod/horovod/issues/2230
        // gpu_context_->ErrorCheck("BatchedD2DMemcpyCudaImpl",
        // cudaGetLastError());
        count = 0;
      }
    }
  } else {
    // use the default AllgatherOps implementation
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
      MemcpyEntryInFusionBuffer(entries, e, buffer_data_at_offset);
      offset += e.tensor->size();
    }
  }
}

void GPUAllgather::MemcpyEntryInFusionBuffer(
    const std::vector<TensorTableEntry>& entries, const TensorTableEntry& e,
    void* buffer_data_at_offset) {
  auto& first_entry = entries[0];
  gpu_context_->MemcpyAsyncD2D(
      buffer_data_at_offset, e.tensor->data(), (size_t)e.tensor->size(),
      gpu_context_
          ->streams[global_state_->current_nccl_stream][first_entry.device]);
}

void GPUAllgather::MemcpyOutFusionBuffer(
    const int64_t* const* entry_component_offsets,
    const int64_t* const* entry_component_sizes, const void* buffer_data,
    int element_size, std::vector<TensorTableEntry>& entries) {

  if (global_state_->batch_d2d_memcopies) {

    int idx = 0;
    int count = 0;

    BatchedD2DParams d2d_params;
    auto& first_entry = entries[0];

    for (size_t ec = 0; ec < entries.size(); ec++) {
      auto& e = entries[ec];
      auto& process_set =
          global_state_->process_set_table.Get(e.process_set_id);
      int global_size = process_set.controller->GetSize();
      int64_t copy_offset = 0;
      for (int rc = 0; rc < global_size; rc++) {
        // entry_component_offsets includes rank padding
        int64_t entry_offset = entry_component_offsets[ec][rc] * element_size;
        int64_t entry_size = entry_component_sizes[ec][rc] * element_size;
        void* buffer_data_at_offset = (uint8_t*)buffer_data + entry_offset;

        // populate the d2d params for blocked mem copies
        d2d_params.out[idx % BATCHED_D2D_CAPACITY] =
            (int8_t*)e.output->data() + copy_offset;
        d2d_params.in[idx % BATCHED_D2D_CAPACITY] = buffer_data_at_offset;
        d2d_params.sizes[idx % BATCHED_D2D_CAPACITY] = entry_size;

        copy_offset += entry_size;
        idx++;
        count++;

        if (idx % BATCHED_D2D_CAPACITY == 0) {

#if HAVE_CUDA
          BatchedD2DMemcpyCudaImpl(
              d2d_params, count,
              gpu_context_->streams[global_state_->current_nccl_stream]
                                   [first_entry.device]);
#elif HAVE_ROCM
          BatchedD2DMemcpyROCmImpl(
              d2d_params, count,
              gpu_context_->streams[global_state_->current_nccl_stream]
                                   [first_entry.device]);
#endif
          // (kvignesh142): Keeping in sync with all-reduce based implementation
          // TODO: https://github.com/horovod/horovod/issues/2230
          // gpu_context_->ErrorCheck("BatchedD2DMemcpyCudaImpl",
          // cudaGetLastError());
          count = 0;
        }
      }
    }
    // handle any leftover entries
    if (count != 0) {
#if HAVE_CUDA
      BatchedD2DMemcpyCudaImpl(
          d2d_params, count,
          gpu_context_->streams[global_state_->current_nccl_stream]
                                [first_entry.device]);
#elif HAVE_ROCM
      BatchedD2DMemcpyROCmImpl(
          d2d_params, count,
          gpu_context_->streams[global_state_->current_nccl_stream]
                                [first_entry.device]);
#endif
      // (kvignesh142): Keeping in sync with all-reduce based implementation
      // TODO: https://github.com/horovod/horovod/issues/2230
      // gpu_context_->ErrorCheck("BatchedD2DMemcpyCudaImpl",
      // cudaGetLastError());
    }
  } else {
    // use the default AllgatherOps implementation
    for (size_t ec = 0; ec < entries.size(); ++ec) {
      auto& e = entries[ec];
      auto& process_set =
          global_state_->process_set_table.Get(e.process_set_id);
      int global_size = process_set.controller->GetSize();
      int64_t copy_offset = 0;
      for (int rc = 0; rc < global_size; ++rc) {
        // entry_component_offsets includes rank padding
        int64_t entry_offset = entry_component_offsets[ec][rc] * element_size;
        int64_t entry_size = entry_component_sizes[ec][rc] * element_size;
        const void* buffer_data_at_offset =
            (uint8_t*)buffer_data + entry_offset;
        MemcpyEntryOutFusionBuffer(entries, buffer_data_at_offset, e,
                                   copy_offset, entry_size);
        copy_offset += entry_size;
      }
    }
  }
}

void GPUAllgather::MemcpyEntryOutFusionBuffer(
    const std::vector<TensorTableEntry>& entries,
    const void* buffer_data_at_offset, TensorTableEntry& e,
    int64_t entry_offset, size_t entry_size) {
  auto& first_entry = entries[0];
  gpu_context_->MemcpyAsyncD2D(
      (int8_t*)e.output->data() + entry_offset, buffer_data_at_offset,
      entry_size,
      gpu_context_
          ->streams[global_state_->current_nccl_stream][first_entry.device]);
}

GPUBroadcast::GPUBroadcast(GPUContext* context,
                           HorovodGlobalState* global_state)
    : BroadcastOp(global_state), gpu_context_(context),
      gpu_op_context_(context, global_state) {}

bool GPUBroadcast::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}

GPUAlltoall::GPUAlltoall(GPUContext* context, HorovodGlobalState* global_state)
    : AlltoallOp(global_state), gpu_context_(context),
      gpu_op_context_(context, global_state) {}

bool GPUAlltoall::Enabled(const ParameterManager& param_manager,
                          const std::vector<TensorTableEntry>& entries,
                          const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}

GPUReducescatter::GPUReducescatter(GPUContext* context,
                                   HorovodGlobalState* global_state)
    : ReducescatterOp(global_state), gpu_context_(context),
      gpu_op_context_(context, global_state) {}

bool GPUReducescatter::Enabled(const ParameterManager& param_manager,
                               const std::vector<TensorTableEntry>& entries,
                               const Response& response) const {
  return entries[0].device != CPU_DEVICE_ID;
}

void GPUReducescatter::MemcpyEntryInFusionBuffer(const TensorTableEntry& e,
                                                 size_t entry_offset,
                                                 size_t entry_size,
                                                 void* buffer_data_at_offset) {
  void* tensor_data_at_offset = (uint8_t*)e.tensor->data() + entry_offset;
  gpu_context_->MemcpyAsyncD2D(
      buffer_data_at_offset, tensor_data_at_offset, entry_size,
      gpu_context_->streams[global_state_->current_nccl_stream][e.device]);
}

void GPUReducescatter::MemcpyEntryOutFusionBuffer(
    const void* buffer_data_at_offset, TensorTableEntry& e) {
  gpu_context_->MemcpyAsyncD2D(
      (void*)e.output->data(), buffer_data_at_offset, (size_t)e.output->size(),
      gpu_context_->streams[global_state_->current_nccl_stream][e.device]);
}

void GPUReducescatter::MemcpyInFusionBuffer(
    const std::vector<TensorTableEntry>& entries,
    const std::vector<std::vector<TensorShape>>& output_shapes,
    std::size_t element_size, void*& buffer_data, size_t& buffer_len) {
  if (global_state_->batch_d2d_memcopies) {
    // Access the fusion buffer.
    const auto& first_entry = entries[0];
    auto buffer = global_state_->fusion_buffer.GetBuffer(
        first_entry.device, first_entry.context->framework(),
        global_state_->current_nccl_stream);
    buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));

    size_t buffer_offset = 0;
    std::vector<size_t> entry_offsets(entries.size(), 0);

    size_t idx = 0;
    int count = 0;
    const size_t total_count = output_shapes.size() * entries.size();
    BatchedD2DParams d2d_params;
    for (const auto& rank_shapes : output_shapes) {
      for (size_t ec = 0; ec < entries.size(); ++ec) {
        auto& e = entries[ec];
        const auto& entry_shape = rank_shapes[ec];
        const size_t entry_offset = entry_offsets[ec];
        const size_t entry_size = entry_shape.num_elements() * element_size;
        void* buffer_data_at_offset = (uint8_t*)buffer_data + buffer_offset;
        void* tensor_data_at_offset = (uint8_t*)e.tensor->data() + entry_offset;

        // accumulate the buffered data
        d2d_params.out[idx % BATCHED_D2D_CAPACITY] = buffer_data_at_offset;
        d2d_params.in[idx % BATCHED_D2D_CAPACITY] = tensor_data_at_offset;
        d2d_params.sizes[idx % BATCHED_D2D_CAPACITY] = entry_size;

        entry_offsets[ec] += entry_size;
        buffer_offset += entry_size;
        idx++;
        count++;

        if (idx % BATCHED_D2D_CAPACITY == 0 || idx == total_count) {
          // do the batched d2d memcopies of all the accumulated entries
#if HAVE_CUDA
          BatchedD2DMemcpyCudaImpl(
              d2d_params, count,
              gpu_context_->streams[global_state_->current_nccl_stream]
                                   [first_entry.device]);
#elif HAVE_ROCM
          BatchedD2DMemcpyROCmImpl(
              d2d_params, count,
              gpu_context_->streams[global_state_->current_nccl_stream]
                                   [first_entry.device]);
#endif
          // TODO: https://github.com/horovod/horovod/issues/2230
          // gpu_context_->ErrorCheck("BatchedD2DMemcpyCudaImpl",
          // cudaGetLastError());
          count = 0;
        }
      }
    }
    buffer_len = buffer_offset;
  } else {
    // default implementation using GPUReducescatter::MemcpyEntryInFusionBuffer
    ReducescatterOp::MemcpyInFusionBuffer(entries, output_shapes, element_size,
                                          buffer_data, buffer_len);
  }
}

void GPUReducescatter::ScaleMemcpyInFusionBuffer(
    const std::vector<TensorTableEntry>& entries,
    const std::vector<std::vector<TensorShape>>& output_shapes,
    std::size_t element_size, void*& buffer_data, double scale_factor) {
  const auto& first_entry = entries[0];
  if (global_state_->batch_d2d_memcopies) {
    // Access the fusion buffer.
    auto buffer = global_state_->fusion_buffer.GetBuffer(
        first_entry.device, first_entry.context->framework(),
        global_state_->current_nccl_stream);
    buffer_data = const_cast<void*>(buffer->AccessData(first_entry.context));

    size_t buffer_offset = 0;
    std::vector<size_t> entry_offsets(entries.size(), 0);

    size_t idx = 0;
    int count = 0;
    const size_t total_count = output_shapes.size() * entries.size();
    BatchedD2DParams d2d_params;
    for (const auto& rank_shapes : output_shapes) {
      for (size_t ec = 0; ec < entries.size(); ++ec) {
        auto& e = entries[ec];
        const auto& entry_shape = rank_shapes[ec];
        const size_t entry_offset = entry_offsets[ec];
        const size_t entry_size = entry_shape.num_elements() * element_size;
        void* buffer_data_at_offset = (uint8_t*)buffer_data + buffer_offset;
        void* tensor_data_at_offset = (uint8_t*)e.tensor->data() + entry_offset;

        // accumulate the buffered data
        d2d_params.out[idx % BATCHED_D2D_CAPACITY] = buffer_data_at_offset;
        d2d_params.in[idx % BATCHED_D2D_CAPACITY] = tensor_data_at_offset;
        d2d_params.sizes[idx % BATCHED_D2D_CAPACITY] = entry_size;

        entry_offsets[ec] += entry_size;
        buffer_offset += entry_size;
        idx++;
        count++;

        if (idx % BATCHED_D2D_CAPACITY == 0 || idx == total_count) {
          // do the batched d2d memcopies of all the accumulated entries
#if HAVE_CUDA
          BatchedScaledD2DMemcpyCudaImpl(
              d2d_params, count, scale_factor, first_entry.tensor->dtype(),
              gpu_context_->streams[global_state_->current_nccl_stream]
                                   [first_entry.device]);
#elif HAVE_ROCM
          BatchedScaledD2DMemcpyROCmImpl(
              d2d_params, count, scale_factor, first_entry.tensor->dtype(),
              gpu_context_->streams[global_state_->current_nccl_stream]
                                   [first_entry.device]);
#endif
          // TODO: https://github.com/horovod/horovod/issues/2230
          // gpu_context_->ErrorCheck("BatchedScaledD2DMemcpyCudaImpl",
          // cudaGetLastError());
          count = 0;
        }
      }
    }
  } else {
    // default implementation using GPUReducescatter::MemcpyEntryInFusionBuffer
    size_t buffer_len;
    ReducescatterOp::MemcpyInFusionBuffer(entries, output_shapes, element_size,
                                          buffer_data, buffer_len);
    if (scale_factor != 1.0) {
      int64_t num_elements =
          buffer_len / DataType_Size(first_entry.tensor->dtype());
      ScaleBuffer(scale_factor, entries, buffer_data, buffer_data,
                  num_elements);
    }
  }
}


void GPUReducescatter::MemcpyOutFusionBuffer(
    const void* buffer_data, std::vector<TensorTableEntry>& entries) {
  if (global_state_->batch_d2d_memcopies) {
    int device = entries[0].device;
    int64_t offset = 0;
    size_t idx = 0;
    int count = 0;
    BatchedD2DParams d2d_params;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;

      // populate the d2d params for blocked mem copies
      d2d_params.out[idx % BATCHED_D2D_CAPACITY] = (void*)e.output->data();
      d2d_params.in[idx % BATCHED_D2D_CAPACITY] = buffer_data_at_offset;
      d2d_params.sizes[idx % BATCHED_D2D_CAPACITY] = (size_t)e.output->size();

      offset += e.output->size();
      idx++;
      count++;

      if (idx % BATCHED_D2D_CAPACITY == 0 || idx == entries.size()) {
        // do the batched d2d memcopies of all the accumulated entries
#if HAVE_CUDA
        BatchedD2DMemcpyCudaImpl(
            d2d_params, count,
            gpu_context_->streams[global_state_->current_nccl_stream][device]);
#elif HAVE_ROCM
        BatchedD2DMemcpyROCmImpl(
            d2d_params, count,
            gpu_context_->streams[global_state_->current_nccl_stream][device]);
#endif
        // TODO: https://github.com/horovod/horovod/issues/2230
        // gpu_context_->ErrorCheck("BatchedD2DMemcpyCudaImpl",
        // cudaGetLastError());
        count = 0;
      }
    }
  } else {
    // default implementation using GPUReducescatter::MemcpyEntryOutFusionBuffer
    ReducescatterOp::MemcpyOutFusionBuffer(buffer_data, entries);
  }
}

void GPUReducescatter::ScaleMemcpyOutFusionBuffer(
    const void* buffer_data, double scale_factor,
    std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  if (global_state_->batch_d2d_memcopies) {
    int device = entries[0].device;
    int64_t offset = 0;
    size_t idx = 0;
    int count = 0;
    BatchedD2DParams d2d_params;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;

      // populate the d2d params for blocked mem copies
      d2d_params.out[idx % BATCHED_D2D_CAPACITY] = (void*)e.output->data();
      d2d_params.in[idx % BATCHED_D2D_CAPACITY] = buffer_data_at_offset;
      d2d_params.sizes[idx % BATCHED_D2D_CAPACITY] = (size_t)e.output->size();

      offset += e.output->size();
      idx++;
      count++;

      if (idx % BATCHED_D2D_CAPACITY == 0 || idx == entries.size()) {
        // do the batched d2d memcopies of all the accumulated entries
#if HAVE_CUDA
        BatchedScaledD2DMemcpyCudaImpl(
            d2d_params, count, scale_factor, first_entry.tensor->dtype(),
            gpu_context_->streams[global_state_->current_nccl_stream][device]);
#elif HAVE_ROCM
        BatchedScaledD2DMemcpyROCmImpl(
            d2d_params, count, scale_factor, first_entry.tensor->dtype(),
            gpu_context_->streams[global_state_->current_nccl_stream][device]);
#endif
        // TODO: https://github.com/horovod/horovod/issues/2230
        // gpu_context_->ErrorCheck("BatchedD2DMemcpyCudaImpl", cudaGetLastError());
        count = 0;
      }
    }
  } else {
    // default implementation using GPUReducescatter::MemcpyEntryOutFusionBuffer
    ReducescatterOp::MemcpyOutFusionBuffer(buffer_data, entries);
    if (scale_factor != 1.0) {
      for (auto& e : entries) {
        ScaleBuffer(scale_factor, entries, e.output->data(),
                    const_cast<void*>(e.output->data()),
                    e.output->shape().num_elements());
      }
    }
  }
}

void GPUReducescatter::ScaleBuffer(double scale_factor,
                                   const std::vector<TensorTableEntry>& entries,
                                   const void* fused_input_data,
                                   void* buffer_data, int64_t num_elements) {
  gpu_context_->ScaleBufferImpl(
      fused_input_data, buffer_data, num_elements, scale_factor,
      entries[0].tensor->dtype(),
      gpu_context_
          ->streams[global_state_->current_nccl_stream][entries[0].device]);
}

} // namespace common
} // namespace horovod
