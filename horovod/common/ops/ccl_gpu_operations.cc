// Copyright (C) 2023 Intel CORPORATION. All rights reserved.
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

#include "ccl_gpu_operations.h"

#include "sycl/sycl_kernels.h"

namespace horovod {
namespace common {

std::mutex CCLGPUContext::GlobalMutex;

ccl::datatype GetCCLDataType(const std::shared_ptr<Tensor>& tensor) {
  switch (tensor->dtype()) {
  case HOROVOD_UINT8:
    return ccl::datatype::uint8;
  case HOROVOD_INT8:
    return ccl::datatype::int8;
  case HOROVOD_UINT16:
    return ccl::datatype::uint16;
  case HOROVOD_INT16:
    return ccl::datatype::int16;
  case HOROVOD_FLOAT16:
    return ccl::datatype::float16;
  case HOROVOD_FLOAT32:
    return ccl::datatype::float32;
  case HOROVOD_FLOAT64:
    return ccl::datatype::float64;
  case HOROVOD_INT32:
    return ccl::datatype::int32;
  case HOROVOD_INT64:
    return ccl::datatype::int64;
  default:
    throw std::logic_error("Type " + DataType_Name(tensor->dtype()) +
                           " is not supported in CCL.");
  }
}

inline void
CheckTensorTableEntry(const std::vector<TensorTableEntry>& entries) {
  if (entries.empty()) {
    throw std::runtime_error("TensorTableEntry is empty!");
  }
}

void CCLGPUContext::Initialize(HorovodGlobalState& state) {
  ccl::init();

  enable_cache = GetBoolEnvOrDefault(HOROVOD_CCL_CACHE, false);
  LOG(INFO) << "CCLGPUContext initialized: \n"
            << "enable_cache " << enable_cache << "\n";

  ccl_comms.resize(state.num_nccl_streams);
}

void CCLGPUContext::Finalize() { ccl_comms.clear(); }

CCLGPUOpContext::~CCLGPUOpContext() {
  ccl_streams.clear();
  ccl_devices.clear();
  ccl_contexts.clear();
}

void CCLGPUOpContext::InitCCLComm(const gpuStream_t& stream,
                                  const std::vector<TensorTableEntry>& entries,
                                  const std::vector<int32_t>& ccl_device_map) {
  CheckTensorTableEntry(entries);

  auto process_set_id = entries[0].process_set_id;
  auto& process_set = global_state_->process_set_table.Get(process_set_id);

  if (ccl_context_->ccl_comms[global_state_->current_nccl_stream].empty() ||
      !ccl_context_->ccl_comms[global_state_->current_nccl_stream].count(
          std::make_tuple(process_set_id, ccl_device_map))) {
    auto& timeline = global_state_->timeline;
    timeline.ActivityStartAll(entries, INIT_CCL);

    int ccl_rank, ccl_size;
    Communicator ccl_id_bcast_comm;
    PopulateCCLCommStrategy(ccl_rank, ccl_size, ccl_id_bcast_comm, process_set);

    if (ccl_rank == 0) {
      if (!kvs_) {
        kvs_ = ccl::create_main_kvs();
      }
      auto main_addr = kvs_->get_address();
      process_set.controller->Bcast((void*)main_addr.data(), main_addr.size(),
                                    0, ccl_id_bcast_comm);
    } else {
      ccl::kvs::address_type main_addr;
      process_set.controller->Bcast((void*)main_addr.data(), main_addr.size(),
                                    0, ccl_id_bcast_comm);
      kvs_ = ccl::create_kvs(main_addr);
    }

    auto queue = stream;
    {
      std::lock_guard<std::mutex> lock(CCLGPUContext::GlobalMutex);
      ccl_streams.push_back(ccl::create_stream(*queue));
      ccl_devices.push_back(ccl::create_device(queue->get_device()));
      ccl_contexts.push_back(ccl::create_context(queue->get_context()));
      // fill ccl_comms via creating communicator
      ccl_context_->ccl_comms[global_state_->current_nccl_stream].insert(
          {std::make_tuple(process_set_id, ccl_device_map),
           ccl4hvd{ccl_streams[0],
                   ccl::create_communicator(ccl_size, ccl_rank, ccl_devices[0],
                                            ccl_contexts[0], kvs_)}});
    }

    process_set.controller->Barrier(Communicator::GLOBAL);
    timeline.ActivityEndAll(entries);
  }
}

// helpers
void CCLGPUOpContext::PopulateCCLCommStrategy(int& ccl_rank, int& ccl_size,
                                              Communicator& ccl_id_bcast_comm,
                                              const ProcessSet& process_set) {
  if (communicator_type_ == Communicator::GLOBAL) {
    ccl_rank = process_set.controller->GetRank();
    ccl_size = process_set.controller->GetSize();
  } else if (communicator_type_ == Communicator::LOCAL) {
    ccl_rank = process_set.controller->GetLocalRank();
    ccl_size = process_set.controller->GetLocalSize();
  } else {
    throw std::logic_error("Communicator type " +
                           std::to_string(communicator_type_) +
                           " is not supported in CCL mode.");
  }
  ccl_id_bcast_comm = communicator_type_;
}

bool CCLGPUOpContext::IsEnabled(
    const std::vector<TensorTableEntry>& entries) const {
  return entries[0].device != CPU_DEVICE_ID;
}

ccl::communicator&
CCLGPUOpContext::GetCCLComm(const TensorTableEntry& entry,
                            const std::vector<int32_t>& devices) {
  return ccl_context_->ccl_comms[global_state_->current_nccl_stream]
      .at(std::make_tuple(entry.process_set_id, devices))
      .ccl_comm_;
}

ccl::stream&
CCLGPUOpContext::GetCCLStream(const TensorTableEntry& entry,
                              const std::vector<int32_t>& devices) {
  return ccl_context_->ccl_comms[global_state_->current_nccl_stream]
      .at(std::make_tuple(entry.process_set_id, devices))
      .ccl_stream_;
}

// Allreduce
Status CCLGPUAllreduce::Execute(std::vector<TensorTableEntry>& entries,
                                const Response& response) {
  CheckTensorTableEntry(entries);
  auto& first_entry = entries[0];

  gpu_op_context_.InitGPU(entries);
  gpu_op_context_.InitGPUQueue(entries, response);

  auto stream =
      gpu_context_
          ->streams[global_state_->current_nccl_stream][first_entry.device];
  ccl_op_context_.InitCCLComm(stream, entries, response.devices());

  WaitForData(entries);

  auto ccl_reduction_op = ccl::reduction::sum;
  double prescale_factor = response.prescale_factor();
  double postscale_factor = response.postscale_factor();

  if (response.reduce_op() == ReduceOp::AVERAGE) {
    ccl_reduction_op = ccl::reduction::sum;
    auto process_set_id = first_entry.process_set_id;
    auto& process_set = global_state_->process_set_table.Get(process_set_id);
    // Averaging happens via postscale_factor
    postscale_factor /= process_set.controller->GetSize();
  } else if (response.reduce_op() == ReduceOp::SUM) {
    ccl_reduction_op = ccl::reduction::sum;
  } else if (response.reduce_op() == ReduceOp::MIN) {
    ccl_reduction_op = ccl::reduction::min;
  } else if (response.reduce_op() == ReduceOp::MAX) {
    ccl_reduction_op = ccl::reduction::max;
  } else if (response.reduce_op() == ReduceOp::PRODUCT) {
    ccl_reduction_op = ccl::reduction::prod;
  } else {
    throw std::logic_error("Reduction op type not supported.");
  }

  const void* fused_input_data;
  void* buffer_data;
  size_t buffer_len;

  if (entries.size() > 1) {
    ScaleMemcpyInFusionBuffer(entries, fused_input_data, buffer_data,
                              buffer_len, prescale_factor);
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
                                MEMCPY_IN_FUSION_BUFFER,
                                *gpu_op_context_.stream);
    }
  } else {
    fused_input_data = first_entry.tensor->data();
    buffer_data = const_cast<void*>(first_entry.output->data());
    buffer_len = (size_t)first_entry.output->size();
    int64_t num_elements =
        buffer_len / DataType_Size(first_entry.tensor->dtype());
    if (prescale_factor != 1.0) {
      // Execute prescaling op
      ScaleBuffer(prescale_factor, entries, fused_input_data, buffer_data,
                  num_elements);
      fused_input_data = buffer_data; // for unfused, scale is done out of place
    }
  }

  // cache
  auto attr = ccl::create_operation_attr<ccl::allreduce_attr>();
  if (ccl_context_->enable_cache) {
    std::string match_id = "dt_" + DataType_Name(first_entry.tensor->dtype()) +
                           "_len_" + std::to_string(buffer_len);

    if (prescale_factor != 1.0) {
      match_id += "_prescale_" + std::to_string(prescale_factor);
    }
    if (postscale_factor != 1.0) {
      match_id += "_postscale_" + std::to_string(postscale_factor);
    }
    for (size_t idx = 0; idx < entries.size(); idx++) {
      match_id += "_" + entries[idx].tensor_name;
    }

    attr.set<ccl::operation_attr_id::match_id>(ccl::string_class(match_id));
    attr.set<ccl::operation_attr_id::to_cache>(true);

    LOG(DEBUG) << "CCLGPUAllreduce::Execute enable_cache"
               << " buffer_len: " << buffer_len << " recv_buf: " << buffer_data
               << " match_id: " << match_id;
  } else {
    attr.set<ccl::operation_attr_id::to_cache>(false);
  }

  // Do allreduce
  int64_t num_elements =
      buffer_len / DataType_Size(first_entry.tensor->dtype());
  LOG(DEBUG) << "Do CCLGPUAllreduce, number of elements: " << num_elements
             << ", dtype: " << DataType_Name(first_entry.tensor->dtype());

  CCLGPUContext::CallWithLock(CCLGPUContext::GlobalMutex, [&]() {
    ccl::allreduce(
        fused_input_data, buffer_data, (size_t)num_elements,
        GetCCLDataType(first_entry.tensor), ccl_reduction_op,
        ccl_op_context_.GetCCLComm(first_entry, response.devices()),
        ccl_op_context_.GetCCLStream(first_entry, response.devices()), attr)
        .wait();
  });

  if (global_state_->timeline.Initialized()) {
    gpu_context_->RecordEvent(gpu_op_context_.event_queue, CCL_ALLREDUCE,
                              *gpu_op_context_.stream);
  }

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    ScaleMemcpyOutFusionBuffer(buffer_data, buffer_len, postscale_factor,
                               entries);
    if (global_state_->timeline.Initialized()) {
      gpu_context_->RecordEvent(gpu_op_context_.event_queue,
                                MEMCPY_OUT_FUSION_BUFFER,
                                *gpu_op_context_.stream);
    }
  } else {
    if (postscale_factor != 1.0) {
      // Execute postscaling op
      ScaleBuffer(postscale_factor, entries, buffer_data, buffer_data,
                  num_elements);
    }
  }

  return gpu_op_context_.FinalizeGPUQueue(
      entries, true, ccl_op_context_.error_check_callback_);
}

bool CCLGPUAllreduce::Enabled(const ParameterManager& param_manager,
                              const std::vector<TensorTableEntry>& entries,
                              const Response& response) const {
  return ccl_op_context_.IsEnabled(entries);
}

void CCLGPUAllreduce::WaitForData(std::vector<TensorTableEntry>& entries) {
  if (global_state_->timeline.Initialized()) {
    // If timeline is initialized, need to use normal CPU syncing path
    HorovodOp::WaitForData(entries);
  } else {
    // Push events to set to deduplicate entries
    std::unordered_set<gpuEvent_t> event_set;
    for (auto& e : entries) {
      e.ready_event_list.PushEventsToSet(event_set);
    }

    for (auto ev : event_set) {
      ev.wait();
    }
  }
}

// Broadcast
Status CCLGPUBroadcast::Execute(std::vector<TensorTableEntry>& entries,
                                const Response& response) {
  CheckTensorTableEntry(entries);
  auto first_entry = entries[0];
  auto& process_set =
      global_state_->process_set_table.Get(first_entry.process_set_id);

  gpu_op_context_.InitGPU(entries);
  gpu_op_context_.InitGPUQueue(entries, response);

  auto stream =
      gpu_context_
          ->streams[global_state_->current_nccl_stream][first_entry.device];
  ccl_op_context_.InitCCLComm(stream, entries, response.devices());

  WaitForData(entries);

  // On root rank, ccl broadcast sends data, on other ranks it receives data.
  void* data_ptr;
  if (process_set.controller->GetRank() == first_entry.root_rank) {
    data_ptr = (void*)first_entry.tensor->data();
  } else {
    data_ptr = (void*)first_entry.output->data();
  }

  // cache
  auto attr = ccl::create_operation_attr<ccl::broadcast_attr>();
  CCLGPUContext::CallWithLock(CCLGPUContext::GlobalMutex, [&]() {
    ccl::broadcast(
        data_ptr,
        /* size */ first_entry.tensor->shape().num_elements() *
            DataType_Size(first_entry.tensor->dtype()),
        ccl::datatype::int8, first_entry.root_rank,
        ccl_op_context_.GetCCLComm(first_entry, response.devices()),
        ccl_op_context_.GetCCLStream(first_entry, response.devices()), attr)
        .wait();
  });

  if (global_state_->timeline.Initialized()) {
    gpu_context_->RecordEvent(gpu_op_context_.event_queue, CCL_BCAST,
                              *gpu_op_context_.stream);
  }

  return gpu_op_context_.FinalizeGPUQueue(
      entries, true, ccl_op_context_.error_check_callback_);
}

bool CCLGPUBroadcast::Enabled(const ParameterManager& param_manager,
                              const std::vector<TensorTableEntry>& entries,
                              const Response& response) const {
  return ccl_op_context_.IsEnabled(entries);
}

void CCLGPUBroadcast::WaitForData(std::vector<TensorTableEntry>& entries) {
  if (global_state_->timeline.Initialized()) {
    // If timeline is initialized, need to use normal CPU syncing path
    HorovodOp::WaitForData(entries);
  } else {
    // Push events to set to deduplicate entries
    std::unordered_set<gpuEvent_t> event_set;
    for (auto& e : entries) {
      e.ready_event_list.PushEventsToSet(event_set);
    }
    for (auto ev : event_set) {
      ev.wait();
    }
  }
}

} // namespace common
} // namespace horovod
