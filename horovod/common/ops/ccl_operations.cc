// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
// Modifications copyright (C) 2019 Intel Corporation
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

#include "ccl_operations.h"

#include "../logging.h"

#define CCL_CALL(expr)                                                      \
  do {                                                                      \
        ccl_status_t status = expr;                                         \
        if (status != ccl_status_success)                                   \
        {                                                                   \
           throw std::runtime_error(__FUNCTION__ + std::string(" failed."));\
        }                                                                   \
  } while (0)


namespace horovod {
namespace common {

ccl_datatype_t GetCCLDataType(const std::shared_ptr<Tensor>& tensor) {
  switch (tensor->dtype()) {
  case HOROVOD_FLOAT32:
    return ccl_dtype_float;
  case HOROVOD_FLOAT64:
    return ccl_dtype_double;
  case HOROVOD_INT32:
    return ccl_dtype_int;
  case HOROVOD_INT64:
    return ccl_dtype_int64;
  default:
    throw std::logic_error("Type " + DataType_Name(tensor->dtype()) +
                           " is not supported in CCL.");
  }
}

void CCLContext::Init() {

  LOG(DEBUG) << "Background thread start";

  // Initialize CCL
  ccl_init();
}

void CCLContext::Finalize() {
  LOG(DEBUG) << "Background thread destroy";

  // Finalize CCL
  ccl_finalize();
}

CCLAllreduce::CCLAllreduce(CCLContext* ccl_context, HorovodGlobalState* global_state)
    : AllreduceOp(global_state), ccl_context_(ccl_context) {}

Status CCLAllreduce::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& first_entry = entries[0];

  void* buffer_data;
  size_t buffer_len;
  int64_t num_elements = NumElements(entries);

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
  }

  // Do allreduce.
  timeline.ActivityStartAll(entries, CCL_ALLREDUCE);
  const void* sendbuf = entries.size() > 1 || first_entry.tensor->data() == first_entry.output->data()
                        ? buffer_data : first_entry.tensor->data();
  ccl_request_t ccl_req;
  CCL_CALL(ccl_allreduce((void*)sendbuf, buffer_data, num_elements, GetCCLDataType(first_entry.tensor),
                         ccl_reduction_sum, nullptr /*attr*/, nullptr /*comm*/, nullptr /*stream*/, &ccl_req));
  CCL_CALL(ccl_wait(ccl_req));
  timeline.ActivityEndAll(entries);

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(buffer_data, entries);
    timeline.ActivityEndAll(entries);
  }

  return Status::OK();
}

bool CCLAllreduce::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

void CCLAllreduce::MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                             const TensorTableEntry& e, void* buffer_data_at_offset) {
  std::memcpy(buffer_data_at_offset, e.tensor->data(),
              (size_t) e.tensor->size());
}

void CCLAllreduce::MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                              const void* buffer_data_at_offset, TensorTableEntry& e) {
  std::memcpy((void*) e.output->data(), buffer_data_at_offset,
              (size_t) e.tensor->size());
}

CCLAllgather::CCLAllgather(CCLContext* ccl_context, HorovodGlobalState* global_state)
    : AllgatherOp(global_state), ccl_context_(ccl_context) {}

bool CCLAllgather::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

Status CCLAllgather::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& timeline = global_state_->timeline;

  // Sizes of subcomponents of each entry from all ranks
  auto** entry_component_sizes = new int64_t* [entries.size()];

  // Offset of each subcomponent of every entry in the final buffer after
  // allgatherv
  auto** entry_component_offsets = new int64_t* [entries.size()];

  int global_size = global_state_->controller->GetSize();
  auto* recvcounts = new int[global_size]();
  auto* displcmnts = new int[global_size]();

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    entry_component_sizes[ec] = new int64_t[global_size]();
    entry_component_offsets[ec] = new int64_t[global_size]();
  }

  auto& first_entry = entries[0];

  timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);
  Status status = AllocateOutput(entries, response, entry_component_sizes, recvcounts);
  if (!status.ok()) {
    /* Cleanup */
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

  SetDisplacements(recvcounts, displcmnts);
  SetEntryComponentOffsets(entries, entry_component_sizes, recvcounts, entry_component_offsets);

  int element_size = global_state_->controller->GetTypeSize(first_entry.tensor->dtype());

  const void* sendbuf = nullptr;
  void* buffer_data;
  int64_t total_num_elements = NumElements(entries);

  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    MemcpyInFusionBuffer(entries, displcmnts, element_size, buffer_data);
    timeline.ActivityEndAll(entries);
  } else {
    sendbuf = first_entry.tensor->data();
    buffer_data = (void*) first_entry.output->data();
  }

  auto* rcounts = new uint64_t[global_size]();
  for (unsigned int rc = 0; rc < global_size; rc++) {
    rcounts[rc] = recvcounts[rc] * element_size;
  }

  global_state_->timeline.ActivityStartAll(entries, CCL_ALLGATHER);
  ccl_request_t ccl_req;
  CCL_CALL(ccl_allgatherv(sendbuf != nullptr ? (void*)sendbuf : buffer_data,
           total_num_elements * element_size, buffer_data, rcounts, ccl_dtype_char,
           nullptr /*attr*/, nullptr /*comm*/, nullptr /*stream*/, &ccl_req));
  CCL_CALL(ccl_wait(ccl_req));
  global_state_->timeline.ActivityEndAll(entries);

  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(entry_component_offsets, entry_component_sizes,
                          buffer_data, element_size, entries);
    timeline.ActivityEndAll(entries);
  }

  delete[] rcounts;
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

CCLBroadcast::CCLBroadcast(CCLContext* ccl_context, HorovodGlobalState* global_state)
    : BroadcastOp(global_state), ccl_context_(ccl_context) {}

Status CCLBroadcast::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  assert(entries.size() == 1);
  auto e = entries[0];

  // On root rank, CCL_Bcast sends data, on other ranks it receives data.
  void* data_ptr;
  size_t size;
  if (global_state_->controller->GetRank() == e.root_rank) {
    data_ptr = (void*) e.tensor->data();
    size = e.tensor->size();
  } else {
    data_ptr = (void*) e.output->data();
    size = e.output->size();
  }

  global_state_->timeline.ActivityStartAll(entries, CCL_BCAST);
  ccl_request_t ccl_req;
  CCL_CALL(ccl_bcast(data_ptr, size, ccl_dtype_char, e.root_rank, nullptr /*attr*/,
                     nullptr /*comm*/, nullptr /*stream*/, &ccl_req));
  CCL_CALL(ccl_wait(ccl_req));
  global_state_->timeline.ActivityEndAll(entries);

  return Status::OK();
}

bool CCLBroadcast::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

} // namespace common
} // namespace horovod
