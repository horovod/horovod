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

#include "mlsl_operations.h"

#include "../logging.h"

namespace horovod {
namespace common {

MLSL::DataType GetMLSLDataType(const std::shared_ptr<Tensor> tensor) {
  switch (tensor->dtype()) {
  case HOROVOD_UINT8:
    return MLSL::DT_BYTE;
  case HOROVOD_FLOAT32:
    return MLSL::DT_FLOAT;
  case HOROVOD_FLOAT64:
    return MLSL::DT_DOUBLE;
  default:
    throw std::logic_error("Type " + DataType_Name(tensor->dtype()) +
                           " is not supported in MLSL.");
  }
}

void server_affinity_set(int affinity) {
  cpu_set_t cpuset;
  pthread_t current_thread = pthread_self();

  __CPU_ZERO_S(sizeof(cpu_set_t), &cpuset);
  __CPU_SET_S(affinity, sizeof(cpu_set_t), &cpuset);

  if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
    LOG(ERROR) << "setaffinity failed";
  }

  // Check if we set the affinity correctly
  if (pthread_getaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
    LOG(ERROR) << "sched_getaffinity failed";
  }

  for (int core_idx = 0; core_idx < __CPU_SETSIZE; core_idx++) {
    if (__CPU_ISSET_S(core_idx, sizeof(cpu_set_t), &cpuset)) {
      LOG(DEBUG) << "Background thread affinity " << core_idx;
    }
  }
}

void MLSLContext::Init() {
  char* hvd_mlsl_bg_thread_env = NULL;
  int bg_thread_affinity = 0;
  if ((hvd_mlsl_bg_thread_env = getenv("HOROVOD_MLSL_BGT_AFFINITY")) != NULL)
  {
      bg_thread_affinity = atoi(hvd_mlsl_bg_thread_env);
      server_affinity_set(bg_thread_affinity);
  }

  LOG(DEBUG) << "Background thread start";

  // Initialize MLSL
  MLSL::Environment::GetEnv().Init(NULL, NULL);
}

void MLSLContext::Setup(int size) {
  dist = MLSL::Environment::GetEnv().CreateDistribution(size, 1);
}

void MLSLContext::Finalize() {
  dist->Barrier(MLSL::GT_GLOBAL);
  LOG(DEBUG) << "Background thread comm destroy";

  // Destroy MLSL communicators
  MLSL::Environment::GetEnv().DeleteDistribution(dist);

  // Finalize MLSL
  MLSL::Environment::GetEnv().Finalize();
}

MLSLAllreduce::MLSLAllreduce(MLSLContext* mlsl_context, HorovodGlobalState* global_state)
    : AllreduceOp(global_state), mlsl_context_(mlsl_context) {}

Status MLSLAllreduce::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
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
  timeline.ActivityStartAll(entries, MLSL_ALLREDUCE);
  const void* sendbuf = entries.size() > 1 || first_entry.tensor->data() == first_entry.output->data()
                        ? buffer_data : first_entry.tensor->data();
  auto mlsl_req = mlsl_context_->dist->AllReduce((void*)sendbuf, buffer_data, num_elements,
                                                 GetMLSLDataType(first_entry.tensor),
                                                 MLSL::RT_SUM, MLSL::GT_DATA);

  try {
      MLSL::Environment::GetEnv().Wait(mlsl_req);
  } catch (...) {
      throw std::logic_error("MLSL_Allreduce failed.");
  }
  timeline.ActivityEndAll(entries);

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(buffer_data, entries);
    timeline.ActivityEndAll(entries);
  }

  return Status::OK();
}

bool MLSLAllreduce::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

void MLSLAllreduce::MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                             const TensorTableEntry& e, void* buffer_data_at_offset) {
  std::memcpy(buffer_data_at_offset, e.tensor->data(),
              (size_t) e.tensor->size());
}

void MLSLAllreduce::MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                              const void* buffer_data_at_offset, TensorTableEntry& e) {
  std::memcpy((void*) e.output->data(), buffer_data_at_offset,
              (size_t) e.tensor->size());
}

MLSLAllgather::MLSLAllgather(MLSLContext* mlsl_context, HorovodGlobalState* global_state)
    : AllgatherOp(global_state), mlsl_context_(mlsl_context) {}

bool MLSLAllgather::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

Status MLSLAllgather::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
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

  global_state_->timeline.ActivityStartAll(entries, MLSL_ALLGATHER);
  auto mlsl_req = mlsl_context_->dist->AllGatherv(sendbuf != nullptr ? (void*)sendbuf : buffer_data,
                                                  total_num_elements * element_size,
                                                  buffer_data, rcounts,
                                                  MLSL::DT_BYTE, MLSL::GT_DATA);
  try {
      MLSL::Environment::GetEnv().Wait(mlsl_req);
  } catch (...) {
      throw std::logic_error("MLSL_Allgather failed.");
  }
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

MLSLBroadcast::MLSLBroadcast(MLSLContext* mlsl_context, HorovodGlobalState* global_state)
    : BroadcastOp(global_state), mlsl_context_(mlsl_context) {}

Status MLSLBroadcast::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  assert(entries.size() == 1);
  auto e = entries[0];

  // On root rank, MLSL_Bcast sends data, on other ranks it receives data.
  void* data_ptr;
  size_t size;
  if (global_state_->controller->GetRank() == e.root_rank) {
    data_ptr = (void*) e.tensor->data();
    size = e.tensor->size();
  } else {
    data_ptr = (void*) e.output->data();
    size = e.output->size();
  }

  global_state_->timeline.ActivityStartAll(entries, MLSL_BCAST);
  auto mlsl_req = mlsl_context_->dist->Bcast(data_ptr, size, MLSL::DT_BYTE,
                                             e.root_rank, MLSL::GT_DATA);
  try {
      MLSL::Environment::GetEnv().Wait(mlsl_req);
  } catch (...) {
      throw std::logic_error("MLSL_Bcast failed.");
  }
  global_state_->timeline.ActivityEndAll(entries);

  return Status::OK();
}

bool MLSLBroadcast::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

} // namespace common
} // namespace horovod
