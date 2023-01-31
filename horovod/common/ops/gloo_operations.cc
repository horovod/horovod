// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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

#include "gloo_operations.h"

#include "gloo/allgather.h"
#include "gloo/allgatherv.h"
#include "gloo/allreduce.h"
#include "gloo/alltoallv.h"
#include "gloo/broadcast.h"
#include "gloo/reduce_scatter.h"
#include "gloo/math.h"
#include "gloo/types.h"

#include "../common.h"
#include "../global_state.h"

namespace horovod {
namespace common {

IGlooAlgorithms* GetAlgorithmsForType(DataType dtype,
                                      GlooContext* gloo_context) {
  switch (dtype) {
  case HOROVOD_UINT8:
    return new GlooAlgorithms<u_int8_t>(gloo_context);
  case HOROVOD_INT8:
    return new GlooAlgorithms<int8_t>(gloo_context);
  case HOROVOD_UINT16:
    return new GlooAlgorithms<u_int16_t>(gloo_context);
  case HOROVOD_INT16:
    return new GlooAlgorithms<int16_t>(gloo_context);
  case HOROVOD_INT32:
    return new GlooAlgorithms<int32_t>(gloo_context);
  case HOROVOD_INT64:
    return new GlooAlgorithms<int64_t>(gloo_context);
  case HOROVOD_FLOAT16:
    return new GlooAlgorithms<gloo::float16>(gloo_context);
  case HOROVOD_FLOAT32:
    return new GlooAlgorithms<float>(gloo_context);
  case HOROVOD_FLOAT64:
    return new GlooAlgorithms<double>(gloo_context);
  case HOROVOD_BOOL:
    return new GlooAlgorithms<bool>(gloo_context);
  default:
    throw std::logic_error("Type " + DataType_Name(dtype) +
                           " is not supported in Gloo mode.");
  }
}

template <typename T>
GlooAlgorithms<T>::GlooAlgorithms(GlooContext* gloo_context)
    : gloo_context_(gloo_context) {}

template <typename T>
void GlooAlgorithms<T>::Allreduce(void* buffer_data, int num_elements,
                                  ReduceOp reduce_op) {
  gloo::AllreduceOptions opts(gloo_context_->ctx);
  opts.setOutput<T>(static_cast<T*>(buffer_data), (size_t) num_elements);

  if (reduce_op == ReduceOp::SUM) {
    void (*func)(void*, const void*, const void*, size_t) = &::gloo::sum<T>;
    opts.setReduceFunction(gloo::AllreduceOptions::Func(func));
   } else if (reduce_op == ReduceOp::MIN) {
    void (*func)(void*, const void*, const void*, size_t) = &::gloo::min<T>;
    opts.setReduceFunction(gloo::AllreduceOptions::Func(func));
   } else if (reduce_op == ReduceOp::MAX) {
    void (*func)(void*, const void*, const void*, size_t) = &::gloo::max<T>;
    opts.setReduceFunction(gloo::AllreduceOptions::Func(func));
   } else if (reduce_op == ReduceOp::PRODUCT) {
    void (*func)(void*, const void*, const void*, size_t) = &::gloo::product<T>;
    opts.setReduceFunction(gloo::AllreduceOptions::Func(func));
   }

  gloo::allreduce(opts);
}

template <typename T>
void GlooAlgorithms<T>::Allgather(void* buffer_data, void* buffer_out,
                                  int* recvcounts, int* displcmnts) {
  // create count index
  std::vector<size_t> counts(recvcounts, recvcounts + gloo_context_->ctx->size);

  gloo::AllgathervOptions opts(gloo_context_->ctx);
  opts.setInput<T>(static_cast<T*>(buffer_data) +
                       displcmnts[gloo_context_->ctx->rank],
                   counts[gloo_context_->ctx->rank]);
  opts.setOutput<T>(static_cast<T*>(buffer_out), counts);

  gloo::allgatherv(opts);
}

template <typename T>
void GlooAlgorithms<T>::Broadcast(void* buffer_data, int num_elements,
                                  int root_rank) {
  gloo::BroadcastOptions opts(gloo_context_->ctx);
  opts.setRoot(root_rank);
  opts.setOutput<T>(static_cast<T*>(buffer_data), (size_t) num_elements);
  gloo::broadcast(opts);
}

template <typename T>
void GlooAlgorithms<T>::Alltoall(void* buffer_data, void* buffer_out,
                                 std::vector<int64_t>& sendcounts,
                                 std::vector<int64_t>& recvcounts) {
  gloo::AlltoallvOptions opts(gloo_context_->ctx);
  opts.setInput<T>(static_cast<T*>(buffer_data), sendcounts);
  opts.setOutput<T>(static_cast<T*>(buffer_out), recvcounts);

  gloo::alltoallv(opts);
}

template <>
void GlooAlgorithms<bool>::Reducescatter(void* buffer_data,
                                         std::vector<int>& recvcounts) {
  // Need to add this specialization for T=bool because
  // gloo::ReduceScatterHalvingDoubling does not support vector<bool>
  throw std::logic_error("GlooReducescatter does not support bool data.");
}

template <typename T>
void GlooAlgorithms<T>::Reducescatter(void* buffer_data,
                                      std::vector<int>& recvcounts) {
  std::vector<T*> ptrs{reinterpret_cast<T*>(buffer_data)};
  int num_elements = std::accumulate(recvcounts.begin(), recvcounts.end(), 0);
  gloo::ReduceScatterHalvingDoubling<T> rs_hd(gloo_context_->ctx, ptrs,
                                              num_elements, recvcounts);
  rs_hd.run();
}

template <typename T> int GlooAlgorithms<T>::ElementSize() const {
  return sizeof(T);
}

GlooAllreduce::GlooAllreduce(HorovodGlobalState* global_state)
    : AllreduceOp(global_state) {}

Status GlooAllreduce::Execute(std::vector<TensorTableEntry>& entries,
                              const Response& response) {
  assert(!entries.empty());
  WaitForData(entries);
  auto& first_entry = entries[0];
  auto& process_set =
      global_state_->process_set_table.Get(first_entry.process_set_id);
  auto& gloo_context = process_set.gloo_context;

  ReduceOp glooOp = ReduceOp::SUM;
  double prescale_factor = response.prescale_factor();
  double postscale_factor = response.postscale_factor();

  if (response.reduce_op() == ReduceOp::AVERAGE) {
    glooOp = ReduceOp::SUM;
    // Averaging happens via postscale_factor
    postscale_factor /= process_set.controller->GetSize();
  } else if (response.reduce_op() == ReduceOp::SUM) {
    glooOp = ReduceOp::SUM;
  } else if (response.reduce_op() == ReduceOp::MIN) {
    glooOp = ReduceOp::MIN;
  } else if (response.reduce_op() == ReduceOp::MAX) {
    glooOp = ReduceOp::MAX;
  } else if (response.reduce_op() == ReduceOp::PRODUCT) {
    glooOp = ReduceOp::PRODUCT;
  } else {
    throw std::logic_error("Reduction op type not supported.");
  }

  const void* fused_input_data;
  void* buffer_data;
  int num_elements = (int)NumElements(entries);

  // Copy memory into the fusion buffer.
  auto& timeline = global_state_->timeline;
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    size_t buffer_len;
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);
    timeline.ActivityEndAll(entries);
  } else {
    buffer_data = (void*)first_entry.output->data();
    std::memcpy(buffer_data, first_entry.tensor->data(),
                (size_t)first_entry.tensor->size());
    fused_input_data = buffer_data;
  }

  if (prescale_factor != 1.0) {
    // Execute prescaling op
    ScaleBuffer(prescale_factor, entries, fused_input_data, buffer_data, num_elements);
  }

  // Do allreduce.
  timeline.ActivityStartAll(entries, GLOO_ALLREDUCE);
  std::unique_ptr<IGlooAlgorithms> gloo_algos(
      GetAlgorithmsForType(first_entry.tensor->dtype(), &gloo_context));
  gloo_algos->Allreduce(buffer_data, num_elements, glooOp);
  timeline.ActivityEndAll(entries);

  if (postscale_factor != 1.0) {
    // Execute postscaling op
    ScaleBuffer(postscale_factor, entries, buffer_data, buffer_data, num_elements);
  }

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(buffer_data, entries);
    timeline.ActivityEndAll(entries);
  }

  return Status::OK();
}

bool GlooAllreduce::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return true;
}

GlooAllgather::GlooAllgather(HorovodGlobalState* global_state)
    : AllgatherOp(global_state) {}

bool GlooAllgather::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return true;
}

Status GlooAllgather::Execute(std::vector<TensorTableEntry>& entries,
                              const Response& response) {
  WaitForData(entries);

  assert(!entries.empty());
  auto& first_entry = entries[0];
  auto& process_set =
      global_state_->process_set_table.Get(first_entry.process_set_id);
  auto& gloo_context = process_set.gloo_context;
  auto& timeline = global_state_->timeline;

  // Sizes of subcomponents of each entry from all ranks
  auto** entry_component_sizes = new int64_t*[entries.size()];

  // Offset of each subcomponent of every entry in the final buffer after
  // allgatherv
  auto** entry_component_offsets = new int64_t*[entries.size()];

  int global_size = process_set.controller->GetSize();
  auto* recvcounts = new int[global_size]();
  auto* displcmnts = new int[global_size]();

  for (size_t ec = 0; ec < entries.size(); ++ec) {
    entry_component_sizes[ec] = new int64_t[global_size]();
    entry_component_offsets[ec] = new int64_t[global_size]();
  }


  timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);
  Status status = AllocateOutput(entries, response, entry_component_sizes);
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

  SetRecvcounts(entry_component_sizes, entries.size(), global_size, recvcounts);
  SetDisplacements(recvcounts, displcmnts, global_size);
  SetEntryComponentOffsets(entry_component_sizes, recvcounts, entries.size(),
                           global_size, entry_component_offsets);

  std::unique_ptr<IGlooAlgorithms> gloo_algos(
      GetAlgorithmsForType(first_entry.tensor->dtype(), &gloo_context));
  int element_size = gloo_algos->ElementSize();

  void* sendbuf = nullptr;
  void* buffer_data;

  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    MemcpyInFusionBuffer(entries, displcmnts, element_size, buffer_data);
    sendbuf = buffer_data;
    timeline.ActivityEndAll(entries);
  } else {
    // need to move input data to its corresponding location in the output
    sendbuf = (void*)first_entry.tensor->data();
    buffer_data = (void*)first_entry.output->data();
    int buffer_offset = displcmnts[gloo_context.ctx->rank] * element_size;
    std::memcpy((uint8_t*)buffer_data + buffer_offset, sendbuf,
                (size_t)first_entry.tensor->size());
    sendbuf = buffer_data;
  }

  // call gloo allgather api
  timeline.ActivityStartAll(entries, GLOO_ALLGATHER);
  gloo_algos->Allgather(sendbuf, buffer_data, recvcounts, displcmnts);
  timeline.ActivityEndAll(entries);

  // if multiple tensors are gathered, restore the sequence from output
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(entry_component_offsets, entry_component_sizes,
                          buffer_data, element_size, entries);
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

GlooBroadcast::GlooBroadcast(HorovodGlobalState* global_state)
    : BroadcastOp(global_state) {}

Status GlooBroadcast::Execute(std::vector<TensorTableEntry>& entries,
                              const Response& response) {
  WaitForData(entries);

  assert(entries.size() == 1);
  auto e = entries[0];
  auto& process_set = global_state_->process_set_table.Get(e.process_set_id);
  auto& gloo_context = process_set.gloo_context;
  // On root rank, MPI_Bcast sends data, on other ranks it receives data.
  // for gloo broadcast, only output needs to be set if inplace

  void* data_ptr;
  if (process_set.controller->GetRank() == e.root_rank) {
    data_ptr = (void*)e.tensor->data();
  } else {
    data_ptr = (void*)e.output->data();
  }

  global_state_->timeline.ActivityStartAll(entries, GLOO_BCAST);
  std::unique_ptr<IGlooAlgorithms> gloo_algos(
      GetAlgorithmsForType(e.tensor->dtype(), &gloo_context));
  gloo_algos->Broadcast(data_ptr, (int)e.tensor->shape().num_elements(),
                        e.root_rank);
  global_state_->timeline.ActivityEndAll(entries);

  return Status::OK();
}

bool GlooBroadcast::Enabled(const ParameterManager& param_manager,
                            const std::vector<TensorTableEntry>& entries,
                            const Response& response) const {
  return true;
}

GlooAlltoall::GlooAlltoall(HorovodGlobalState* global_state)
    : AlltoallOp(global_state) {}

Status GlooAlltoall::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  WaitForData(entries);

  assert(entries.size() == 1);
  auto e = entries[0];
  auto& process_set = global_state_->process_set_table.Get(e.process_set_id);
  auto& gloo_context = process_set.gloo_context;
  std::vector<int64_t> sdispls, rdispls;
  std::vector<int64_t> sendcounts, recvcounts;
  Status status = PrepareOutputAndParams(e, sdispls, rdispls, sendcounts, recvcounts);
  if (!status.ok()) {
    return status;
  }

  global_state_->timeline.ActivityStartAll(entries, MPI_ALLTOALL);

  std::unique_ptr<IGlooAlgorithms> gloo_algos(
      GetAlgorithmsForType(e.tensor->dtype(), &gloo_context));
  gloo_algos->Alltoall((void*)e.tensor->data(), (void*)e.output->data(),
                       sendcounts, recvcounts);

  global_state_->timeline.ActivityEndAll(entries);

  return Status::OK();
}

bool GlooAlltoall::Enabled(const ParameterManager& param_manager,
                           const std::vector<TensorTableEntry>& entries,
                           const Response& response) const {
  return true;
}

GlooReducescatter::GlooReducescatter(HorovodGlobalState* global_state)
    : ReducescatterOp(global_state) {}

Status GlooReducescatter::Execute(std::vector<TensorTableEntry>& entries,
                                  const Response& response) {
  WaitForData(entries);

  assert(!entries.empty());
  auto& first_entry = entries[0];
  auto& process_set =
      global_state_->process_set_table.Get(first_entry.process_set_id);
  auto& gloo_context = process_set.gloo_context;
  auto& timeline = global_state_->timeline;

  double prescale_factor = response.prescale_factor();
  double postscale_factor = response.postscale_factor();

  void* buffer_data = nullptr;
  int num_elements = (int)NumElements(entries);

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

  std::unique_ptr<IGlooAlgorithms> gloo_algos(
      GetAlgorithmsForType(first_entry.tensor->dtype(), &gloo_context));
  int element_size = gloo_algos->ElementSize();

  // Copy memory into the fusion buffer.
  timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
  if (entries.size() > 1) {
    size_t buffer_len;
    MemcpyInFusionBuffer(entries, output_shapes, element_size, buffer_data,
                         buffer_len);
  } else {
    // Allocating a temp buffer because the Reducescatter will be performed
    // in place and the output tensor will be smaller than the input.
    buffer_data = new uint8_t[first_entry.tensor->size()];
    std::memcpy(buffer_data, first_entry.tensor->data(),
                first_entry.tensor->size());
  }
  if (prescale_factor != 1.0) {
    // Execute prescaling op
    ScaleBuffer(prescale_factor, entries, buffer_data, buffer_data,
                num_elements);
  }
  timeline.ActivityEndAll(entries);

  // Call Gloo Reducescatter API
  timeline.ActivityStartAll(entries, GLOO_REDUCESCATTER);
  gloo_algos->Reducescatter(buffer_data, recvcounts);
  timeline.ActivityEndAll(entries);

  // Copy memory out of the fusion buffer. Optionally scale outputs in place.
  timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
  if (entries.size() > 1) {
    MemcpyOutFusionBuffer(buffer_data, entries);
  } else {
    void* output_pointer = const_cast<void*>(first_entry.output->data());
    std::memcpy(output_pointer, buffer_data, first_entry.output->size());
    delete[] reinterpret_cast<uint8_t*>(buffer_data);
  }
  timeline.ActivityEndAll(entries);
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

bool GlooReducescatter::Enabled(const ParameterManager& param_manager,
                                const std::vector<TensorTableEntry>& entries,
                                const Response& response) const {
#ifdef __linux__
  return true;
#else
  // On MacOS we cannot use the TCP backend of Gloo. The
  // ReduceScatterHalvingDoubling algorithm does not work with the libuv backend
  // that is available because functions like Pair::createSendBuffer() are not
  // implemented. https://github.com/facebookincubator/gloo/issues/257
  return false;
#endif
}

} // namespace common
} // namespace horovod
