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

#ifndef HOROVOD_NCCL_OPERATIONS_H
#define HOROVOD_NCCL_OPERATIONS_H

#if HAVE_CUDA
#include <nccl.h>
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 7, 0)
#define NCCL_P2P_SUPPORTED
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 10, 0)
#define NCCL_AVG_SUPPORTED
#endif
#elif HAVE_ROCM
#include <rccl.h>
#define NCCL_P2P_SUPPORTED
#endif

#include "../hashes.h"
#include "gpu_operations.h"

#include <functional>

namespace horovod {
namespace common {

ncclDataType_t GetNCCLDataType(const std::shared_ptr<Tensor> tensor);

struct NCCLContext {
  // indexed by [nccl stream][{process set id, device id vector}]
  std::vector<
      std::unordered_map<std::tuple<int32_t, std::vector<int32_t>>, ncclComm_t>>
      nccl_comms;

  void ErrorCheck(std::string op_name, ncclResult_t nccl_result,
                  ncclComm_t& nccl_comm);

  void ShutDown();

  bool elastic;
};

class NCCLOpContext {
public:
  NCCLOpContext(NCCLContext* nccl_context, HorovodGlobalState* global_state,
                Communicator communicator_type)
      : nccl_comm_(nullptr),
        error_check_callback_(std::bind(&NCCLOpContext::AsyncErrorCheck, this)),
        nccl_context_(nccl_context), global_state_(global_state),
        communicator_type_(communicator_type){};

  void InitNCCLComm(const std::vector<TensorTableEntry>& entries,
                    const std::vector<int32_t>& nccl_device_map);

  void AsyncErrorCheck();

  ncclComm_t* nccl_comm_;
  std::function<void()> error_check_callback_;

private:
  void PopulateNCCLCommStrategy(int& nccl_rank, int& nccl_size,
                                Communicator& nccl_id_bcast_comm,
                                const ProcessSet& process_set);

  NCCLContext* nccl_context_;
  HorovodGlobalState* global_state_;
  Communicator communicator_type_;
};

class NCCLAllreduce : public GPUAllreduce {
public:
  NCCLAllreduce(NCCLContext* nccl_context, GPUContext* gpu_context,
                HorovodGlobalState* global_state,
                Communicator communicator_type = Communicator::GLOBAL)
      : GPUAllreduce(gpu_context, global_state), nccl_context_(nccl_context),
        nccl_op_context_(nccl_context, global_state, communicator_type),
        global_state_(global_state){};

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

protected:
  void WaitForData(std::vector<TensorTableEntry>& entries) override;

  NCCLContext* nccl_context_;
  NCCLOpContext nccl_op_context_;
  HorovodGlobalState* global_state_;
};

class NCCLBroadcast : public GPUBroadcast {
public:
  NCCLBroadcast(NCCLContext* nccl_context, GPUContext* gpu_context,
                HorovodGlobalState* global_state)
      : GPUBroadcast(gpu_context, global_state), nccl_context_(nccl_context),
        nccl_op_context_(nccl_context, global_state, Communicator::GLOBAL),
        global_state_(global_state){};

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

protected:
  void WaitForData(std::vector<TensorTableEntry>& entries) override;

  NCCLContext* nccl_context_;
  NCCLOpContext nccl_op_context_;
  HorovodGlobalState* global_state_;
};

class NCCLAlltoall : public GPUAlltoall {
public:
  NCCLAlltoall(NCCLContext* nccl_context, GPUContext* gpu_context,
               HorovodGlobalState* global_state)
      : GPUAlltoall(gpu_context, global_state), nccl_context_(nccl_context),
        nccl_op_context_(nccl_context, global_state, Communicator::GLOBAL),
        global_state_(global_state){};

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

protected:
  void WaitForData(std::vector<TensorTableEntry>& entries) override;

  template <typename T>
  Status PrepareOutputAndParams(TensorTableEntry& e,
                                std::vector<T>& sdispls,
                                std::vector<T>& rdispls,
                                std::vector<T>& sendcounts,
                                std::vector<T>& recvcounts) {
    auto& process_set = global_state_->process_set_table.Get(e.process_set_id);
    auto world_size = process_set.controller->GetSize();

    const auto& splits = e.splits;
    std::vector<int32_t> recvsplits;

    process_set.controller->AlltoallGetRecvSplits(splits, recvsplits);

    // Every tensor participating in Alltoall operation may have different
    // first dimension size, but the rest of dimensions are same for all
    // tensors.  Here we get shape of tensor sliced by first dimension.
    TensorShape slice_shape;
    for (int i = 1; i < e.tensor->shape().dims(); ++i) {
      slice_shape.AddDim(e.tensor->shape().dim_size(i));
    }
    int64_t slice_num_elements = slice_shape.num_elements();

    // Prepare send/recvcounts and displacements for Alltoallv
    sdispls.resize(world_size);
    rdispls.resize(world_size);
    sendcounts.resize(world_size);
    recvcounts.resize(world_size);

    size_t output_first_dim = 0;
    for (int i = 0; i < world_size; ++i) {
      sendcounts[i] = splits[i] * slice_num_elements;
      recvcounts[i] = recvsplits[i] * slice_num_elements;
      output_first_dim += recvsplits[i];
    }

    for (int i = 1; i < world_size; ++i) {
      sdispls[i] = sdispls[i-1] + sendcounts[i-1];
      rdispls[i] = rdispls[i-1] + recvcounts[i-1];
    }

    // Allocate output
    TensorShape output_shape;
    output_shape.AddDim(output_first_dim);
    output_shape.AppendShape(slice_shape);

    std::shared_ptr<ReadyEvent> event;
    Status status = e.context->AllocateOutput(output_shape, &e.output, &event);
    if (!status.ok()) {
      LOG(WARNING)
          << "NCCLAlltoall::PrepareOutputAndParams failed to allocate output: "
          << status.reason();
      return status;
    }

    // Add event dependency for output allocation to stream
    if (event) {
      HVD_GPU_CHECK(gpuStreamWaitEvent(*gpu_op_context_.stream, event->event(), 0));
    }

    // Allocate and fill received_splits output
    TensorShape received_splits_shape;
    received_splits_shape.AddDim(recvsplits.size());

    std::shared_ptr<ReadyEvent> revent;
    Status rstatus = e.context->AllocateOutput(1, received_splits_shape,
                                               &e.received_splits,
                                               &revent);
    if (!rstatus.ok()) {
      LOG(WARNING) << "NCCLAlltoall::PrepareOutputAndParams failed to allocate "
                      "received_splits: "
                   << status.reason();
      return rstatus;
    }

    // Add event dependency for received_splits allocation to stream
    if (revent) {
      HVD_GPU_CHECK(gpuStreamWaitEvent(*gpu_op_context_.stream, revent->event(), 0));
    }

    auto* target_pointer = reinterpret_cast<int32_t*>(
        const_cast<void*>(e.received_splits->data()));
    std::copy(recvsplits.cbegin(), recvsplits.cend(), target_pointer);

    return Status::OK();
  }

  NCCLContext* nccl_context_;
  NCCLOpContext nccl_op_context_;
  HorovodGlobalState* global_state_;
};

#if HAVE_MPI
class NCCLHierarchicalAllreduce : public NCCLAllreduce {
public:
  NCCLHierarchicalAllreduce(NCCLContext* nccl_context, GPUContext* gpu_context,
                            HorovodGlobalState* global_state)
      : NCCLAllreduce(nccl_context, gpu_context, global_state,
                      Communicator::LOCAL){};

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  void WaitForData(std::vector<TensorTableEntry>& entries) override;

private:
  MPIContext* mpi_context_;
};
#endif

class NCCLTorusAllreduce : public GPUAllreduce {
public:
  NCCLTorusAllreduce(NCCLContext* local_nccl_context, NCCLContext* cross_nccl_context,
                     GPUContext* gpu_context, HorovodGlobalState* global_state)
      : GPUAllreduce(gpu_context, global_state),
        local_nccl_context_(local_nccl_context),
        cross_nccl_context_(cross_nccl_context),
        local_nccl_op_context_(local_nccl_context, global_state, Communicator::LOCAL),
        cross_nccl_op_context_(cross_nccl_context, global_state, Communicator::CROSS),
        global_state_(global_state){};

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  void WaitForData(std::vector<TensorTableEntry>& entries) override;

  NCCLContext* local_nccl_context_;
  NCCLOpContext local_nccl_op_context_;
  NCCLContext* cross_nccl_context_;
  NCCLOpContext cross_nccl_op_context_;
  HorovodGlobalState* global_state_;
};

class NCCLAllgather : public GPUAllgather {
public:
  NCCLAllgather(NCCLContext* nccl_context, GPUContext* gpu_context,
                HorovodGlobalState* global_state)
      : GPUAllgather(gpu_context, global_state), nccl_context_(nccl_context),
        nccl_op_context_(nccl_context, global_state, Communicator::GLOBAL),
        global_state_(global_state) {}

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  Status AllocateOutput(std::vector<TensorTableEntry>& entries,
                        const Response& response,
                        int64_t**& entry_component_sizes) override;

  void WaitForData(std::vector<TensorTableEntry>& entries) override;

  NCCLContext* nccl_context_;
  NCCLOpContext nccl_op_context_;
  HorovodGlobalState* global_state_;
};

class NCCLReducescatter : public GPUReducescatter {
public:
  NCCLReducescatter(NCCLContext* nccl_context, GPUContext* gpu_context,
                    HorovodGlobalState* global_state)
      : GPUReducescatter(gpu_context, global_state),
        nccl_context_(nccl_context),
        nccl_op_context_(nccl_context, global_state, Communicator::GLOBAL),
        global_state_(global_state) {}

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  Status AllocateOutput(std::vector<TensorTableEntry>& entries,
                        const std::vector<TensorShape>& output_shapes) override;

  void WaitForData(std::vector<TensorTableEntry>& entries) override;

  NCCLContext* nccl_context_;
  NCCLOpContext nccl_op_context_;
  HorovodGlobalState* global_state_;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_NCCL_OPERATIONS_H
