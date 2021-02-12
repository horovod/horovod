// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
// Modifications copyright (C) 2020 Intel Corporation
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

#ifndef HOROVOD_CCL_OPERATIONS_H
#define HOROVOD_CCL_OPERATIONS_H

#include <iostream>
#include <pthread.h>

#include "oneapi/ccl.hpp"

#include "../common.h"
#include "../global_state.h"
#include "../logging.h"
#include "collective_operations.h"

namespace horovod {
namespace common {

// Context used for internal machinery
class CCLOpContext;

// CCL Context used to control CCL as a whole from the outside
class CCLContext {
public:
  CCLContext();
  void Initialize();
  void Finalize();
  bool IsInited() { return opctxt_ != nullptr; }

  bool enable_cache;

  CCLOpContext* opctxt_;

private:
  CCLOpContext* NewOpContext();
};

// Common operation base class
// mostly for code-reuse/unification
template <typename T> struct CCLOp : public T {
  CCLOp(CCLContext* ccl_context, HorovodGlobalState* global_state)
      : T(global_state), ccl_context_(ccl_context) {}

  virtual ~CCLOp() = default;

protected:
  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override {
    return this->ccl_context_ && entries.front().device == CPU_DEVICE_ID &&
           this->ccl_context_->IsInited();
  }

  CCLContext* ccl_context_;
};

class CCLAllreduce : public CCLOp<AllreduceOp> {
public:
  using CCLOp<AllreduceOp>::CCLOp;

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

protected:
  void MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                 const TensorTableEntry& e,
                                 void* buffer_data_at_offset) override;

  void MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                  const void* buffer_data_at_offset,
                                  TensorTableEntry& e) override;

  void ScaleBuffer(double scale_factor,
                   const std::vector<TensorTableEntry>& entries,
                   const void* fused_input_data, void* buffer_data,
                   int64_t num_elements) override;
};

class CCLAllgather : public CCLOp<AllgatherOp> {
public:
  using CCLOp<AllgatherOp>::CCLOp;

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

protected:
  virtual void
  MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                            const TensorTableEntry& e,
                            void* buffer_data_at_offset) override;

  virtual void
  MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                             const void* buffer_data_at_offset,
                             TensorTableEntry& e, int64_t entry_offset,
                             size_t entry_size) override;
};

class CCLBroadcast : public CCLOp<BroadcastOp> {
public:
  using CCLOp<BroadcastOp>::CCLOp;

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;
};

class CCLAlltoall : public CCLOp<AlltoallOp> {
public:
  using CCLOp<AlltoallOp>::CCLOp;

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_CCL_OPERATIONS_H
