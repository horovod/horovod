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

#ifndef HOROVOD_CCL_OPERATIONS_H
#define HOROVOD_CCL_OPERATIONS_H

#include <iostream>
#include <pthread.h>

#include "ccl.h"

#include "collective_operations.h"
#include "../common.h"
#include "../global_state.h"

namespace horovod {
namespace common {

struct CCLContext {
  void Init();

  void Finalize();
};

class CCLAllreduce : public AllreduceOp {
public:
  CCLAllreduce(CCLContext* ccl_context, HorovodGlobalState* global_state);

  virtual ~CCLAllreduce() = default;

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  void MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                 const TensorTableEntry& e, void* buffer_data_at_offset) override;

  void MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                                  const void* buffer_data_at_offset, TensorTableEntry& e) override;

  CCLContext* ccl_context_;
};

class CCLAllgather : public AllgatherOp {
public:
  CCLAllgather(CCLContext* ccl_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  CCLContext* ccl_context_;
};

class CCLBroadcast : public BroadcastOp {
public:
  CCLBroadcast(CCLContext* ccl_context, HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;

protected:
  CCLContext* ccl_context_ ;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_CCL_OPERATIONS_H
