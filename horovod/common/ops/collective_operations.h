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

#ifndef HOROVOD_COLLECTIVE_OPERATIONS_H
#define HOROVOD_COLLECTIVE_OPERATIONS_H

#include <iostream>

#include "../common.h"
#include "../controller.h"
#include "../global_state.h"
#include "../operations.h"
#include "../parameter_manager.h"

namespace horovod {
namespace common {

class HorovodOp {
public:
  HorovodOp(HorovodGlobalState* global_state);

  virtual Status Execute(std::vector<TensorTableEntry>& entries,
                         const Response& response) = 0;

protected:
  int64_t NumElements(std::vector<TensorTableEntry>& entries);

  HorovodGlobalState* global_state_;
};

class AllreduceOp : public HorovodOp {
public:
  AllreduceOp(HorovodGlobalState* global_state);

  virtual ~AllreduceOp() = default;

  virtual Status Execute(std::vector<TensorTableEntry>& entries,
                         const Response& response) = 0;

  virtual bool Enabled(const ParameterManager& param_manager,
                       const std::vector<TensorTableEntry>& entries,
                       const Response& response) const = 0;

protected:
  virtual void
  MemcpyInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                       const void*& fused_input_data, void*& buffer_data,
                       size_t& buffer_len);

  virtual void MemcpyOutFusionBuffer(const void* buffer_data,
                                     std::vector<TensorTableEntry>& entries);

  virtual void
  MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                            const TensorTableEntry& e,
                            void* buffer_data_at_offset);

  virtual void
  MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                             const void* buffer_data_at_offset,
                             TensorTableEntry& e);
};

class AllgatherOp : public HorovodOp {
public:
  AllgatherOp(HorovodGlobalState* global_state);

  virtual ~AllgatherOp() = default;

  virtual Status Execute(std::vector<TensorTableEntry>& entries,
                         const Response& response) = 0;

  virtual bool Enabled(const ParameterManager& param_manager,
                       const std::vector<TensorTableEntry>& entries,
                       const Response& response) const = 0;

protected:
  virtual Status AllocateOutput(std::vector<TensorTableEntry>& entries,
                                const Response& response,
                                int64_t**& entry_component_sizes,
                                int*& recvcounts);

  virtual void SetDisplacements(const int* recvcounts, int*& displcmnts);

  virtual void
  SetEntryComponentOffsets(const std::vector<TensorTableEntry>& entries,
                           const int64_t* const* entry_component_sizes,
                           const int* recvcounts,
                           int64_t**& entry_component_offsets);

  virtual void
  MemcpyInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                       const int* displcmnts, int element_size,
                       void*& buffer_data);

  virtual void
  MemcpyOutFusionBuffer(const int64_t* const* entry_component_offsets,
                        const int64_t* const* entry_component_sizes,
                        const void* buffer_data, int element_size,
                        std::vector<TensorTableEntry>& entries);

  virtual void
  MemcpyEntryInFusionBuffer(const std::vector<TensorTableEntry>& entries,
                            const TensorTableEntry& e,
                            void* buffer_data_at_offset);

  virtual void
  MemcpyEntryOutFusionBuffer(const std::vector<TensorTableEntry>& entries,
                             const void* buffer_data_at_offset,
                             TensorTableEntry& e,
                             int64_t entry_offset,
                             size_t entry_size);
};

class BroadcastOp : public HorovodOp {
public:
  BroadcastOp(HorovodGlobalState* global_state);

  virtual ~BroadcastOp() = default;

  virtual Status Execute(std::vector<TensorTableEntry>& entries,
                         const Response& response) = 0;

  virtual bool Enabled(const ParameterManager& param_manager,
                       const std::vector<TensorTableEntry>& entries,
                       const Response& response) const = 0;
};

class JoinOp : public HorovodOp {
public:
  JoinOp(HorovodGlobalState* global_state);

  virtual ~JoinOp() = default;

  virtual Status Execute(std::vector<TensorTableEntry>& entries,
                         const Response& response);
};

class ErrorOp : public HorovodOp {
public:
  ErrorOp(HorovodGlobalState* global_state);

  virtual ~ErrorOp() = default;

  virtual Status Execute(std::vector<TensorTableEntry>& entries, const Response& response);
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_COLLECTIVE_OPERATIONS_H
