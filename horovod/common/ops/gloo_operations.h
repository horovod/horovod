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

#ifndef HOROVOD_GLOO_OPERATIONS_H
#define HOROVOD_GLOO_OPERATIONS_H

#include <stdint.h>
#include <vector>

#include "collective_operations.h"
#include "../gloo/gloo_context.h"

namespace horovod {
namespace common {

class IGlooAlgorithms {
public:
  virtual void Allreduce(void* buffer_data, int num_elements) = 0;

  virtual void Allgather(void* buffer_data, void* buffer_out, int* recvcounts,
                         int* displcmnts) = 0;

  virtual void Broadcast(void* buffer_data, int num_elements,
                         int root_rank) = 0;

  virtual void Alltoall(void* buffer_data, void* buffer_out,
                        std::vector<int64_t>& sendcounts,
                        std::vector<int64_t>& recvcounts) = 0;

  virtual int ElementSize() const = 0;
};

template <typename T> class GlooAlgorithms : public IGlooAlgorithms {
public:
  explicit GlooAlgorithms(GlooContext* gloo_context);

  ~GlooAlgorithms() = default;

  void Allreduce(void* buffer_data, int num_elements) override;

  void Allgather(void* buffer_data, void* buffer_out, int* recvcounts,
                 int* displcmnts) override;

  void Broadcast(void* buffer_data, int num_elements, int root_rank) override;

  void Alltoall(void* buffer_data, void* buffer_out,
                std::vector<int64_t>& sendcounts,
                std::vector<int64_t>& recvcounts) override;

  int ElementSize() const override;

private:
  GlooContext* gloo_context_;
};

class GlooAllreduce : public AllreduceOp {
public:
  explicit GlooAllreduce(HorovodGlobalState* global_state);

  virtual ~GlooAllreduce() = default;

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;
};

class GlooAllgather : public AllgatherOp {
public:
  explicit GlooAllgather(HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;
};

class GlooBroadcast : public BroadcastOp {
public:
  explicit GlooBroadcast(HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;
};

class GlooAlltoall : public AlltoallOp {
public:
  explicit GlooAlltoall(HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_GLOO_OPERATIONS_H
