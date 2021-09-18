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

#ifndef HOROVOD_GLOO_CONTROLLER_H
#define HOROVOD_GLOO_CONTROLLER_H

#include "gloo_context.h"
#include "../controller.h"
#include "../logging.h"

namespace horovod {
namespace common {

class GlooController : public Controller {
public:
  GlooController(ResponseCache& response_cache, TensorQueue& tensor_queue,
                 Timeline& timeline, ParameterManager& parameter_manager,
                 GroupTable& group_table,
                 TimelineController& timeline_controller,
                 GlooContext& gloo_context)
      : Controller(response_cache, tensor_queue, timeline, parameter_manager,
                   group_table, timeline_controller),
        gloo_context_(gloo_context){};

  int GetTypeSize(DataType dtype) override;

  void CrossRankBitwiseAnd(std::vector<long long>& bitvector,
                           int count) override;

  void CrossRankBitwiseOr(std::vector<long long>& bitvector,
                          int count) override;

  void RecvReadyTensors(std::vector<std::string>& ready_to_reduce,
                        std::vector<RequestList>& ready_list) override;

  void SendFinalTensors(ResponseList& response_list) override;

  void SendReadyTensors(RequestList& message_list) override;

  void RecvFinalTensors(ResponseList& response_list) override;

  void Bcast(void* buffer, size_t size, int root_rank,
             Communicator communicator) override;

  void AlltoallGetRecvSplits(const std::vector<int32_t>& splits,
                             std::vector<int32_t>& recvsplits) override;

  void Barrier(Communicator communicator) override;

  void Allgather2Ints(std::array<int, 2> values,
                      std::vector<int>& recv_values) override;

protected:
  void DoInitialization() override;

  GlooContext& gloo_context_;
};

template <typename T>
void BitOr(void* c_, const void* a_, const void* b_, size_t n) {
  T* c = static_cast<T*>(c_);
  const T* a = static_cast<const T*>(a_);
  const T* b = static_cast<const T*>(b_);
  for (size_t i = 0; i < n; i++) {
    c[i] = a[i] | b[i];
  }
}

template <typename T>
void BitAnd(void* c_, const void* a_, const void* b_, size_t n) {
  T* c = static_cast<T*>(c_);
  const T* a = static_cast<const T*>(a_);
  const T* b = static_cast<const T*>(b_);
  for (size_t i = 0; i < n; i++) {
    c[i] = a[i] & b[i];
  }
}

} // namespace common
} // namespace horovod
#endif // HOROVOD_GLOO_CONTROLLER_H
