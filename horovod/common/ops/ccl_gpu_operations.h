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

#ifndef HOROVOD_CCL_GPU_OPERATIONS_H_
#define HOROVOD_CCL_GPU_OPERATIONS_H_

#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "oneapi/ccl.hpp"

#include "../logging.h"
#include "gpu_operations.h"

namespace horovod {
namespace common {
ccl::datatype GetCCLDataType(const std::shared_ptr<Tensor>& tensor);

struct ccl4hvd {
  ccl::stream ccl_stream_;
  ccl::communicator ccl_comm_;
};

class CCLGPUContext {
public:
  CCLGPUContext() = default;
  ~CCLGPUContext() = default;

  void Initialize(HorovodGlobalState& state);
  void Finalize();

  template <typename ccl_fn_type>
  static decltype(auto) CallWithLock(std::mutex& lock, ccl_fn_type fn) {
    std::unique_lock<std::mutex> GlobalMutex(lock);
    return fn();
  }

  // base primitives
  std::vector<
      std::unordered_map<std::tuple<int32_t, std::vector<int32_t>>, ccl4hvd>>
      ccl_comms;

  // ccl helpers knobs
  bool enable_cache;

  static std::mutex GlobalMutex;
};

class CCLGPUOpContext {
public:
  CCLGPUOpContext(CCLGPUContext* context, HorovodGlobalState* global_state,
                  Communicator communicator_type)
      : kvs_(nullptr), ccl_context_(context), global_state_(global_state),
        communicator_type_(communicator_type){};
  ~CCLGPUOpContext();

  void InitCCLComm(const gpuStream_t& stream,
                   const std::vector<TensorTableEntry>& entries,
                   const std::vector<int32_t>& ccl_device_map);

  // helpers
  bool IsEnabled(const std::vector<TensorTableEntry>& entries) const;
  ccl::communicator& GetCCLComm(const TensorTableEntry& entry,
                                const std::vector<int32_t>& devices);
  ccl::stream& GetCCLStream(const TensorTableEntry& entry,
                            const std::vector<int32_t>& devices);

  std::shared_ptr<ccl::kvs> kvs_;

  // oneCCL does not support async error check
  std::function<void()> error_check_callback_;

private:
  void PopulateCCLCommStrategy(int& ccl_rank, int& ccl_size,
                               Communicator& ccl_id_bcast_comm,
                               const ProcessSet& process_set);

  CCLGPUContext* ccl_context_;
  HorovodGlobalState* global_state_;
  Communicator communicator_type_;

  std::vector<ccl::stream> ccl_streams;
  std::vector<ccl::device> ccl_devices;
  std::vector<ccl::context> ccl_contexts;
};

class CCLGPUAllreduce : public GPUAllreduce {
public:
  CCLGPUAllreduce(CCLGPUContext* ccl_context, GPUContext* gpu_context,
                  HorovodGlobalState* global_state,
                  Communicator communicator_type = Communicator::GLOBAL)
      : GPUAllreduce(gpu_context, global_state), ccl_context_(ccl_context),
        ccl_op_context_(ccl_context, global_state, communicator_type),
        global_state_(global_state) {}

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;
  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

protected:
  void WaitForData(std::vector<TensorTableEntry>& entries) override;

  CCLGPUContext* ccl_context_;
  CCLGPUOpContext ccl_op_context_;
  HorovodGlobalState* global_state_;
};

class CCLGPUBroadcast : public GPUBroadcast {
public:
  CCLGPUBroadcast(CCLGPUContext* ccl_context, GPUContext* gpu_context,
                  HorovodGlobalState* global_state,
                  Communicator communicator_type = Communicator::GLOBAL)
      : GPUBroadcast(gpu_context, global_state), ccl_context_(ccl_context),
        ccl_op_context_(ccl_context, global_state, communicator_type),
        global_state_(global_state) {}

  bool Enabled(const ParameterManager& param_manager,
               const std::vector<TensorTableEntry>& entries,
               const Response& response) const override;
  Status Execute(std::vector<TensorTableEntry>& entries,
                 const Response& response) override;

protected:
  void WaitForData(std::vector<TensorTableEntry>& entries) override;

  CCLGPUContext* ccl_context_;
  CCLGPUOpContext ccl_op_context_;
  HorovodGlobalState* global_state_;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_CCL_GPU_OPERATIONS_H_
