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

#ifndef HOROVOD_DDL_OPERATIONS_H
#define HOROVOD_DDL_OPERATIONS_H

#include <ddl.hpp>

#include "gpu_operations.h"

namespace horovod {
namespace common {

struct DDLContext {
  int32_t ddl_local_device_id = 0;
};

class DDLAllreduce : public GPUAllreduce {
public:
  DDLAllreduce(DDLContext* ddl_context,
               GPUContext* gpu_context,
               HorovodGlobalState* global_state);

  Status Execute(std::vector<TensorTableEntry>& entries, const Response& response) override;

  static void DDLInit(DDLContext* ddl_context, GPUContext* gpu_context);

protected:
  DDLContext* ddl_context_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_DDL_OPERATIONS_H
