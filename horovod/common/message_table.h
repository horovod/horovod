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

#ifndef HOROVOD_MESSAGE_TABLE_H
#define HOROVOD_MESSAGE_TABLE_H

#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "message.h"
#include "timeline.h"

namespace horovod {
namespace common {

// Table for storing Tensor metadata on rank zero. This is used for error
// checking, stall checking and size calculations, as well as determining
// when a reduction is ready to be done (when all nodes are ready to do it).
using MessageTable = std::unordered_map<
    std::string,
    std::tuple<std::vector<Request>, std::chrono::steady_clock::time_point>>;

// Store the Request for a name, and return whether the total count of
// Requests for that tensor is now equal to the MPI size (and thus we are
// ready to reduce the tensor).
bool IncrementTensorCount(std::shared_ptr<MessageTable> message_table,
                          const Request& msg, int global_size,
                          Timeline& timeline);

} // namespace common

} // namespace horovod
#endif // HOROVOD_MESSAGE_TABLE_H
