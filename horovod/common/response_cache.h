// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef HOROVOD_RESPONSE_CACHE_H
#define HOROVOD_RESPONSE_CACHE_H

#include <cassert>
#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "message.h"

namespace horovod {
namespace common {

// Structure to store relevant tensor parameters to deal with name collisions
struct TensorParams {
  DataType dtype;
  std::vector<int64_t> shape;
  int32_t device;
};

// LRU cache of Responses
class ResponseCache {
public:
  enum CacheState {
    MISS = 0, HIT = 1, INVALIDATE = 2
  };

  void set_capacity(uint32_t capacity);

  uint32_t capacity() const;

  size_t current_size() const;

  CacheState cached(const Request& message) const;

  CacheState cached(const Response& response, const TensorParams& params) const;

  void put(const Response& response, TensorTable& tensor_table);

  const Response& get_response(const Request& message);

  const Response& get_response(uint32_t cache_bit);

  const Response& peek_response(const Request& message) const;

  const Response& peek_response(uint32_t cache_bit) const;

  uint32_t peek_cache_bit(const Request& message) const;

  void erase_response(uint32_t cache_bit);

  void update_cache_bits();

private:
  void put_(const Response& response, TensorParams& params);

  uint32_t capacity_;
  std::list<std::pair<Response, TensorParams>> cache_;
  std::vector<std::list<std::pair<Response, TensorParams>>::iterator> iters_;
  std::unordered_map<std::string, uint32_t> table_;
  bool bits_outdated_ = false;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_RESPONSE_CACHE_H
