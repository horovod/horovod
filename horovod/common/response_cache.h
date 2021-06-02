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
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common.h"
#include "message.h"

#define NUM_STATUS_BITS 3

namespace horovod {
namespace common {

class Controller;
class TensorQueue;

// Structure to store relevant tensor parameters to deal with name collisions
struct TensorParams {
  DataType dtype;
  std::vector<int64_t> shape;
  int32_t device;
};

// LRU cache of Responses
class ResponseCache {
public:
  ResponseCache() = default;
  ResponseCache(const ResponseCache&) = delete;

  enum CacheState { MISS = 0, HIT = 1, INVALID = 2 };

  void clear();

  void set_capacity(uint32_t capacity);

  uint32_t capacity() const;

  size_t num_active_bits() const;

  CacheState cached(const Request& message) const;

  CacheState cached(const Response& response, const TensorParams& params,
                    bool joined = false) const;

  void put(const Response& response, TensorQueue& tensor_queue,
           bool joined = false);

  const Response& get_response(uint32_t cache_bit);

  const Response& peek_response(uint32_t cache_bit) const;

  uint32_t peek_cache_bit(const Request& message) const;

  uint32_t peek_cache_bit(const std::string& tensor_name) const;

  std::vector<uint32_t> list_all_bits() const;

  void erase_response(uint32_t cache_bit);

  void update_cache_bits();

private:
  void put_(const Response& response, TensorParams& params,
            bool joined = false);

  uint32_t capacity_ = 0;

  // List containing cached entries. Each entry in the cache is a pair
  // of a Response and a TensorParams struct.
  std::list<std::pair<Response, TensorParams>> cache_;

  // Vector of iterators to cache entries. Indexed by cache bit.
  std::vector<std::list<std::pair<Response, TensorParams>>::iterator>
      cache_iters_;

  // Lookup table mapping tensor names to assigned cache bits.
  std::unordered_map<std::string, uint32_t> tensor_name_to_bit_;

  bool bits_outdated_ = false;

  bool print_warning_ = true;
};

// Helper class to coordinate cache and state information
// across workers. Uses global controller operations on a bit vector
// for cheaper coordination.
class CacheCoordinator {
public:
  explicit CacheCoordinator(size_t num_active_bits_);

  void record_hit(uint32_t bit);

  void record_invalid_bit(uint32_t bit);

  void erase_hit(uint32_t bit);

  void set_should_shut_down(bool should_shut_down);

  void set_uncached_in_queue(bool uncached_in_queue);

  const std::set<uint32_t>& cache_hits() const;

  const std::set<uint32_t>& invalid_bits() const;

  const std::set<uint32_t>& timeline_bits() const;

  bool should_shut_down() const;

  bool uncached_in_queue() const;

  // Method to sync state and bit sets across workers.
  void sync(std::shared_ptr<Controller> controller, bool timeline_enabled);

private:
  enum StatusBit {
    SHOULD_SHUT_DOWN = 0,
    UNCACHED_IN_QUEUE = 1,
    INVALID_IN_QUEUE = 2
  };

  // Number of active bits in the cache. Required to size the
  // bitvector identically across workers.
  size_t num_active_bits_;

  // Set of cache hit bits. After sync(), contains only common
  // cache hit bits across workers.
  std::set<uint32_t> cache_hits_;

  // Set of invalid bits. After sync(), contains only common
  // invalid bits across workers.
  std::set<uint32_t> invalid_bits_;

  // Set of bits for timeline handling. After sync(), contains bits
  // where at least one worker recorded a cache hit. This indicates
  // that the timeline negotion phase should be started/continued.
  std::set<uint32_t> timeline_bits_;

  // States used externally in cycle loop.
  bool should_shut_down_ = false;
  bool uncached_in_queue_ = false;

  // State used internally to trigger second bit vector communication
  // to sync invalid bits.
  bool invalid_in_queue_ = false;

  std::vector<long long> bitvector_;

  bool synced_ = false;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_RESPONSE_CACHE_H
