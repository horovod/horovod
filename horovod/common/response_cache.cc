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

#include "response_cache.h"
#include <iostream>

namespace horovod {
namespace common {

void ResponseCache::set_capacity(uint32_t capacity) {
  capacity_ = capacity;
  iters_.reserve(capacity);
}

uint32_t ResponseCache::capacity() const {return capacity_;}

size_t ResponseCache::current_size() const {return cache_.size();}

ResponseCache::CacheState ResponseCache::cached(const Request& message) const {
  auto it = table_.find(message.tensor_name());
  if (it != table_.end()) {
    // If entry associated with this request already exists in cache, check
    // if tensor parameters match. If not, mark entry to be invalidated.
    uint32_t cache_bit = it->second;
    auto& cache_params = std::get<1>(*iters_[cache_bit]);
    return (cache_params.device == message.device() &&
            cache_params.dtype == message.tensor_type() &&
            cache_params.shape == message.tensor_shape()) ?
            CacheState::HIT : CacheState::INVALIDATE;
  } else {
    return CacheState::MISS;
  }
}

ResponseCache::CacheState ResponseCache::cached(const Response& response,
                                                const TensorParams& params) const {
  assert(response.tensor_names().size() == 1);
  auto it = table_.find(response.tensor_names()[0]);
  if (it != table_.end()) {
    uint32_t cache_bit = it->second;
    auto& cache_params = std::get<1>(*iters_[cache_bit]);
    return (cache_params.device == params.device &&
            cache_params.dtype == params.dtype &&
            cache_params.shape == params.shape) ?
            CacheState::HIT : CacheState::INVALIDATE;
  } else {
    return CacheState::MISS;
  }
}

void ResponseCache::put_(const Response& response, TensorParams& params) {
  uint32_t cache_bit;
  auto cache_state = this->cached(response, params);

  // Disallow caching name-conflicted responses here. Invalid cache entries
  // must be removed prior to caching new entries.
  if (cache_state == CacheState::INVALIDATE) {
    throw std::logic_error("Trying to overwrite cached response with existing name. "
                           "This is not allowed.");
  }

  if (this->cached(response, params) == CacheState::HIT) {
    cache_bit = table_[response.tensor_names()[0]];
    auto it = iters_[cache_bit];
    cache_.push_front(std::move(*it));
    cache_.erase(it);
  } else if (cache_.size() == capacity_) {
    auto& entry = cache_.back().first;
    cache_bit = table_[entry.tensor_names()[0]];
    table_.erase(entry.tensor_names()[0]);
    cache_.pop_back();
    cache_.push_front(std::make_pair(response, std::move(params)));
  } else {
    cache_bit = iters_.size();
    iters_.resize(cache_bit + 1);
    cache_.push_front(std::make_pair(response, std::move(params)));
  }

  iters_[cache_bit] = cache_.begin();
  table_[response.tensor_names()[0]] = cache_bit;

  bits_outdated_ = true;
}

void ResponseCache::put(const Response& response, TensorTable& tensor_table) {
  if (capacity_ == 0) return;

  // If response is fused, split back into individual responses
  if (response.tensor_names().size() > 0) {
    for (auto& name : response.tensor_names()) {
      Response new_response;
      new_response.add_tensor_name(name);
      new_response.set_response_type(response.response_type());
      new_response.set_devices(response.devices());
      new_response.set_tensor_sizes(response.tensor_sizes());

      // Populate tensor parameters from tensor_table entry
      auto& tensor_entry = tensor_table[name];
      TensorParams params;
      params.device = tensor_entry.device;
      params.dtype = tensor_entry.tensor->dtype();
      params.shape = tensor_entry.tensor->shape().shape();

      this->put_(new_response, params);
    }
  } else {
    auto& tensor_entry = tensor_table[response.tensor_names()[0]];
    TensorParams params;
    params.device = tensor_entry.device;
    params.dtype = tensor_entry.tensor->dtype();
    params.shape = tensor_entry.tensor->shape().shape();

    this->put_(response, params);
  }
}

const Response& ResponseCache::get_response(const Request& message) {
  assert(this->cached(message));
  uint32_t cache_bit = table_[message.tensor_name()];
  auto it = iters_[cache_bit];
  cache_.push_front(std::move(*it));
  cache_.erase(it);
  iters_[cache_bit] = cache_.begin();
  bits_outdated_ = true;
  return cache_.front().first;
}

const Response& ResponseCache::get_response(uint32_t cache_bit) {
  assert(cache_bit < iters_.size());
  auto it = iters_[cache_bit];
  cache_.push_front(std::move(*it));
  cache_.erase(it);
  iters_[cache_bit] = cache_.begin();
  bits_outdated_ = true;
  return cache_.front().first;
}

const Response& ResponseCache::peek_response(const Request& message) const {
  assert(this->cached(message));
  uint32_t cache_bit = table_.at(message.tensor_name());
  return std::get<0>(*iters_[cache_bit]);
}

const Response& ResponseCache::peek_response(uint32_t cache_bit) const {
  assert(cache_bit < iters_.size());
  return std::get<0>(*iters_[cache_bit]);
}

uint32_t ResponseCache::peek_cache_bit(const Request& message) const {
  assert(this->cached(message));
  return table_.at(message.tensor_name());
}

void ResponseCache::erase_response(uint32_t cache_bit) {
  assert(cache_bit < iters_.size());
  auto it = iters_[cache_bit];
  table_.erase(it->first.tensor_names()[0]);
  cache_.erase(it);
  // When erasing entry, do not resize iters_ vector. Positions
  // are reset when update_cache_bits function is called.
  iters_[cache_bit] = cache_.end();
  bits_outdated_ = true;
}

void ResponseCache::update_cache_bits() {
  if (!bits_outdated_) return;

  // Iterate over current cache state and reassign cache bits. Least recently
  // used get lower cache positions.
  auto it = --cache_.end();
  for (int i = 0; i < (int)cache_.size(); ++i) {
    iters_[i] = it;
    table_[it->first.tensor_names()[0]] =  i;
    --it;
  }

  // Resize iters_ vector to contain only valid bit positions.
  iters_.resize(cache_.size());

  bits_outdated_ = false;
}

} // namespace common
} // namespace horovod
