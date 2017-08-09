// Copyright 2017 Uber Technologies, Inc. All Rights Reserved.
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

#ifndef HOROVOD_HASH_VECTOR_H
#define HOROVOD_HASH_VECTOR_H

#include <functional>

namespace std {

template <typename T> struct hash<std::vector<T>> {
  typedef std::vector<T> argument_type;
  typedef std::size_t result_type;

  result_type operator()(argument_type const& in) const {
    size_t size = in.size();
    size_t seed = 0;
    for (size_t i = 0; i < size; i++)
      seed ^= std::hash<T>()(in[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};

} // namespace std

#endif //HOROVOD_HASH_VECTOR_H
