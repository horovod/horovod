// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
// Modifications copyright (C) 2019 Intel Corporation
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

#include "common.h"
#include "logging.h"

#include <sstream>
#include <cassert>
#include <cstring>
#include <utility>
#include <limits.h>

namespace horovod {
namespace common {

Status::Status() = default;

Status::Status(StatusType type, std::string reason)
    : type_(type), reason_(std::move(reason)) {
}

Status Status::OK() {
  return Status();
}

Status Status::UnknownError(const std::string& message) {
  return Status(StatusType::UNKNOWN_ERROR, message);
}

Status Status::PreconditionError(const std::string& message) {
  return Status(StatusType::PRECONDITION_ERROR, message);
}

Status Status::Aborted(const std::string& message) {
  return Status(StatusType::ABORTED, message);
}

Status Status::InvalidArgument(const std::string& message) {
  return Status(StatusType::INVALID_ARGUMENT, message);
}

Status Status::InProgress() {
  return Status(StatusType::IN_PROGRESS, "");
}

bool Status::ok() const {
  return type_ == StatusType::OK;
}

bool Status::in_progress() const {
  return type_ == StatusType::IN_PROGRESS;
}

StatusType Status::type() const {
  return type_;
}

const std::string& Status::reason() const {
  return reason_;
}

void TensorShape::AddDim(int64_t dim) {
  shape_.push_back(dim);
}

void TensorShape::AppendShape(TensorShape& other) {
  for (auto dim : other.shape_) {
    shape_.push_back(dim);
  }
}

std::string TensorShape::DebugString() const {
  std::stringstream args;
  args << "[";
  for (auto it = shape_.begin(); it != shape_.end(); ++it) {
    if (it != shape_.begin()) {
      args << ", ";
    }
    args << *it;
  }
  args << "]";
  return args.str();
}

int TensorShape::dims() const {
  return (int)shape_.size();
}

int64_t TensorShape::dim_size(int idx) const {
  assert(idx >= 0);
  assert(idx < (int)shape_.size());
  return shape_[idx];
}

int64_t TensorShape::num_elements() const {
  int64_t result = 1;
  for (auto dim : shape_) {
    result *= dim;
  }
  return result;
}

const std::vector<int64_t>& TensorShape::to_vector() const { return shape_; }

#ifdef __linux__
void set_affinity(int affinity) {
  cpu_set_t cpuset;
  pthread_t current_thread = pthread_self();

  __CPU_ZERO_S(sizeof(cpu_set_t), &cpuset);
  __CPU_SET_S(affinity, sizeof(cpu_set_t), &cpuset);

  if (pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
    LOG(ERROR) << "setaffinity failed";
  }

  // Check if we set the affinity correctly
  if (pthread_getaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset) != 0) {
    LOG(ERROR) << "sched_getaffinity failed";
  }

  for (int core_idx = 0; core_idx < __CPU_SETSIZE; core_idx++) {
    if (__CPU_ISSET_S(core_idx, sizeof(cpu_set_t), &cpuset)) {
      LOG(INFO) << "Background thread affinity " << core_idx;
    }
  }
}
#else
void set_affinity(int affinity) {
  // TODO(travis): explore equivalent for macOS
  throw std::runtime_error("Environment variable HOROVOD_THREAD_AFFINITY is not supported on macOS.");
}
#endif

void parse_and_set_affinity(const char* affinity, int local_size, int local_rank) {
  if (affinity == nullptr) {
    return;
  }

  size_t affinity_len = strlen(affinity);

  // copy is needed because strsep is going to modify the buffer
  char* affinity_copy = (char*)calloc(affinity_len + 1, sizeof(char));
  memcpy(affinity_copy, affinity, affinity_len);
  char* tmp = affinity_copy;
  char* endptr;

  std::vector<int> core_ids(local_size);
  int count = 0;

  while (tmp && count < local_size) {
    auto core_id_str = strsep(&tmp, ",");
    errno = 0;
    auto core_id = std::strtol(core_id_str, &endptr, 10);
    if ((errno == ERANGE && (core_id == LONG_MAX || core_id == LONG_MIN))
        || (errno != 0 && core_id == 0)){
        LOG(ERROR) << "Core ID value is invalid in " << HOROVOD_THREAD_AFFINITY
                   << "=" << affinity;
        break;
    }

    if (endptr == core_id_str) {
        LOG(ERROR) << "No digits were found in " << HOROVOD_THREAD_AFFINITY
                   << "=" << affinity;
        break;
    }
    
    if (core_id < 0) {
      LOG(ERROR) << "Core ID cannot be less than zero but got "
                 << core_id << " in "
                 << HOROVOD_THREAD_AFFINITY << "=" << affinity;
      break;
    } else {
      core_ids[count] = core_id;
      count++;
    }
  }
    
  if (count < local_size) {
    LOG(ERROR) << "Expected " << local_size << " core ids but got " << count << ". "
               << HOROVOD_THREAD_AFFINITY << "=" << affinity;
  } else {
    set_affinity(core_ids[local_rank]);
  }

  free(affinity_copy);
}

void TensorTableEntry::FinishWithCallback(const Status& status) {
  // Callback can be null if the rank sent Join request.
  if (callback != nullptr) {
    callback(status);
  }
  nvtx_op_range.End();
}

} // namespace common
} // namespace horovod
