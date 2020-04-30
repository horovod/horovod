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

namespace horovod {
namespace common {

Status::Status() = default;

Status::Status(StatusType type, std::string reason) {
  type_ = type;
  reason_ = reason;
}

Status Status::OK() {
  return Status();
}

Status Status::UnknownError(std::string message) {
  return Status(StatusType::UNKNOWN_ERROR, message);
}

Status Status::PreconditionError(std::string message) {
  return Status(StatusType::PRECONDITION_ERROR, message);
}

Status Status::Aborted(std::string message) {
  return Status(StatusType::ABORTED, message);
}

Status Status::InvalidArgument(std::string message) {
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

const std::string TensorShape::DebugString() const {
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
  assert(idx < shape_.size());
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
void server_affinity_set(int affinity) {
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
void server_affinity_set(int affinity) {
  // TODO(travis): explore equivalent for macOS
  throw std::runtime_error("Environment variable HOROVOD_THREAD_AFFINITY is not supported on macOS.");
}
#endif

} // namespace common
} // namespace horovod
