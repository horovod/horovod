// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
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

#include "parameter_manager.h"

#include <cmath>
#include <limits>

namespace horovod {
namespace common {

#define MAX_DEPTH 4
#define MAX_SAMPLES 1.0

ParameterManager::ParameterManager() :
    tensor_fusion_threshold_(NumericParameter<int64_t>(
        64 * 1024 * 1024, 1024 * 1024, 256 * 1024 * 1024, 4 * 1024 * 1024, *this, nullptr)),
    cycle_time_ms_(NumericParameter<double>(
        5.0, 1.0, 25.0, 2.0, *this, &tensor_fusion_threshold_)),
    leaf_param_(&cycle_time_ms_),
    active_(false) {}

void ParameterManager::Update(int64_t bytes, double seconds) {
  if (!active_) {
    return;
  }

  leaf_param_->Step(bytes, seconds);
}

int64_t ParameterManager::TensorFusionThreshold() {
  return active_ ? tensor_fusion_threshold_.Value() : tensor_fusion_threshold_.BestValue();
};

void ParameterManager::SetTensorFusionThreshold(int64_t threshold) {
  tensor_fusion_threshold_.SetValue(threshold);
}

double ParameterManager::CycleTimeMs() {
  return active_ ? cycle_time_ms_.Value() : cycle_time_ms_.BestValue();
};

void ParameterManager::SetCycleTimeMs(double cycle_time_ms) {
  cycle_time_ms_.SetValue(cycle_time_ms);
}

template <class T>
ParameterManager::NumericParameter<T>::NumericParameter(
    T initial_value, T low, T high, T epsilon,
    ParameterManager& parent,
    ParameterManager::NumericParameter<T>::ITunableParameter* const next_param) :
    low_(low),
    high_(high),
    epsilon_(epsilon),
    current_({ initial_value, std::numeric_limits<double>::quiet_NaN() }),
    left_({ initial_value - epsilon, std::numeric_limits<double>::quiet_NaN() }),
    right_({ initial_value + epsilon, std::numeric_limits<double>::quiet_NaN() }),
    value_(initial_value),
    depth_(0),
    samples_(0),
    sum_score_(0),
    best_value_(initial_value),
    best_score_(0),
    parent_(parent),
    next_param_(next_param)  {}

template <class T>
void ParameterManager::NumericParameter<T>::SetValue(T value) {
  value_ = value;
  ResetState();

  best_value_ = value_;
  best_score_ = 0;
}

template <class T>
void ParameterManager::NumericParameter<T>::Step(double score, double samples) {
  sum_score_ += score;
  samples_ += samples;

  if (samples >= MAX_SAMPLES) {
    double score = sum_score_ / samples_;
    Tune(score);
  }
}

template <class T>
void ParameterManager::NumericParameter<T>::Tune(double score) {
  // Update the best score
  if (score > best_score_) {
    best_score_ = score;
    best_value_ = value_;
  }

  if (depth_ < MAX_DEPTH) {
    if (std::isnan(current_.score)) {
      current_.score = score;
      value_ = left_.value;
    } else if (std::isnan(left_.score)) {
      left_.score = score;
      value_ = right_.value;
    } else {
      right_.score = score;

      depth_++;
      CheckGradient();

      // Binary search
      value_ = (low_ + high_) / 2;
      if (value_ - epsilon_ <= low_ || value_ + epsilon_ >= high_) {
        // Out of bound
        DoneTune();
      }
    }
  } else {
    DoneTune();
  }

  ResetState();
}

template <class T>
void ParameterManager::NumericParameter<T>::CheckGradient() {
  if (current_.score >= left_.score && current_.score >= right_.score) {
    // Local maximum
    DoneTune();
    return;
  }

  if (left_.score >= current_.score) {
    // Explore left
    high_ = current_.value;
  } else {
    // Explore right
    low_ = current_.value;
  }
}

template <class T>
void ParameterManager::NumericParameter<T>::DoneTune() {
  if (next_param_ != nullptr) {
    next_param_->Tune(best_score_);
  } else {
    parent_.SetAutoTuning(false);
  }
  depth_ = 0;
}

template <class T>
void ParameterManager::NumericParameter<T>::ResetState() {
  current_.value = value_;
  current_.score = std::numeric_limits<double>::quiet_NaN();

  left_.value = value_ - epsilon_;
  left_.score = std::numeric_limits<double>::quiet_NaN();

  right_.value = value_ + epsilon_;
  right_.score = std::numeric_limits<double>::quiet_NaN();

  samples_ = 0;
  sum_score_ = 0;
}

} // namespace common
} // namespace horovod