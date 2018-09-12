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

#define INVPHI 0.61803398875
#define INVPHI2 0.38196601125
#define TOL 0.00001

ParameterManager::ParameterManager() :
    tensor_fusion_threshold_(CategoricalParameter<int64_t>(
        std::vector<int64_t>{0, 1, 2, 4, 8, 16, 32, 64}, *this, nullptr)),
    cycle_time_ms_(CategoricalParameter<double>(
        std::vector<double>{1, 2.5, 5, 7.5, 10, 20, 30, 50}, *this, &tensor_fusion_threshold_)),
//    tensor_fusion_threshold_(NumericParameter<int64_t>(
//        1024 * 1024, 256 * 1024 * 1024, *this, nullptr)),
//    cycle_time_ms_(NumericParameter<double>(
//        1.0, 25.0, *this, &tensor_fusion_threshold_)),
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
ParameterManager::TunableParameter<T>::TunableParameter(
    T initial_value, ParameterManager &parent, ITunableParameter* const next_param) :
    initial_value_(initial_value),
    value_(initial_value),
    best_value_(initial_value),
    best_score_(0),
    parent_(parent),
    next_param_(next_param) {
  NextValue();
}

template <class T>
void ParameterManager::TunableParameter<T>::Step(double score, double samples) {
  sum_score_ += score;
  samples_ += samples;

  if (samples >= MAX_SAMPLES) {
    double score = sum_score_ / samples_;
    Tune(score);
  }
}

template <class T>
void ParameterManager::TunableParameter<T>::Tune(double score) {
  if (score > best_score_) {
    best_score_ = score;
    best_value_ = value_;
  }

  OnTune(score, value_);
  if (IsDoneTuning()) {
    CompleteTuning();
  }
  NextValue();
}

template <class T>
void ParameterManager::TunableParameter<T>::SetValue(T value) {
  value_ = value;
  ResetState();

  best_value_ = value_;
  best_score_ = 0;
}

template <class T>
void ParameterManager::TunableParameter<T>::CompleteTuning() {
  if (next_param_ != nullptr) {
    next_param_->Tune(best_score_);
  } else {
    parent_.SetAutoTuning(false);
  }

  value_ = initial_value_;
  ResetState();
}

template <class T>
void ParameterManager::TunableParameter<T>::NextValue() {
  samples_ = 0;
  sum_score_ = 0;
}

template <class T>
ParameterManager::NumericParameter<T>::NumericParameter(
    T low, T high,
    ParameterManager& parent,
    ParameterManager::ITunableParameter* const next_param) :
    TunableParameter<T>(low, parent, next_param),
    low_(low),
    high_(high) {
  ResetState();
}

template <class T>
void ParameterManager::NumericParameter<T>::OnTune(double score, T& value) {
  if (std::isnan(left_.score)) {
    left_.score = score;
    value = right_;
  } else if (std::isnan(right_.score)) {
    right_.score = score;
  }

  if (!std::isnan(left_.score) && !std::isnan(right_.score)) {
    if (left_.score < right_.score) {
      high_ = right_.value;
      right_.value = left_.value;
      right_.score = left_.score;
      h_ = INVPHI * h_;
      value = low_ + INVPHI2 * h_;
      left_.value = value;
      left_.score = std::numeric_limits<double>::quiet_NaN();
    } else {
      low_ = left_.value;
      left_.value = right_.value;
      left_.score = right_.score;
      h_ = INVPHI * h_;
      value = low_ + INVPHI * h_;
      right_.value = value;
      right_.score = std::numeric_limits<double>::quiet_NaN();
    }

    k_++;
  }
}

template <class T>
bool ParameterManager::NumericParameter<T>::IsDoneTuning() {
  return k_ >= n_ - 1;
}

template <class T>
void ParameterManager::NumericParameter<T>::ResetState() {
  h_ = high_ - low_;
  n_ = int32_t(ceil(log(TOL / h_) / log(INVPHI)));
  left_ = {low_ + INVPHI2 * h_, std::numeric_limits<double>::quiet_NaN()};
  right_ = {low_ + INVPHI * h_, std::numeric_limits<double>::quiet_NaN()};
  k_ = 0;
}

template <class T>
ParameterManager::CategoricalParameter<T>::CategoricalParameter(
    std::vector<T> values,
    ParameterManager& parent,
    ParameterManager::ITunableParameter* const next_param) :
    TunableParameter<T>(values[0], parent, next_param),
    values_(values) {
  ResetState();
}

template <class T>
void ParameterManager::CategoricalParameter<T>::OnTune(double score, T& value) {
  index_++;
  if (index_ < values_.size()) {
    value = values_[index_];
  }
}

template <class T>
bool ParameterManager::CategoricalParameter<T>::IsDoneTuning() {
  return index_ >= values_.size();
}

template <class T>
void ParameterManager::CategoricalParameter<T>::ResetState() {
  index_ = 0;
}

} // namespace common
} // namespace horovod