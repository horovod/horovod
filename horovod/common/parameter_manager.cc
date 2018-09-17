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

#include <algorithm>
#include <cmath>
#include <limits>

namespace horovod {
namespace common {

#define WARMUPS 3

#define INVPHI 0.61803398875
#define INVPHI2 0.38196601125
#define TOL 0.00001

// ParameterManager
ParameterManager::ParameterManager() :
    tensor_fusion_threshold_mb_(CategoricalParameter<int64_t>(
        std::vector<int64_t>{0, 1, 2, 4, 8, 16, 32, 64}, *this, nullptr)),
    cycle_time_ms_(CategoricalParameter<double>(
        std::vector<double>{1, 2.5, 5, 7.5, 10, 20, 30, 50}, *this, &tensor_fusion_threshold_mb_)),
//    tensor_fusion_threshold_mb_(NumericParameter<int64_t>(
//        1024 * 1024, 256 * 1024 * 1024, *this, nullptr)),
//    cycle_time_ms_(NumericParameter<double>(
//        1.0, 25.0, *this, &tensor_fusion_threshold_mb_)),
    leaf_param_(&cycle_time_ms_),
    active_(false),
    warmup_remaining_(WARMUPS),
    cycle_(0),
    rank_(-1),
    root_rank_(0),
    writing_(false) {
  ReadyTune();
}

void ParameterManager::Initialize(int32_t rank, int32_t root_rank, std::string file_name) {
  rank_ = rank;
  root_rank_ = root_rank;
  if (rank_ == root_rank && !file_name.empty()) {
    file_.open(file_name, std::ios::out | std::ios::trunc);
    if (file_.good()) {
      file_ << "cycle_time_ms,tensor_fusion_threshold,score" << std::endl;
      writing_ = true;
    }
  }
}

void ParameterManager::SetAutoTuning(bool active) {
  if (active != active_) {
    warmup_remaining_ = WARMUPS;
  }
  active_ = active;
};

int64_t ParameterManager::TensorFusionThresholdBytes() const {
  int64_t b = active_ ? tensor_fusion_threshold_mb_.Value() : tensor_fusion_threshold_mb_.BestValue();
  return b * 1024 * 1024;
};

void ParameterManager::SetTensorFusionThresholdBytes(int64_t threshold) {
  tensor_fusion_threshold_mb_.SetValue(threshold / (1024 * 1024));
}

double ParameterManager::CycleTimeMs() const {
  return active_ ? cycle_time_ms_.Value() : cycle_time_ms_.BestValue();
};

void ParameterManager::SetCycleTimeMs(double cycle_time_ms) {
  cycle_time_ms_.SetValue(cycle_time_ms);
}

void ParameterManager::Update(const std::vector<std::string>& tensor_names, int64_t bytes, double seconds) {
  if (!active_) {
    return;
  }

  for (const std::string& tensor_name : tensor_names) {
    int32_t cycle = tensor_counts_[tensor_name]++;
    if (cycle > cycle_) {
      scores_[cycle_] = total_bytes_ / total_seconds_;
      total_bytes_ = 0;
      total_seconds_ = 0;
      cycle_ = cycle;
      break;
    }
  }

  total_bytes_ += bytes;
  total_seconds_ += seconds;

  if (cycle_ >= CYCLES) {
    std::sort(scores_, scores_ + CYCLES);
    double med_score = scores_[CYCLES / 2];
    Tune(med_score);
  }
}

void ParameterManager::Tune(double score) {
  if (warmup_remaining_ > 0) {
    warmup_remaining_--;
    std::cerr << "WARMUP DONE" << std::endl;
  } else {
    if (rank_ == root_rank_) {
      std::cerr << total_bytes_ << ", " << total_seconds_ << " "
                << "[" << cycle_time_ms_.Value() << ", " << tensor_fusion_threshold_mb_.Value() << "] " << score << "  "
                << "[" << cycle_time_ms_.BestValue() << ", " << tensor_fusion_threshold_mb_.BestValue() << "] "
                << leaf_param_->BestScore()
                << std::endl;
      if (writing_ && file_.good()) {
        file_ << cycle_time_ms_.Value() << "," << tensor_fusion_threshold_mb_.Value() << "," << score << std::endl;
      }
    }

    leaf_param_->Tune(score);
  }
  ReadyTune();
}

void ParameterManager::ReadyTune() {
  total_bytes_ = 0;
  total_seconds_ = 0;
  tensor_counts_.clear();
  cycle_ = 0;
}

// TunableParameter
template <class T>
ParameterManager::TunableParameter<T>::TunableParameter(
    T initial_value, ParameterManager &parent, ITunableParameter* const next_param) :
    initial_value_(initial_value),
    value_(initial_value),
    best_value_(initial_value),
    best_score_(0),
    parent_(parent),
    next_param_(next_param) {}

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
}

template <class T>
void ParameterManager::TunableParameter<T>::SetValue(T value) {
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

// NumericParameter
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
bool ParameterManager::NumericParameter<T>::IsDoneTuning() const {
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

// CategoricalParameter
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
bool ParameterManager::CategoricalParameter<T>::IsDoneTuning() const {
  return index_ >= values_.size();
}

template <class T>
void ParameterManager::CategoricalParameter<T>::ResetState() {
  index_ = 0;
}

} // namespace common
} // namespace horovod