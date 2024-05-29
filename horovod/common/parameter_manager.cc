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

#include "logging.h"
#include "utils/env_parser.h"

namespace horovod {
namespace common {

#define DEFAULT_WARMUPS 3
#define DEFAULT_STEPS_PER_SAMPLE 10
#define DEFAULT_BAYES_OPT_MAX_SAMPLES 20
#define DEFAULT_GAUSSIAN_PROCESS_NOISE 0.8

Eigen::VectorXd CreateVector(double x1, double x2) {
  Eigen::VectorXd v(2);
  v(0) = x1;
  v(1) = x2;
  return v;
}

// ParameterManager
ParameterManager::ParameterManager() :
    warmups_(GetIntEnvOrDefault(HOROVOD_AUTOTUNE_WARMUP_SAMPLES, DEFAULT_WARMUPS)),
    steps_per_sample_(GetIntEnvOrDefault(HOROVOD_AUTOTUNE_STEPS_PER_SAMPLE, DEFAULT_STEPS_PER_SAMPLE)),
    hierarchical_allreduce_(CategoricalParameter<bool>(std::vector<bool>{false, true})),
    hierarchical_allgather_(CategoricalParameter<bool>(std::vector<bool>{false, true})),
    torus_allreduce_(CategoricalParameter<bool>(std::vector<bool>{false, true})),
    cache_enabled_(CategoricalParameter<bool>(std::vector<bool>{false, true})),
    joint_params_(BayesianParameter(
      std::vector<BayesianVariableConfig>{
        { BayesianVariable::fusion_buffer_threshold_mb, std::pair<double, double>(0, 64) },
        { BayesianVariable::cycle_time_ms, std::pair<double, double>(1, 100) }
      },
      std::vector<Eigen::VectorXd>{
        CreateVector(4, 5),
        CreateVector(32, 50),
        CreateVector(16, 25),
        CreateVector(8, 10)
      },
      GetIntEnvOrDefault(HOROVOD_AUTOTUNE_BAYES_OPT_MAX_SAMPLES, DEFAULT_BAYES_OPT_MAX_SAMPLES),
      GetDoubleEnvOrDefault(HOROVOD_AUTOTUNE_GAUSSIAN_PROCESS_NOISE, DEFAULT_GAUSSIAN_PROCESS_NOISE))),
    parameter_chain_(std::vector<ITunableParameter*>{&joint_params_, &hierarchical_allreduce_, &hierarchical_allgather_,
                                                     &torus_allreduce_, &cache_enabled_}),
    active_(false),
    warmup_remaining_(warmups_),
    sample_(0),
    rank_(-1),
    root_rank_(0),
    writing_(false) {
  Reset();
}

void ParameterManager::Initialize(int32_t rank, int32_t root_rank,
                                  const std::string& file_name) {
  rank_ = rank;
  root_rank_ = root_rank;
  if (rank_ == root_rank) {
    LOG(INFO) << "Autotuner: Tunable params [hierarchical_allreduce,hierarchical_allgather,torus_allreduce,cache_enabled,cycle_time_ms,tensor_fusion_threshold] score";
  }
  if (rank_ == root_rank && !file_name.empty()) {
    file_.open(file_name, std::ios::out | std::ios::trunc);
    if (file_.good()) {
      file_ << "hierarchical_allreduce,hierarchical_allgather,torus_allreduce,cache_enabled,cycle_time_ms,tensor_fusion_threshold,score" << std::endl;
      writing_ = true;
    }
  }
}

void ParameterManager::SetAutoTuning(bool active) {
  if (active != active_) {
    warmup_remaining_ = warmups_;
  }
  active_ = active;
};

bool ParameterManager::HierarchicalAllreduce() const {
  return active_ ? hierarchical_allreduce_.Value() : hierarchical_allreduce_.BestValue();
}

void ParameterManager::SetHierarchicalAllreduce(bool value, bool fixed) {
  hierarchical_allreduce_.SetValue(value, fixed);
}

bool ParameterManager::HierarchicalAllgather() const {
  return active_ ? hierarchical_allgather_.Value() : hierarchical_allgather_.BestValue();
}

void ParameterManager::SetHierarchicalAllgather(bool value, bool fixed) {
  hierarchical_allgather_.SetValue(value, fixed);
}

bool ParameterManager::TorusAllreduce() const {
  return active_ ? torus_allreduce_.Value() : torus_allreduce_.BestValue();
}

void ParameterManager::SetTorusAllreduce(bool value, bool fixed) {
  torus_allreduce_.SetValue(value, fixed);
}

bool ParameterManager::CacheEnabled() const {
  return active_ ? cache_enabled_.Value() : cache_enabled_.BestValue();
};

void ParameterManager::SetCacheEnabled(bool enabled, bool fixed) {
  cache_enabled_.SetValue(enabled, fixed);
}

int64_t ParameterManager::TensorFusionThresholdBytes() const {
  double b = active_ ?
      joint_params_.Value(fusion_buffer_threshold_mb) :
      joint_params_.BestValue(fusion_buffer_threshold_mb);
  return int64_t(b * 1024 * 1024);
};

void ParameterManager::SetTensorFusionThresholdBytes(int64_t threshold, bool fixed) {
  joint_params_.SetValue(fusion_buffer_threshold_mb, double(threshold) / (1024 * 1024), fixed);
}

double ParameterManager::CycleTimeMs() const {
  return active_ ? joint_params_.Value(cycle_time_ms) : joint_params_.BestValue(cycle_time_ms);
};

void ParameterManager::SetCycleTimeMs(double value, bool fixed) {
  joint_params_.SetValue(cycle_time_ms, value, fixed);
}

/// Using the tensors information to update score and tune parameters.
/// \param tensor_names A vector of names of the tensors being processed in the
/// current timestamp.
/// \param bytes Total size of the tensors.
/// \return Whether the new parameters need to be broadcasted.
bool ParameterManager::Update(const std::vector<std::string>& tensor_names,
                              int64_t bytes) {
  if (!active_) {
    return false;
  }

  for (const std::string& tensor_name : tensor_names) {
    int32_t step = tensor_counts_[tensor_name]++;
    if (step >= (sample_ + 1) * steps_per_sample_) {
      auto now = std::chrono::steady_clock::now();
      double duration = std::chrono::duration_cast<std::chrono::microseconds>(now - last_sample_start_).count();
      scores_[sample_] = total_bytes_ / duration;

      total_bytes_ = 0;
      last_sample_start_ = now;
      ++sample_;
      break;
    }
  }

  total_bytes_ += bytes;

  if (sample_ >= SAMPLES) {
    std::sort(scores_, scores_ + SAMPLES);
    double med_score = scores_[SAMPLES / 2];
    return Tune(med_score);
  }

  return false;
}

/// Tune the parameters based on the score
/// \param score The score for current timestamp
/// \return Whether the parameter should be broadcast to other ranks.
bool ParameterManager::Tune(double score) {
  if (warmup_remaining_ > 0) {
    // Ignore this score as we're still warming up.
    warmup_remaining_--;
    if (rank_ == root_rank_) {
      LOG(INFO) << "Autotuner: Warming up (" << warmup_remaining_ << " remaining)";
    }
  } else {
    // Log the last parameter values before updating.
    LogParameters(score);

    // Only do the tuning on the coordinator to ensure consistency.
    if (rank_ == root_rank_) {
      bool finished_tuning = true;
      double best_score = score;
      for (auto* param : parameter_chain_) {
        double new_best_score;
        bool finished = param->Tune(best_score, &new_best_score);
        best_score = new_best_score;

        if (!finished) {
          finished_tuning = false;
          break;
        }
      }

      if (finished_tuning) {
        SetAutoTuning(false);
        LogBestParameters();
      }
    }

    // Send the updated parameter values to other workers.
    return true;
  }

  // Prepare for the next round of collecting statistics.
  Reset();
  return false;
}

ParameterManager::Params ParameterManager::GetParams() {
  Params params;
  if (active_) {
    // We're actively tuning, so send the current value.
    params.hierarchical_allreduce = hierarchical_allreduce_.Value();
    params.hierarchical_allgather = hierarchical_allgather_.Value();
    params.torus_allreduce = torus_allreduce_.Value();
    params.cache_enabled = cache_enabled_.Value();
    params.tensor_fusion_threshold = joint_params_.Value(fusion_buffer_threshold_mb);
    params.cycle_time = joint_params_.Value(cycle_time_ms);
  } else {
    // Tuning has completed, so send the best value.
    params.hierarchical_allreduce = hierarchical_allreduce_.BestValue();
    params.hierarchical_allgather = hierarchical_allgather_.BestValue();
    params.torus_allreduce = torus_allreduce_.BestValue();
    params.cache_enabled = cache_enabled_.BestValue();
    params.tensor_fusion_threshold = joint_params_.BestValue(fusion_buffer_threshold_mb);
    params.cycle_time = joint_params_.BestValue(cycle_time_ms);
  }

  params.active = active_;

  return params;
}

void ParameterManager::SetParams(const Params& newParams) {
  hierarchical_allreduce_.SetValue(newParams.hierarchical_allreduce, true);
  hierarchical_allgather_.SetValue(newParams.hierarchical_allgather, true);
  torus_allreduce_.SetValue(newParams.torus_allreduce, true);
  cache_enabled_.SetValue(newParams.cache_enabled, true);
  joint_params_.SetValue(fusion_buffer_threshold_mb, newParams.tensor_fusion_threshold, true);
  joint_params_.SetValue(cycle_time_ms, newParams.cycle_time, true);
  active_ = newParams.active;
}

void ParameterManager::Reset() {
  total_bytes_ = 0;
  last_sample_start_ = std::chrono::steady_clock::now();
  tensor_counts_.clear();
  sample_ = 0;
}

void ParameterManager::LogParameters(double score) {
  if (rank_ == root_rank_) {
    LOG(INFO) << "Autotuner: ["
              << hierarchical_allreduce_.Value() << ", "
              << hierarchical_allgather_.Value() << ", "
              << torus_allreduce_.Value() << ", "
              << cache_enabled_.Value() << ", "
              << joint_params_.Value(cycle_time_ms) << " ms, "
              << joint_params_.Value(fusion_buffer_threshold_mb) << " mb] "
              << score;
    if (writing_ && file_.good()) {
      file_ << hierarchical_allreduce_.Value() << ","
            << hierarchical_allgather_.Value() << ","
            << torus_allreduce_.Value() << ", "
            << cache_enabled_.Value() << ","
            << joint_params_.Value(cycle_time_ms) << ","
            << joint_params_.Value(fusion_buffer_threshold_mb) << ","
            << score
            << std::endl;
    }
  }
}

void ParameterManager::LogBestParameters() {
  if (rank_ == root_rank_) {
    LOG(INFO) << "Autotuner: Best params ["
              << hierarchical_allreduce_.BestValue() << ", "
              << hierarchical_allgather_.BestValue() << ", "
              << torus_allreduce_.BestValue() << ", "
              << cache_enabled_.BestValue() << ", "
              << joint_params_.BestValue(cycle_time_ms) << " ms, "
              << joint_params_.BestValue(fusion_buffer_threshold_mb) << " mb] "
              << hierarchical_allreduce_.BestScore();
    if (writing_ && file_.good()) {
      file_ << hierarchical_allreduce_.BestValue() << ","
            << hierarchical_allgather_.BestValue() << ","
            << torus_allreduce_.BestValue() << ", "
            << cache_enabled_.BestValue() << ","
            << joint_params_.BestValue(cycle_time_ms) << ","
            << joint_params_.BestValue(fusion_buffer_threshold_mb) << ","
            << hierarchical_allreduce_.BestScore()
            << std::endl;
    }
  }
}

// TunableParameter
template <class T>
ParameterManager::TunableParameter<T>::TunableParameter(T initial_value) :
    initial_value_(initial_value),
    value_(initial_value),
    best_value_(initial_value),
    best_score_(0),
    tunable_(true) {}

template <class T>
bool ParameterManager::TunableParameter<T>::Tune(double score, double* best_score) {
  UpdateBestValue(score);
  *best_score = best_score_;

  if (!tunable_) {
    return true;
  }

  OnTune(score, value_);
  if (IsDoneTuning()) {
    CompleteTuning();
    return true;
  }

  return false;
}

template <class T>
void ParameterManager::TunableParameter<T>::UpdateBestValue(double score) {
  if (score > best_score_) {
    best_score_ = score;
    best_value_ = value_;
  }
}

template <class T>
void ParameterManager::TunableParameter<T>::SetValue(T value, bool fixed) {
  best_value_ = value;
  if (fixed) {
    value_ = value;
    tunable_ = false;
  }
}

template <class T>
void ParameterManager::TunableParameter<T>::SetCurrentValue(T value) {
  value_ = value;
}

template <class T>
void ParameterManager::TunableParameter<T>::SetBestValue(T value) {
  best_value_ = value;
}

template <class T>
void ParameterManager::TunableParameter<T>::SetInitialValue(T value) {
  initial_value_ = value;
}

template <class T>
void ParameterManager::TunableParameter<T>::Reinitialize(T value) {
  initial_value_ = value;
  value_ = value;
  best_value_ = value;
  best_score_ = 0;
}

template <class T>
void ParameterManager::TunableParameter<T>::CompleteTuning() {
  value_ = initial_value_;
  ResetState();
}

// CategoricalParameter
template <class T>
ParameterManager::CategoricalParameter<T>::CategoricalParameter(std::vector<T> values) :
    TunableParameter<T>(values[0]),
    values_(values) {
  ResetState();
}

template <class T>
void ParameterManager::CategoricalParameter<T>::OnTune(double score, T& value) {
  ++index_;
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

// BayesianParameter
ParameterManager::BayesianParameter::BayesianParameter(
    std::vector<BayesianVariableConfig> variables,
    std::vector<Eigen::VectorXd> test_points,
    int max_samples,
    double gaussian_process_noise) :
    TunableParameter<Eigen::VectorXd>(test_points[0]),
    variables_(variables),
    test_points_(test_points),
    max_samples_(max_samples),
    gaussian_process_noise_(gaussian_process_noise),
    iteration_(0) {
  ResetBayes();
  Reinitialize(FilterTestPoint(0));
  ResetState();
}

void ParameterManager::BayesianParameter::SetValue(BayesianVariable variable, double value, bool fixed) {
  if (fixed) {
    // Only remove this variable if it hasn't already been fixed
    if (fixed_values_.find(variable) == fixed_values_.end()) {
      // Fixed parameter values cannot be changed, and will be removed from the Bayesian optimization
      // process so the search space can be reduced. To remove the parameter from the optimizer, we need
      // to also remove it from the vector outputs of the optimization process. First we find the index
      // of the variable we're removing in the existing vectors, then for each of the current, best, and
      // initial value vectors, we remove that index to create a smaller vector, and reset those values.
      int32_t index = index_[variable];
      TunableParameter::SetCurrentValue(Remove(TunableParameter::Value(), index));
      TunableParameter::SetBestValue(Remove(TunableParameter::BestValue(), index));
      TunableParameter::SetInitialValue(Remove(TunableParameter::InitialValue(), index));
    }

    fixed_values_[variable] = value;
    ResetBayes();
  } else {
    Eigen::VectorXd v = TunableParameter::BestValue();
    v[index_[variable]] = value;
    TunableParameter::SetValue(v, false);
  }
}

double ParameterManager::BayesianParameter::Value(BayesianVariable variable) const {
  auto elem = fixed_values_.find(variable);
  if (elem != fixed_values_.end()) {
    return elem->second;
  }
  return TunableParameter::Value()(index_.at(variable));
}

double ParameterManager::BayesianParameter::BestValue(BayesianVariable variable) const {
  auto elem = fixed_values_.find(variable);
  if (elem != fixed_values_.end()) {
    return elem->second;
  }
  return TunableParameter::BestValue()(index_.at(variable));
}

void ParameterManager::BayesianParameter::OnTune(double score, Eigen::VectorXd& value) {
  bayes_->AddSample(value, score);

  ++iteration_;
  if (iteration_ < test_points_.size()) {
    value = FilterTestPoint(iteration_);
  } else {
    value = bayes_->NextSample();
  }
}

bool ParameterManager::BayesianParameter::IsDoneTuning() const {
  return (int)iteration_ > max_samples_;
}

void ParameterManager::BayesianParameter::ResetState() {
  iteration_ = 0;
  bayes_->Clear();
}

void ParameterManager::BayesianParameter::ResetBayes() {
  index_.clear();

  std::vector<std::pair<double, double>> bounds;
  int j = 0;
  for (auto var : variables_) {
    if (fixed_values_.find(var.variable) == fixed_values_.end()) {
      bounds.push_back(var.bounds);
      index_[var.variable] = j;
      ++j;
    }
  }

  bayes_.reset(new BayesianOptimization(bounds, gaussian_process_noise_));
}

Eigen::VectorXd ParameterManager::BayesianParameter::FilterTestPoint(int i) {
  Eigen::VectorXd& test_point = test_points_[i];
  Eigen::VectorXd filtered_point(test_point.size() - fixed_values_.size());

  int k = 0;
  for (int j = 0; j < test_point.size(); ++j) {
    BayesianVariable variable = variables_[j].variable;
    if (fixed_values_.find(variable) == fixed_values_.end()) {
      filtered_point(k) = test_point(j);
      ++k;
    }
  }

  return filtered_point;
}

Eigen::VectorXd ParameterManager::BayesianParameter::Remove(const Eigen::VectorXd& v, int index) {
  if (v.size() == 0) {
    // Vector is already empty, nothing to do
    return v;
  }

  Eigen::VectorXd filtered_point(v.size() - 1);

  int k = 0;
  for (int j = 0; j < v.size(); ++j) {
    if (j == index) {
      continue;
    }

    filtered_point(k) = v(j);
    ++k;
  }

  return filtered_point;
}

} // namespace common
} // namespace horovod
