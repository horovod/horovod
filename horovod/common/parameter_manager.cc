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

#include "mpi.h"

namespace horovod {
namespace common {

#define WARMUPS 3
#define CYCLES_PER_SAMPLE 10

Eigen::VectorXd CreateVector(double x1, double x2) {
  Eigen::VectorXd v(2);
  v(0) = x1;
  v(1) = x2;
  return v;
}

// ParameterManager
ParameterManager::ParameterManager() :
    hierarchical_allreduce_(CategoricalParameter<bool>(std::vector<bool>{false, true}, *this, nullptr)),
    joint_params_(BayesianParameter(
      std::vector<BayesianVariableConfig>{
        { BayesianVariable::fusion_buffer_threshold_mb, std::pair<double, double>(0, 64) },
        { BayesianVariable::cycle_time_ms, std::pair<double, double>(1, 100) }
      }, std::vector<Eigen::VectorXd>{
        CreateVector(4, 5),
        CreateVector(32, 50),
        CreateVector(16, 25),
        CreateVector(8, 10)
      }, *this, &hierarchical_allreduce_)),
    leaf_param_(&joint_params_),
    active_(false),
    warmup_remaining_(WARMUPS),
    sample_(0),
    rank_(-1),
    root_rank_(0),
    writing_(false) {
  ReadyTune();
}

void ParameterManager::CreateMpiTypes() {
  const int nitems = 4;
  int blocklengths[4] = {1, 1, 1, 1};
  MPI_Datatype types[4] = {MPI_CXX_BOOL, MPI_DOUBLE, MPI_DOUBLE, MPI_CXX_BOOL};

  MPI_Aint offsets[4];
  offsets[0] = offsetof(Params, hierarchical_allreduce);
  offsets[1] = offsetof(Params, tensor_fusion_threshold);
  offsets[2] = offsetof(Params, cycle_time);
  offsets[3] = offsetof(Params, active);

  MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_params_type_);
  MPI_Type_commit(&mpi_params_type_);
}

void ParameterManager::FreeMpiTypes() {
  if (mpi_params_type_ != MPI_DATATYPE_NULL) {
    MPI_Type_free(&mpi_params_type_);
  }
}

void ParameterManager::Initialize(int32_t rank, int32_t root_rank, MPI_Comm mpi_comm, std::string file_name) {
  rank_ = rank;
  root_rank_ = root_rank;
  mpi_comm_ = mpi_comm;
  if (rank_ == root_rank && !file_name.empty()) {
    file_.open(file_name, std::ios::out | std::ios::trunc);
    if (file_.good()) {
      file_ << "hierarchical_allreduce,cycle_time_ms,tensor_fusion_threshold,score" << std::endl;
      writing_ = true;
    }
  }
}

void ParameterManager::SetAutoTuning(bool active) {
  if (active != active_) {
    warmup_remaining_ = WARMUPS;
  }
  active_ = active;
  if (!active_ && rank_ == root_rank_) {
    std::cerr << "Horovod Tune: BEST [ "
              << hierarchical_allreduce_.BestValue() << ", "
              << joint_params_.BestValue(cycle_time_ms) << " ms , "
              << joint_params_.BestValue(fusion_buffer_threshold_mb) << " mb ] "
              << hierarchical_allreduce_.BestScore()
              << std::endl;
  }
};

bool ParameterManager::HierarchicalAllreduce() const {
  return active_ ? hierarchical_allreduce_.Value() : hierarchical_allreduce_.BestValue();
}

void ParameterManager::SetHierarchicalAllreduce(bool value, bool fixed) {
  hierarchical_allreduce_.SetValue(value, fixed);
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

void ParameterManager::Update(const std::vector<std::string>& tensor_names, int64_t bytes, double microseconds) {
  if (!active_) {
    return;
  }

  for (const std::string& tensor_name : tensor_names) {
    int32_t cycle = tensor_counts_[tensor_name]++;
    if (cycle > sample_ * CYCLES_PER_SAMPLE) {
      scores_[sample_] = total_bytes_ / total_microseconds_;
      total_bytes_ = 0;
      total_microseconds_ = 0;
      sample_++;
      break;
    }
  }

  total_bytes_ += bytes;
  total_microseconds_ += microseconds;

  if (sample_ >= SAMPLES) {
    std::sort(scores_, scores_ + SAMPLES);
    double med_score = scores_[SAMPLES / 2];
    Tune(med_score);
  }
}

void ParameterManager::Tune(double score) {
  if (warmup_remaining_ > 0) {
    warmup_remaining_--;
    if (rank_ == root_rank_) {
      std::cerr << "Horovod Tune: WARMUP DONE (" << warmup_remaining_ << " remaining)" << std::endl;
    }
  } else {
    if (rank_ == root_rank_) {
      std::cerr << "Horovod Tune: [" << hierarchical_allreduce_.Value() << ", "
                << joint_params_.Value(cycle_time_ms) << " ms , " << joint_params_.Value(fusion_buffer_threshold_mb) << " mb ] "
                << score
                << std::endl;
      if (writing_ && file_.good()) {
        file_ << hierarchical_allreduce_.Value() << ","
              << joint_params_.Value(cycle_time_ms) << ","
              << joint_params_.Value(fusion_buffer_threshold_mb) << ","
              << score
              << std::endl;
      }

      leaf_param_->Tune(score);
    }

    SyncParams();
  }
  ReadyTune();
}

void ParameterManager::ReadyTune() {
  total_bytes_ = 0;
  total_microseconds_ = 0;
  tensor_counts_.clear();
  sample_ = 0;
}

void ParameterManager::SyncParams() {
  Params params;
  if (rank_ == root_rank_) {
    if (active_) {
      params.hierarchical_allreduce = hierarchical_allreduce_.Value();
      params.tensor_fusion_threshold = joint_params_.Value(fusion_buffer_threshold_mb);
      params.cycle_time = joint_params_.Value(cycle_time_ms);
    } else {
      params.hierarchical_allreduce = hierarchical_allreduce_.BestValue();
      params.tensor_fusion_threshold = joint_params_.BestValue(fusion_buffer_threshold_mb);
      params.cycle_time = joint_params_.BestValue(cycle_time_ms);
    }

    params.active = active_;
  }

  MPI_Bcast(&params, 1, mpi_params_type_, root_rank_, mpi_comm_);
  if (rank_ != root_rank_) {
    hierarchical_allreduce_.SetValue(params.hierarchical_allreduce, true);
    joint_params_.SetValue(fusion_buffer_threshold_mb, params.tensor_fusion_threshold, true);
    joint_params_.SetValue(cycle_time_ms, params.cycle_time, true);
    active_ = params.active;
  }
}

// TunableParameter
template <class T>
ParameterManager::TunableParameter<T>::TunableParameter(
    T initial_value, ParameterManager &parent, ITunableParameter* const next_param) :
    initial_value_(initial_value),
    value_(initial_value),
    best_value_(initial_value),
    best_score_(0),
    tunable_(true),
    parent_(parent),
    next_param_(next_param) {}

template <class T>
void ParameterManager::TunableParameter<T>::Tune(double score) {
  UpdateBestValue(score);
  if (!tunable_) {
    TuneNextParameter();
    return;
  }

  OnTune(score, value_);
  if (IsDoneTuning()) {
    CompleteTuning();
  }
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
void ParameterManager::TunableParameter<T>::Reinitialize(T value) {
  initial_value_ = value;
  value_ = value;
  best_value_ = value;
  best_score_ = 0;
}

template <class T>
void ParameterManager::TunableParameter<T>::TuneNextParameter() {
  if (next_param_ != nullptr) {
    next_param_->Tune(best_score_);
  } else {
    parent_.SetAutoTuning(false);
  }
}

template <class T>
void ParameterManager::TunableParameter<T>::CompleteTuning() {
  TuneNextParameter();
  value_ = initial_value_;
  ResetState();
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

// BayesianParameter
ParameterManager::BayesianParameter::BayesianParameter(
    std::vector<BayesianVariableConfig> variables,
    std::vector<Eigen::VectorXd> test_points,
    ParameterManager& parent,
    ParameterManager::ITunableParameter* const next_param) :
    TunableParameter<Eigen::VectorXd>(test_points[0], parent, next_param),
    variables_(variables),
    test_points_(test_points),
    iteration_(0) {
  ResetBayes();
  ResetState();
}

void ParameterManager::BayesianParameter::SetValue(BayesianVariable variable, double value, bool fixed) {
  if (fixed) {
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

  iteration_++;
  if (iteration_ < test_points_.size()) {
    value = FilterTestPoint(iteration_);
  } else {
    value = bayes_->NextSample();
  }
}

bool ParameterManager::BayesianParameter::IsDoneTuning() const {
  unsigned long d = bayes_->Dim();
  return iteration_ > 20 * d;
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
      j++;
    }
  }

  bayes_.reset(new BayesianOptimization(bounds, 0.2));
  Reinitialize(FilterTestPoint(0));
}

Eigen::VectorXd ParameterManager::BayesianParameter::FilterTestPoint(int i) {
  Eigen::VectorXd& test_point = test_points_[i];
  Eigen::VectorXd filtered_point(test_point.size() - fixed_values_.size());

  int k = 0;
  for (int j = 0; j < test_point.size(); j++) {
    BayesianVariable variable = variables_[j].variable;
    if (fixed_values_.find(variable) == fixed_values_.end()) {
      filtered_point(k) = test_point(j);
      k++;
    }
  }

  return filtered_point;
}

} // namespace common
} // namespace horovod