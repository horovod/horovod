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

#ifndef HOROVOD_PARAMETER_MANAGER_H
#define HOROVOD_PARAMETER_MANAGER_H

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>

#include "optim/bayesian_optimization.h"

namespace horovod {
namespace common {

// ParameterManager encapsulates the various tunable "knobs" in Horovod including the cycle time
// between iterations of the background thread, and the size of the fusion buffer.
//
// During the early training batches, the auto-tuning feature (if enabled) will try various
// combinations of parameters in search of the combination that yields the highest throughput
// in units of bytes processed per second.
//
// Once the auto-tuner has converged to find the highest scoring combination of parameters, the tuning
// will end and the returned values will always be equal to the best scoring.
class ParameterManager {
public:
  ParameterManager();
  ParameterManager(const ParameterManager&) = delete;

  // Initializes this manager if auto tuning was requested.
  void Initialize(int32_t rank, int32_t root_rank, const std::string& file_name);

  // Starts or stop the auto tuning procedure.
  void SetAutoTuning(bool active);

  // Returns true if parameters are being actively tuned currently.
  inline bool IsAutoTuning() const {
    return active_;
  }

  // Do hierarchical allreduce.
  bool HierarchicalAllreduce() const;
  void SetHierarchicalAllreduce(bool value, bool fixed=false);

  // Do hierarchical allgather.
  bool HierarchicalAllgather() const;
  void SetHierarchicalAllgather(bool value, bool fixed=false);

  // Do torus allreduce.
  bool TorusAllreduce() const;
  void SetTorusAllreduce(bool value, bool fixed=false);

  // Threshold for Tensor Fusion.  All tensors that occupy memory beyond this
  // threshold will be fused.
  int64_t TensorFusionThresholdBytes() const;
  void SetTensorFusionThresholdBytes(int64_t threshold, bool fixed=false);

  // Background thread cycle time in milliseconds.  Fractional numbers are
  // permitted.
  double CycleTimeMs() const;
  void SetCycleTimeMs(double cycle_time_ms, bool fixed=false);

  // Enable response caching.
  bool CacheEnabled() const;
  void SetCacheEnabled (bool enabled, bool fixed=false);

  // Observes that the given tensors have been processed (e.g., allreduced) over the given number of microseconds.
  //
  // Args:
  //  tensor_names: The names of the tensors that have been processed.
  //  bytes: The number of bytes that were processed per worker.
  //
  // Return:
  //  Whether the parameters need to be broadcasted to all ranks.
  bool Update(const std::vector<std::string>& tensor_names, int64_t bytes);

  struct Params {
    bool hierarchical_allreduce;
    bool hierarchical_allgather;
    bool torus_allreduce;
    bool cache_enabled;
    double tensor_fusion_threshold;
    double cycle_time;
    bool active;
  };

  Params GetParams();

  // Using given params to update its own params.
  void SetParams(const Params& newParams);

  // Resets the tuning state in preparation for evaluating a new set of parameter values.
  void Reset();

private:
  // Adjusts the parameter values based on the last observed score.
  bool Tune(double score);

  // Outputs parameter values and writes results to a log file (if provided).
  void LogParameters(double score);
  void LogBestParameters();

  // Interface used to represent a parameter (or group of parameters) being tuned.
  class ITunableParameter {
  public:
    virtual bool Tune(double score, double* best_score) = 0;
    virtual void UpdateBestValue(double score) = 0;
    virtual double BestScore() const = 0;
    virtual bool IsTunable() const = 0;
  };

  // Abstract base class used to implement hierarchical parameter tuning.
  template <class T>
  class TunableParameter : public ITunableParameter {
  public:
    TunableParameter(T initial_value);
    bool Tune(double score, double* best_score) override;
    void UpdateBestValue(double score) override;

    void SetValue(T value, bool fixed);
    inline T Value() const { return value_; };
    inline T BestValue() const { return best_value_; };
    inline double BestScore() const override { return best_score_; };

    inline bool IsTunable() const override { return tunable_; };

  protected:
    inline T InitialValue() const { return initial_value_; };

    void SetCurrentValue(T value);
    void SetBestValue(T value);
    void SetInitialValue(T value);

    void Reinitialize(T value);

  private:
    void CompleteTuning();
    virtual void OnTune(double score, T& value) = 0;
    virtual bool IsDoneTuning() const = 0;
    virtual void ResetState() = 0;

    T initial_value_;
    T value_;

    T best_value_;
    double best_score_;

    bool tunable_;
  };

  // A parameter that optimizes over a finite set of discrete values to be tried sequentially.
  template <class T>
  class CategoricalParameter : public TunableParameter<T> {
  public:
    CategoricalParameter(std::vector<T> values);

  private:
    void OnTune(double score, T& value);
    bool IsDoneTuning() const;
    void ResetState();

    std::vector<T> values_;
    uint32_t index_;
  };

  enum BayesianVariable { fusion_buffer_threshold_mb, cycle_time_ms };

  struct BayesianVariableConfig {
    BayesianVariable variable;
    std::pair<double, double> bounds;
  };

  // A set of numerical parameters optimized jointly using Bayesian Optimization.
  class BayesianParameter : public TunableParameter<Eigen::VectorXd> {
  public:
    BayesianParameter(std::vector<BayesianVariableConfig> variables, std::vector<Eigen::VectorXd> test_points,
                      int max_samples, double gaussian_process_noise);

    void SetValue(BayesianVariable variable, double value, bool fixed);
    double Value(BayesianVariable variable) const;
    double BestValue(BayesianVariable variable) const;

  private:
    void OnTune(double score, Eigen::VectorXd& value);
    bool IsDoneTuning() const;
    void ResetState();
    void ResetBayes();
    Eigen::VectorXd FilterTestPoint(int i);
    Eigen::VectorXd Remove(const Eigen::VectorXd& v, int index);

    std::vector<BayesianVariableConfig> variables_;
    std::vector<Eigen::VectorXd> test_points_;
    int max_samples_;
    double gaussian_process_noise_;

    uint32_t iteration_;

    struct EnumClassHash {
      template <typename T>
      std::size_t operator()(T t) const {
        return static_cast<std::size_t>(t);
      }
    };

    std::unique_ptr<BayesianOptimization> bayes_;
    std::unordered_map<BayesianVariable, double, EnumClassHash> fixed_values_;
    std::unordered_map<BayesianVariable, int32_t, EnumClassHash> index_;
  };

  int warmups_;
  int steps_per_sample_;

  CategoricalParameter<bool> hierarchical_allreduce_;
  CategoricalParameter<bool> hierarchical_allgather_;
  CategoricalParameter<bool> torus_allreduce_;
  CategoricalParameter<bool> cache_enabled_;
  BayesianParameter joint_params_;

  std::vector<ITunableParameter*> parameter_chain_;
  bool active_;
  int32_t warmup_remaining_;

  static constexpr int SAMPLES = 5;
  double scores_[SAMPLES];
  int32_t sample_;

  int64_t total_bytes_;
  std::chrono::steady_clock::time_point last_sample_start_;
  std::unordered_map<std::string, int32_t> tensor_counts_;

  int32_t rank_;
  int32_t root_rank_;
  std::ofstream file_;
  bool writing_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_PARAMETER_MANAGER_H
