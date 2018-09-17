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

#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace horovod {
namespace common {

// ParameterManager encapsulates the various tunable "knobs" in Horovod including the cycle time
// between iterations of the background thread, and the size of the fusion buffer.
//
// During the early training batches, the auto-tuning feature (if enabled) will try various
// combinations of parameters in search of the combination that yields the highest throughput
// in units of bytes allreduced per second.
//
// Once the auto-tuner has converged to find the highest scoring combination of parameters, the tuning
// will end and the returned values will always be equal to the best scoring.
class ParameterManager {
public:
  ParameterManager();

  void Initialize(int32_t rank, int32_t root_rank, std::string file_name);
  void SetAutoTuning(bool active);

  inline bool IsAutoTuning() const {
    return active_;
  }

  // Threshold for Tensor Fusion.  All tensors that occupy memory beyond this
  // threshold will be fused.
  int64_t TensorFusionThresholdMb() const;
  void SetTensorFusionThresholdMb(int64_t threshold);

  // Background thread cycle time in milliseconds.  Fractional numbers are
  // permitted.
  double CycleTimeMs() const;
  void SetCycleTimeMs(double cycle_time_ms);

  void Update(const std::vector<std::string>& tensor_names, int64_t bytes, double seconds);

private:
  void Tune(double score);
  void ReadyTune();

  template <class T>
  struct ParameterScore {
    T value;
    double score;
  };

  class ITunableParameter {
  public:
    virtual void Tune(double score) = 0;
    virtual double BestScore() const = 0;
  };

  template <class T>
  class TunableParameter : public ITunableParameter {
  public:
    TunableParameter(T initial_value, ParameterManager& parent, ITunableParameter* const next_param);
    void Tune(double score) override;

    void SetValue(T value);
    inline T Value() const { return value_; };
    inline T BestValue() const { return best_value_; };
    inline double BestScore() const override { return best_score_; };

  private:
    void CompleteTuning();
    virtual void OnTune(double score, T& value) = 0;
    virtual bool IsDoneTuning() const = 0;
    virtual void ResetState() = 0;

    T initial_value_;
    T value_;

    T best_value_;
    double best_score_;

    ParameterManager& parent_;
    ITunableParameter* const next_param_;
  };

  template <class T>
  class NumericParameter : public TunableParameter<T> {
  public:
    NumericParameter(T low, T high, ParameterManager& parent, ITunableParameter* const next_param);

  private:
    void OnTune(double score, T& value) override;
    bool IsDoneTuning() const override;
    void ResetState() override;

    T low_;
    T high_;
    ParameterScore<T> left_;
    ParameterScore<T> right_;

    double h_;
    int32_t n_;
    int32_t k_;
  };

  template <class T>
  class CategoricalParameter : public TunableParameter<T> {
  public:
    CategoricalParameter(std::vector<T> values, ParameterManager& parent, ITunableParameter* const next_param);

  private:
    void OnTune(double score, T& value);
    bool IsDoneTuning() const;
    void ResetState();

    std::vector<T> values_;
    int32_t index_;
  };

//  NumericParameter<int64_t> tensor_fusion_threshold_;
//  NumericParameter<double> cycle_time_ms_;

  CategoricalParameter<int64_t> tensor_fusion_threshold_;
  CategoricalParameter<double> cycle_time_ms_;

  ITunableParameter* const leaf_param_;
  bool active_;
  int32_t warmup_remaining_;

  static constexpr int CYCLES = 9;
  double scores_[CYCLES];
  int32_t cycle_;

  int64_t total_bytes_;
  double total_seconds_;
  std::unordered_map<std::string, int32_t> tensor_counts_;

  int32_t rank_;
  int32_t root_rank_;
  std::ofstream file_;
  bool writing_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_PARAMETER_MANAGER_H
