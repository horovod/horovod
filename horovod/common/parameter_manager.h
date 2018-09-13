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

#include <iostream>
#include <unordered_map>
#include <vector>

namespace horovod {
namespace common {

class ParameterManager {
public:
  ParameterManager();

  void Update(const std::vector<std::string>& tensor_names, int64_t bytes, double seconds);

  inline void SetAutoTuning(bool active) {
    active_ = active;
  };

  inline const bool IsAutoTuning() {
    return active_;
  }

  // Threshold for Tensor Fusion.  All tensors that occupy memory beyond this
  // threshold will be fused.
  int64_t TensorFusionThreshold();
  void SetTensorFusionThreshold(int64_t threshold);

  // Background thread cycle time in milliseconds.  Fractional numbers are
  // permitted.
  double CycleTimeMs();
  void SetCycleTimeMs(double cycle_time_ms);

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
  };

  template <class T>
  class TunableParameter : public ITunableParameter {
  public:
    TunableParameter(T initial_value, ParameterManager& parent, ITunableParameter* const next_param);
    virtual void Tune(double score);

    void SetValue(T value);
    inline T Value() { return value_; };
    inline T BestValue() { return best_value_; };

  private:
    void CompleteTuning();
    virtual void OnTune(double score, T& value) = 0;
    virtual bool IsDoneTuning() = 0;
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
    void OnTune(double score, T& value);
    bool IsDoneTuning();
    void ResetState();

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
    bool IsDoneTuning();
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

  double samples_;
  double sum_score_;
  int64_t total_bytes_;
  std::unordered_map<std::string, int32_t> tensor_counts_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_PARAMETER_MANAGER_H
