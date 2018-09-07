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

namespace horovod {
namespace common {

class ParameterManager {
public:
  ParameterManager();

  void Update(int64_t bytes, double seconds);

  inline void SetAutoTuning(bool active) {
    active_ = active;
  };

  // Threshold for Tensor Fusion.  All tensors that occupy memory beyond this
  // threshold will be fused.
  int64_t TensorFusionThreshold();
  void SetTensorFusionThreshold(int64_t threshold);

  // Background thread cycle time in milliseconds.  Fractional numbers are
  // permitted.
  double CycleTimeMs();
  void SetCycleTimeMs(double cycle_time_ms);

private:

  void Step();

  template <class T>
  struct ParameterScore {
    T value;
    double score;
  };

  class ITunableParameter {
  public:
    virtual void Step(double score, double samples) = 0;
    virtual void Tune(double score) = 0;
  };

  template <class T>
  class NumericParameter : public ITunableParameter {
  public:
    NumericParameter(T initial_value, T low, T high, T epsilon,
                     ParameterManager& parent, ITunableParameter* const next_param);
    void SetValue(T value);
    void Step(double score, double samples);
    void Tune(double score);
    inline T Value() { return value_; };
    inline T BestValue() { return best_value_; };

  private:
    void CheckGradient();
    void DoneTune();
    void ResetState();

    T low_;
    T high_;
    T epsilon_;

    ParameterScore<T> current_;
    ParameterScore<T> left_;
    ParameterScore<T> right_;

    T value_;
    int32_t depth_;
    double samples_;
    double sum_score_;

    T best_value_;
    double best_score_;

    ParameterManager& parent_;
    ITunableParameter* const next_param_;
  };

  NumericParameter<int64_t> tensor_fusion_threshold_;
  NumericParameter<double> cycle_time_ms_;
  ITunableParameter* const leaf_param_;
  bool active_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_PARAMETER_MANAGER_H
