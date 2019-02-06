// Copyright 2018 Martin Krasser. All Rights Reserved.
// Modifications copyright (C) 2018 Uber Technologies, Inc.
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

#ifndef HOROVOD_BAYESIAN_OPTIMIZATION_H
#define HOROVOD_BAYESIAN_OPTIMIZATION_H

#include <random>

#include <Eigen/Core>

#include "gaussian_process.h"

namespace horovod {
namespace common {

// Bayesian Optimization attempts to find the global optimum in a minimum number of steps, by incorporating
// prior belief about the objective function. It updates the prior with samples drawn from the objective function
// to get a posterior that better approximates that objective function. The model used for approximating the objective
// function is called surrogate model. In this implementation, we use Gaussian processes for our surrogate model.
//
// Bayesian optimization also uses an acquisition function that directs sampling to areas where an improvement
// over the current best observation is likely.  Acquisition functions trade-off between exploration (sampling
// where uncertainty is high) and exploitation (sampling where the surrogate model predicts a high objective).
//
// This implementation is based on the scikit-learn GaussianProcessRegressor and the blog
// by Martin Krasser on Gaussian Processes and is an adaptation of Python code to C++.
//
// This implementation is based on the blog by Martin Krasser on Bayesian Optimization and is
// an adaptation of the Python + NumPy code to C++.
//
// See: http://krasserm.github.io/2018/03/21/bayesian-optimization
class BayesianOptimization {
public:
  // Performs binary optimization over the observed data by predicting the next sample to evaluate.
  //
  // Args:
  //  bounds: Vector of (min, max) range values for each parameter (d x 1).
  //  alpha: Gaussian process noise parameter (see GaussianProcessRegressor).
  //  xi: Exploitation-exploration trade-off parameter, increase to explore more of the space.
  BayesianOptimization(std::vector<std::pair<double, double>> bounds, double alpha, double xi=0.01);

  // Returns the dimensionality of the parameter vector (number of parameters).
  inline unsigned long Dim() const { return d_; };

  // Adds an observed sample and its objective value.
  //
  // Args:
  //  x: Sample point tested (d x 1).
  //  y: Evaluated objective value at x.
  void AddSample(const Eigen::VectorXd& x, double y);

  // Provides the next sample point to evaluate subject to maximizing the
  // expected improvement of the target acquisition function.
  Eigen::VectorXd NextSample(bool normalize=true);

  // Reset the state of the optimizer by clearing all samples.
  void Clear();

private:
  // Proposes the next sampling point by optimizing the acquisition function.
  //
  // Args:
  //  acquisition: Acquisition function.
  //  x_sample: Sample locations (n x d).
  //  y_sample: Sample values (n x 1).
  //  n_restarts: How many times to run minimization routine with random restarts.
  //
  // Returns: Location of the acquisition function maximum.
  Eigen::VectorXd ProposeLocation(
      const Eigen::MatrixXd& x_sample, const Eigen::MatrixXd& y_sample, int n_restarts=25);

  // Computes the Expected Improvement at points X based on existing samples X_sample and Y_sample
  // using a Gaussian process surrogate model fitted to the samples.
  //
  // Args:
  //  x: Proposed points at which EI shall be computed (m x d).
  //  x_sample: Sample locations observed (n x d).
  //
  // Returns: Expected improvements at points X.
  Eigen::VectorXd ExpectedImprovement(const Eigen::MatrixXd& x, const Eigen::MatrixXd& x_sample);

  // Returns true if all elements of the vector are within the respective bounds for its dimension.
  bool CheckBounds(const Eigen::VectorXd& x);

  unsigned long d_;  // Dimension of the input data.
  std::vector<std::pair<double, double>> bounds_;
  double xi_;

  std::random_device rd_;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen_ = std::mt19937(rd_()); // Standard mersenne_twister_engine seeded with random_device.
  std::vector<std::uniform_real_distribution<>> dists_;

  GaussianProcessRegressor gpr_;
  std::vector<Eigen::VectorXd> x_samples_;
  std::vector<double> y_samples_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_BAYESIAN_OPTIMIZATION_H
