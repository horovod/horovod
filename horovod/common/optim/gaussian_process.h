// Copyright (c) 2007–2018 The scikit-learn developers. All rights reserved.
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

#ifndef HOROVOD_GAUSSIAN_PROCESS_H
#define HOROVOD_GAUSSIAN_PROCESS_H

#include <vector>

#include <Eigen/Cholesky>

namespace horovod {
namespace common {

// A Gaussian Process is a non-parametric regression model used to estimate the uncertainty
// of predictions. A random process, any point x we observe is given a random variable f(x)
// where the joint distribution over these variables p(f(x1), f(x2), ..., f(xn)) is also
// Gaussian. Thus, a Gaussian Process is a distribution over functions, whose smoothness
// is defined by a kernel function (covariance function). For any two points xi and xj, if
// they are considered close by the kernel function, then we would expect that f(xi) and
// f(xj) would be similar as well.
//
// We use Gaussian Processes to infer functions directly as an alternative to inferring
// point estimates of the functions or posterior distributions. A Gaussian Process defines
// a prior over functions that we can convert into a posterior after observing some inputs
// X and outputs y of the function we are attempting to model. We can then use this posterior
// to make new predictions given new input data.
//
// This implementation is based on the blog by Martin Krasser on Gaussian Processes, along with
// scikit-learn, and is an adaptation of the Python + NumPy code to C++.
//
// See: http://krasserm.github.io/2018/03/19/gaussian-processes
class GaussianProcessRegressor {
public:
  // The implementation is based on Algorithm 2.1 of Gaussian Processes for
  // Machine Learning (GPML) by Rasmussen and Williams.
  //
  // Args:
  //  alpha: Value added to the diagonal of the kernel matrix during fitting.
  //         Larger values correspond to increased noise level in the observations.
  //         This can also prevent a potential numerical issue during fitting, by
  //         ensuring that the calculated values form a positive definite matrix.
  GaussianProcessRegressor(double alpha);

  ~GaussianProcessRegressor() {}

  // Solve for the parameters (length, sigma_f) that best fit the observed training data given.
  void Fit(Eigen::MatrixXd* x_train, Eigen::MatrixXd* y_train);

  // Evaluate mean and (optional) variance at a point.
  void Predict(const Eigen::MatrixXd& x, Eigen::VectorXd& mu, Eigen::VectorXd* sigma=nullptr) const;

  // Computes the suffifient statistics of the GP posterior predictive distribution
  // from m training data X_train and Y_train and n new inputs X_s.
  //
  // Args:
  //  x_s: New input locations (n x d).
  //  x_train: Training locations (m x d).
  //  y_train: Training targets (m x 1).
  //  l: Kernel length parameter.
  //  sigma_f: Kernel vertical variation parameter.
  //  sigma_y: Noise parameter.
  //
  // Returns: Posterior mean vector (n x d) and covariance matrix (n x n).
  void PosteriorPrediction(const Eigen::MatrixXd& x_s, const Eigen::MatrixXd& x_train, const Eigen::MatrixXd& y_train,
                           Eigen::VectorXd& mu_s, Eigen::MatrixXd& cov_s,
                           double l=1.0, double sigma_f=1.0, double sigma_y=1e-8) const;

  // Finite-difference approximation of the gradient of a scalar function.
  static void ApproxFPrime(const Eigen::VectorXd& x, const std::function<double(const Eigen::VectorXd&)>& f,
                           double f0, Eigen::VectorXd& grad, double epsilon=1e-8);

  // Isotropic squared exponential kernel.
  // Computes a covariance matrix from points in X1 and X2.
  //
  // Args:
  //  x1: Matrix of m points (m x d).
  //  x2: Matrix of n points (n x d).
  //
  // Returns: Covariance matrix (m x n).
  Eigen::MatrixXd Kernel(const Eigen::MatrixXd& x1, const Eigen::MatrixXd& x2, double l=1.0, double sigma_f=1.0) const;

private:
  // Kernel parameter for noise. Higher values make more coarse approximations which avoids overfitting to noisy data.
  double alpha_;

  // Kernel parameter for smoothness. Higher values lead to smoother functions and therefore to coarser approximations
  // of the training data. Lower values make functions more wiggly with high confidence intervals between
  // training data points.
  double length_;

  // Kernel parameter that controls the vertical variation of functions drawn from the GP. Higher values lead to wider
  // confidence intervals.
  double sigma_f_;

  // These pointers are not owned.
  Eigen::MatrixXd* x_train_;
  Eigen::MatrixXd* y_train_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_GAUSSIAN_PROCESS_H
