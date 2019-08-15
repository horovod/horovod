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

#include "bayesian_optimization.h"

#include <cmath>
#include <iostream>
#include <numeric>

#include <Eigen/LU>
#include "LBFGS.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace horovod {
namespace common {

const double NORM_PDF_C = std::sqrt(2 * M_PI);

void GetSufficientStats(std::vector<double>& v, double* mu, double* sigma) {
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  *mu = sum / v.size();

  std::vector<double> diff(v.size());
  std::transform(v.begin(), v.end(), diff.begin(), [mu](double& x) { return x - *mu; });
  double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  *sigma = std::sqrt(sq_sum / v.size());
}

// Returns a list of distributions that generate real values uniformly and random between the bounds.
std::vector<std::uniform_real_distribution<>> GetDistributions(std::vector<std::pair<double, double>> bounds) {
  std::vector<std::uniform_real_distribution<>> dists;
  for (const std::pair<double, double>& bound : bounds) {
    dists.push_back(std::uniform_real_distribution<>(bound.first, bound.second));
  }
  return dists;
}


BayesianOptimization::BayesianOptimization(std::vector<std::pair<double, double>> bounds, double alpha, double xi)
    : d_(bounds.size()),
      bounds_(bounds),
      xi_(xi),
      dists_(GetDistributions(bounds)),
      gpr_(GaussianProcessRegressor(alpha)) {}

void BayesianOptimization::AddSample(const Eigen::VectorXd& x, double y) {
  x_samples_.push_back(x);
  y_samples_.push_back(y);
}

VectorXd BayesianOptimization::NextSample(bool normalize) {
  double mu = 0.0;
  double sigma = 1.0;
  if (normalize && y_samples_.size() >= 3) {
    GetSufficientStats(y_samples_, &mu, &sigma);
  }

  // Matrices are immutable and must be regenerated each time a new sample is added.
  MatrixXd x_sample(x_samples_.size(), d_);
  for (unsigned int i = 0; i < x_samples_.size(); ++i) {
    x_sample.row(i) = x_samples_[i];
  }

  MatrixXd y_sample(y_samples_.size(), 1);
  for (unsigned int i = 0; i < y_samples_.size(); ++i) {
    double norm_score = (y_samples_[i] - mu) / sigma;

    VectorXd y_i(1);
    y_i(0) = norm_score;
    y_sample.row(i) = y_i;
  }

  // Generate the posterior distribution for the GP given the observed data.
  gpr_.Fit(&x_sample, &y_sample);

  // Return the next proposed location that maximizes the expected improvement.
  return ProposeLocation(x_sample, y_sample);
}

void BayesianOptimization::Clear() {
  x_samples_.clear();
  y_samples_.clear();
}

VectorXd BayesianOptimization::ProposeLocation(const MatrixXd& x_sample, const MatrixXd& y_sample, int n_restarts) {
  // Objective function we wish to minimize, the negative acquisition function.
  auto f = [&](const VectorXd& x) {
    return -ExpectedImprovement(x.transpose(), x_sample)[0];
  };

  // Minimization routine. To approximate bounded LBFGS, we set to infinity the value of any input outside of bound.
  auto min_obj = [&](const VectorXd& x, VectorXd& grad) {
    double fx = CheckBounds(x) ? f(x) : std::numeric_limits<double>::max();
    GaussianProcessRegressor::ApproxFPrime(x, f, fx, grad);
    return fx;
  };

  // Use the L-BFGS method for minimizing the objective, limit our search to a set number of iterations
  // if convergence has not be reached within the threshold (epsilon).
  LBFGSpp::LBFGSParam<double> param;
  param.epsilon = 1e-5;
  param.max_iterations = 100;
  LBFGSpp::LBFGSSolver<double> solver(param);

  // Optimize with random restarts to avoid getting stuck in local minimum.
  VectorXd x_next = VectorXd::Zero(d_);
  double fx_min = std::numeric_limits<double>::max();
  for (int i = 0; i < n_restarts; ++i) {
    // Generate a random starting point by drawing from our bounded distributions.
    VectorXd x = VectorXd::Zero(d_);
    for (unsigned int j = 0; j < d_; ++j) {
      x[j] = dists_[j](gen_);
    }

    // Minimize the objective function.
    double fx;
    solver.minimize(min_obj, x, fx);

    // Update the new minimum among all attempts.
    if (fx < fx_min) {
      fx_min = fx;
      x_next = x;
    }
  }

  // Return the input point that minimized the negative expected improvement.
  return x_next;
}

VectorXd BayesianOptimization::ExpectedImprovement(const MatrixXd& x, const MatrixXd& x_sample) {
  // Compute sufficient statistics for the proposed locations.
  Eigen::VectorXd mu;
  Eigen::VectorXd sigma;
  gpr_.Predict(x, mu, &sigma);

  // Compute sufficient statistics for the observed locations.
  Eigen::VectorXd mu_sample;
  gpr_.Predict(x_sample, mu_sample);

  // Needed for noise-based model, otherwise use y_sample.maxCoeff().
  // See also section 2.4 in https://arxiv.org/pdf/1012.2599.pdf:
  // Eric Brochu, Vlad M. Cora, Nando de Freitas,
  // A Tutorial on Bayesian Optimization of Expensive Cost Functions
  double mu_sample_opt = mu_sample.maxCoeff();

  // Probability density function of the standard normal distribution.
  auto pdf = [](double x) {
    return std::exp(-(x * x) / 2.0) / NORM_PDF_C;
  };

  // Cumulative distribution function of the standard normal distribution.
  auto cdf = [](double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
  };

  // Parameter xi_ determines the amount of exploration during optimization. Higher values of xi_ results
  // in more exploration. With higher values of xi_, the importance of improvements predicted by the
  // underlying GP posterior mean mu_sample_opt decreases relative to the importance of improvements
  // in regions of high prediction uncertainty, as indicated by large values of variable sigma.
  Eigen::VectorXd imp = mu.array() - mu_sample_opt - xi_;
  VectorXd z = imp.array() / sigma.array();

  // The first term of the summation is the exploitation term, the second the exploration term.
  VectorXd ei = imp.cwiseProduct(z.unaryExpr(cdf)) + sigma.cwiseProduct(z.unaryExpr(pdf));
  ei = (sigma.array() != 0).select(ei, 0.0);
  return ei;
}

bool BayesianOptimization::CheckBounds(const Eigen::VectorXd& x) {
  for (int i = 0; i < x.size(); ++i) {
    if (x[i] < bounds_[i].first || x[i] > bounds_[i].second) {
      return false;
    }
  }
  return true;
}

} // namespace common
} // namespace horovod
