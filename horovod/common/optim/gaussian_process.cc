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

#include "gaussian_process.h"

#include <cmath>
#include <iostream>

#include <Eigen/LU>

#include "LBFGS.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;


namespace horovod {
namespace common {

bool isnan(const VectorXd& x) {
  for (int i = 0; i < x.size(); i++) {
    if (std::isnan(x[i])) {
      return true;
    }
  }
  return false;
}

GaussianProcessRegressor::GaussianProcessRegressor(double alpha) : alpha_(alpha) {}

void GaussianProcessRegressor::Fit(MatrixXd* x_train, MatrixXd* y_train) {
  x_train_ = x_train;
  y_train_ = y_train;

  auto ln = [](double x) {
    return std::log(x);
  };

  double a2 = alpha_ * alpha_;
  double d3 = 0.5 * x_train_->rows() * std::log(2 * M_PI);
  auto f = [&, a2, d3](const VectorXd& x) {
    int64_t m = x_train_->rows();
    MatrixXd k = Kernel(*x_train_, *x_train_, x[0], x[1]) + (a2 * MatrixXd::Identity(m, m));
    MatrixXd k_inv = k.inverse();

    // Compute determinant via Cholesky decomposition
    MatrixXd l = k.llt().matrixL().toDenseMatrix();
    double d1 = l.diagonal().unaryExpr(ln).sum();
    MatrixXd d2 = 0.5 * (y_train_->transpose() * (k_inv * (*y_train_)));
    MatrixXd cov = d2.array() + (d1 + d3);

    return cov(0, 0);
  };

  double f_min = std::numeric_limits<double>::max();
  VectorXd x_min;
  auto nll_fn = [&](const VectorXd& x, VectorXd& grad) {
    double f0 = f(x);

    if (!isnan(x) && f0 < f_min) {
      f_min = f0;
      x_min = x;
    }

    ApproxFPrime(x, f, f0, grad);
    return f0;
  };

  LBFGSpp::LBFGSParam<double> param;
  param.epsilon = 1e-5;
  param.max_iterations = 100;

  LBFGSpp::LBFGSSolver<double> solver(param);

  VectorXd x = VectorXd::Ones(2);
  double fx;
  solver.minimize(nll_fn, x, fx);

  if (!isnan(x)) {
    length_ = x[0];
    sigma_f_ = x[1];
  } else {
    length_ = x_min[0];
    sigma_f_ = x_min[1];
  }
}

void GaussianProcessRegressor::Predict(const MatrixXd& x, VectorXd& mu, VectorXd* sigma) const {
  MatrixXd cov;
  PosteriorPrediction(x, *x_train_, *y_train_, mu, cov, length_, sigma_f_, alpha_);

  if (sigma != nullptr) {
    auto sqrt = [](double x) {
      return std::sqrt(x);
    };
    *sigma = cov.diagonal().unaryExpr(sqrt);
  }
}

void GaussianProcessRegressor::PosteriorPrediction(
    const MatrixXd& x_s, const MatrixXd& x_train, const MatrixXd& y_train, VectorXd& mu_s, MatrixXd& cov_s,
    double l, double sigma_f, double sigma_y) const {
  int64_t n = x_s.rows();
  int64_t m = x_train.rows();
  double sy2 = sigma_y * sigma_y;

  MatrixXd k = Kernel(x_train, x_train, l, sigma_f) + (sy2 * MatrixXd::Identity(m, m));
  MatrixXd k_s = Kernel(x_train, x_s, l, sigma_f);
  MatrixXd k_ss = Kernel(x_s, x_s, l, sigma_f) + (1e-8 * MatrixXd::Identity(n, n));
  MatrixXd k_inv = k.inverse();

  // Equation (4)
  mu_s = (k_s.transpose() * k_inv) * y_train;

  // Equation (5)
  cov_s = k_ss - (k_s.transpose() * k_inv) * k_s;
}

void GaussianProcessRegressor::ApproxFPrime(const VectorXd& x, const std::function<double(const VectorXd&)>& f,
                                            double f0, VectorXd& grad, double epsilon) {
  VectorXd ei = VectorXd::Zero(x.size());
  for (int k = 0; k < x.size(); k++) {
    ei[k] = 1.0;
    VectorXd d = epsilon * ei;
    grad[k] = (f(x + d) - f0) / d[k];
    ei[k] = 0.0;
  }
}

MatrixXd GaussianProcessRegressor::Kernel(const MatrixXd& x1, const MatrixXd& x2,
                                          double l, double sigma_f) const {
  auto x1_vec = x1.cwiseProduct(x1).rowwise().sum();
  auto x2_vec = x2.cwiseProduct(x2).rowwise().sum();
  auto x1_x2 = x1_vec.replicate(1, x2_vec.size()).rowwise() + x2_vec.transpose();

  auto& dot = x1 * x2.transpose();
  auto sqdist = x1_x2 - (dot.array() * 2).matrix();

  double sigma_f2 = sigma_f * sigma_f;
  double l2 = l * l;
  auto op = [sigma_f2, l2](double x) {
    return sigma_f2 * std::exp(-0.5 / l2 * x);
  };

  return sqdist.unaryExpr(op);
}

} // namespace common
} // namespace horovod
