// Copyright (C) 2016 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LBFGS_H
#define LBFGS_H

#include <Eigen/Core>
#include "LBFGS/Param.h"
#include "LBFGS/LineSearch.h"


namespace LBFGSpp {


///
/// LBFGS solver for unconstrained numerical optimization
///
template <typename Scalar>
class LBFGSSolver
{
private:
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
  typedef Eigen::Map<Vector> MapVec;

  const LBFGSParam<Scalar>& m_param;  // Parameters to control the LBFGS algorithm
  Matrix                    m_s;      // History of the s vectors
  Matrix                    m_y;      // History of the y vectors
  Vector                    m_ys;     // History of the s'y values
  Vector                    m_alpha;  // History of the step lengths
  Vector                    m_fx;     // History of the objective function values
  Vector                    m_xp;     // Old x
  Vector                    m_grad;   // New gradient
  Vector                    m_gradp;  // Old gradient
  Vector                    m_drt;    // Moving direction

  inline void reset(int n)
  {
    const int m = m_param.m;
    m_s.resize(n, m);
    m_y.resize(n, m);
    m_ys.resize(m);
    m_alpha.resize(m);
    m_xp.resize(n);
    m_grad.resize(n);
    m_gradp.resize(n);
    m_drt.resize(n);
    if(m_param.past > 0)
      m_fx.resize(m_param.past);
  }

public:
  ///
  /// Constructor for LBFGS solver.
  ///
  /// \param param An object of \ref LBFGSParam to store parameters for the
  ///        algorithm
  ///
  LBFGSSolver(const LBFGSParam<Scalar>& param) :
      m_param(param)
  {
    m_param.check_param();
  }

  ///
  /// Minimizing a multivariate function using LBFGS algorithm.
  /// Exceptions will be thrown if error occurs.
  ///
  /// \param f  A function object such that `f(x, grad)` returns the
  ///           objective function value at `x`, and overwrites `grad` with
  ///           the gradient.
  /// \param x  In: An initial guess of the optimal point. Out: The best point
  ///           found.
  /// \param fx Out: The objective function value at `x`.
  ///
  /// \return Number of iterations used.
  ///
  template <typename Foo>
  inline int minimize(Foo& f, Vector& x, Scalar& fx)
  {
    const int n = x.size();
    const int fpast = m_param.past;
    reset(n);

    // Evaluate function and compute gradient
    fx = f(x, m_grad);
    Scalar xnorm = x.norm();
    Scalar gnorm = m_grad.norm();
    if(fpast > 0)
      m_fx[0] = fx;

    // Early exit if the initial x is already a minimizer
    if(gnorm <= m_param.epsilon * std::max(xnorm, Scalar(1.0)))
    {
      return 1;
    }

    // Initial direction
    m_drt.noalias() = -m_grad;
    // Initial step
    Scalar step = Scalar(1.0) / m_drt.norm();

    int k = 1;
    int end = 0;
    for( ; ; )
    {
      // Save the curent x and gradient
      m_xp.noalias() = x;
      m_gradp.noalias() = m_grad;

      // Line search to update x, fx and gradient
      LineSearch<Scalar>::Backtracking(f, fx, x, m_grad, step, m_drt, m_xp, m_param);

      // New x norm and gradient norm
      xnorm = x.norm();
      gnorm = m_grad.norm();

      // Convergence test -- gradient
      if(gnorm <= m_param.epsilon * std::max(xnorm, Scalar(1.0)))
      {
        return k;
      }
      // Convergence test -- objective function value
      if(fpast > 0)
      {
        if(k >= fpast && std::abs((m_fx[k % fpast] - fx) / fx) < m_param.delta)
          return k;

        m_fx[k % fpast] = fx;
      }
      // Maximum number of iterations
      if(m_param.max_iterations != 0 && k >= m_param.max_iterations)
      {
        return k;
      }

      // Update s and y
      // s_{k+1} = x_{k+1} - x_k
      // y_{k+1} = g_{k+1} - g_k
      MapVec svec(&m_s(0, end), n);
      MapVec yvec(&m_y(0, end), n);
      svec.noalias() = x - m_xp;
      yvec.noalias() = m_grad - m_gradp;

      // ys = y's = 1/rho
      // yy = y'y
      Scalar ys = yvec.dot(svec);
      Scalar yy = yvec.squaredNorm();
      m_ys[end] = ys;

      // Recursive formula to compute d = -H * g
      m_drt.noalias() = -m_grad;
      int bound = std::min(m_param.m, k);
      end = (end + 1) % m_param.m;
      int j = end;
      for(int i = 0; i < bound; i++)
      {
        j = (j + m_param.m - 1) % m_param.m;
        MapVec sj(&m_s(0, j), n);
        MapVec yj(&m_y(0, j), n);
        m_alpha[j] = sj.dot(m_drt) / m_ys[j];
        m_drt.noalias() -= m_alpha[j] * yj;
      }

      m_drt *= (ys / yy);

      for(int i = 0; i < bound; i++)
      {
        MapVec sj(&m_s(0, j), n);
        MapVec yj(&m_y(0, j), n);
        Scalar beta = yj.dot(m_drt) / m_ys[j];
        m_drt.noalias() += (m_alpha[j] - beta) * sj;
        j = (j + 1) % m_param.m;
      }

      // step = 1.0 as initial guess
      step = Scalar(1.0);
      k++;
    }

    return k;
  }
};


} // namespace LBFGSpp

#endif // LBFGS_H