/*
 * Copyright (C) 2026 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

namespace Eigen {

// A batched (not blocked) preconditioned CG implementation.
//
// This is described in
//
//   GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference
//   with GPU Acceleration by Gardner, Pleiss et al.
//
// Eigen's built-in CG routine operates on matrix `b` one column at
// a time.  This gives the right answer but involves a lot of
// matrix-vector calculations.  This routine computes the same
// thing, but it does all the columns at once, so that higher-level
// BLAS calls are used.
//
// The application with iterative solvers and GP models is that the
// additional columns of B are "probe vectors" whose solutions are
// then used to compute approximate posterior covariances and
// likelihoods.

struct BatchedPCGConfig {
  static constexpr const Index cDefaultMaxIterations{100};
  static constexpr const double cDefaultTolerance{1.e-8};
  static constexpr const Index cDefaultResidualReplacementIterations{20};
  Index max_iterations{cDefaultMaxIterations};
  double tolerance{cDefaultTolerance};
  Index residual_replacement_iterations{cDefaultResidualReplacementIterations};
  // If not positive, don't compute any.  If positive, compute for the
  // first `compute_tridiagonals` terms.
  Index compute_tridiagonals{-1};
};

struct BatchedPCGResult {
  MatrixXd X;
  VectorXd residuals;
  Index iterations{0};
  std::vector<SymmetricTridiagonalization> trids;
};

using ArrayXb = Eigen::Array<bool, Eigen::Dynamic, 1>;

// Compute X that solves AX = B from A and B via conjugate gradients.
//
// The given preconditioner is applied via `M.solve(const RhsType &)`.
//
// You may configure the algorithm via `config`.
//
// If you have a relevant starting point (from a previous similar
// problem), please provide this in `X_guess`.  If you do not, just
// pass zeros of the same shape as B.
template <typename MatrixType, typename Rhs, typename Preconditioner>
BatchedPCGResult batched_preconditioned_conjugate_gradient(
    const MatrixType &A, const Preconditioner &M, const Rhs &B,
    const BatchedPCGConfig &config, const Rhs &X_guess) {
  static constexpr const double cSmallValue{1e-16};
  const Index n = B.rows();
  const Index t = B.cols();

  MatrixXd X{X_guess};
  MatrixXd R{B - A * X};
  MatrixXd Z{M.solve(R)};
  MatrixXd P{Z};

  const Index n_tridiagonals =
      std::max(Index{0}, std::min(config.compute_tridiagonals, t));
  MatrixXd alpha_history{MatrixXd::Zero(config.max_iterations, n_tridiagonals)};
  MatrixXd beta_history{MatrixXd::Zero(config.max_iterations, n_tridiagonals)};
  ArrayXi tridiagonal_count{ArrayXi::Zero(n_tridiagonals)};

  RowVectorXd b_norm = B.colwise().norm().array().max(cSmallValue).matrix();

  RowVectorXd r_z_old = (R.array() * Z.array()).colwise().sum();
  ArrayXb active{ArrayXb::Constant(t, true)};
  RowVectorXd residual = R.colwise().norm().array() / b_norm.array();

  BatchedPCGResult res;
  MatrixXd AP(A.rows(), P.cols());
  RowVectorXd denom(P.cols());
  RowVectorXd alpha(t);
  Index iterations{0};
  for (Index i = 0; i < config.max_iterations; ++i) {
    AP.noalias() = A * P;
    denom = (P.array() * AP.array()).colwise().sum();

    // TODO(@peddie): needed?
    alpha.setZero();
    for (Index j = 0; j < t; ++j) {
      if (!active[j]) {
        continue;
      }

      if (std::abs(denom(j)) < cSmallValue ||
          std::abs(r_z_old(j)) < cSmallValue) {
        active[j] = false;
        P.col(j).setZero();
        continue;
      }

      alpha(j) = r_z_old(j) / denom(j);
    }

    X.noalias() += (P.array().rowwise() * alpha.array()).matrix();
    R.noalias() -= (AP.array().rowwise() * alpha.array()).matrix();

    for (int j = 0; j < n_tridiagonals; ++j) {
      if (std::abs(alpha(j)) > 0.0 && active[j]) {
        alpha_history(i, j) = alpha(j);
        tridiagonal_count(j)++;
      }
    }

    if (config.residual_replacement_iterations > 0 &&
        ((i + 1) % config.residual_replacement_iterations == 0)) {
      R = B - A * X;
    }

    residual = R.colwise().norm().array() / b_norm.array();

    for (Index j = 0; j < t; ++j) {
      if (active[j] && residual(j) <= config.tolerance) {
        active[j] = false;
        P.col(j).setZero();
      }
    }

    if (std::find(&active[0], &active[t], true) == &active[t]) {
      iterations = i + 1;
      break;
    }

    Z = M.solve(R);
    RowVectorXd r_z_new{(R.array() * Z.array()).colwise().sum()};
    RowVectorXd beta{RowVectorXd::Zero(t)};

    for (Index j = 0; j < t; ++j) {
      if (!active[j]) {
        continue;
      }

      beta(j) = r_z_new(j) / std::max(r_z_old(j), cSmallValue);
    }

    for (Index j = 0; j < n_tridiagonals; ++j) {
      if (active[j]) {
        beta_history(i, j) = beta(j);
      }
    }

    P = Z + (P.array().rowwise() * beta.array()).matrix();

    for (Index j = 0; j < t; ++j) {
      if (!active[j]) {
        P.col(j).setZero();
      }
    }

    r_z_old = r_z_new;
  }

  std::vector<SymmetricTridiagonalization> trids;
  trids.reserve(albatross::cast::to_size(n_tridiagonals));
  for (Index j = 0; j < n_tridiagonals; ++j) {
    trids.push_back(
        tridiagonal_from_cg(alpha_history.col(j).head(tridiagonal_count(j)),
                            beta_history.col(j).head(tridiagonal_count(j))));
  }
  return {std::move(X), std::move(residual), iterations, std::move(trids)};
}

} // namespace Eigen