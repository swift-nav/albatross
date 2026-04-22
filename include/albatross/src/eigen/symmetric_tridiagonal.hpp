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

struct SymmetricTridiagonalization {
  VectorXd diagonal;
  VectorXd subdiagonal;

  [[nodiscard]] inline Index rows() const { return diagonal.size(); }

  [[nodiscard]] inline Index cols() const { return diagonal.size(); }

  SymmetricTridiagonalization() = default;

  explicit SymmetricTridiagonalization(const MatrixXd &T)
      : diagonal{T.diagonal()}, subdiagonal{T.diagonal<-1>()} {
    assert(T.rows() == T.cols() && "Can only construct a symmetric tridiagonal "
                                   "matrix from a square matrix!");
  };
};

// Compute the tridiagonal matrix T of the Lanczos
// tridiagonalization factor corresponding to the CG iterations that
// produced `alpha` and `beta`
[[nodiscard]] inline SymmetricTridiagonalization
tridiagonal_from_cg(const Eigen::VectorXd &alpha, const Eigen::VectorXd &beta) {
  SymmetricTridiagonalization trid;
  trid.diagonal = Eigen::VectorXd::Zero(alpha.size());
  trid.subdiagonal = Eigen::VectorXd::Zero(alpha.size() - 1);

  trid.diagonal(0) = 1 / alpha(0);

  for (Index k = 1; k < alpha.size(); ++k) {
    trid.diagonal(k) = 1 / alpha(k) + beta(k - 1) / alpha(k - 1);
    trid.subdiagonal(k - 1) = std::sqrt(beta(k - 1)) / alpha(k - 1);
  }

  return trid;
}

struct BidiagonalCholeskyFactor {
  VectorXd diagonal;
  VectorXd subdiagonal;

  [[nodiscard]] inline Index rows() const { return diagonal.size(); }

  [[nodiscard]] inline Index cols() const { return diagonal.size(); }
};

// Compute the lower-triangular Cholesky factor of an SPD
// tridiagonal matrix `trid`.
[[nodiscard]] inline BidiagonalCholeskyFactor
tridiagonal_llt(const SymmetricTridiagonalization &trid) {
  BidiagonalCholeskyFactor chol;
  chol.diagonal = VectorXd::Zero(trid.rows());
  chol.subdiagonal = VectorXd::Zero(trid.rows() - 1);
  chol.diagonal(0) = std::sqrt(trid.diagonal(0));

  for (Index k = 1; k < trid.rows(); ++k) {
    chol.subdiagonal(k - 1) = trid.subdiagonal(k - 1) / chol.diagonal(k - 1);
    chol.diagonal(k) = std::sqrt(
        trid.diagonal(k) - chol.subdiagonal(k - 1) * chol.subdiagonal(k - 1));
  }

  return chol;
}

} // namespace Eigen