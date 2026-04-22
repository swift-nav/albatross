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

struct KrylovBasisResult {
  MatrixXd Q;                    // n x k
  SymmetricTridiagonalization T; // k x k
};

// Decompose the operator A into a symmetric product Q T Q^T where Q
// \in R^{n \times k}'s columns are orthonormal and T \in R^{k \times
// k} is symmetric and tridiagonal.  k is either `maxK` or the number
// of iterations until the given `breakdownTol` is reached.
//
// The decomposition is computed iteratively by the Lanczos algorithm
// starting at `start`.  Accuracy is maintained by MGS
// orthogonalisation at each step.
template <typename Operator>
KrylovBasisResult reorthLanczosBasis(const Operator &A,
                                     const Ref<const VectorXd> &start, int maxK,
                                     double breakdownTol = 1e-12) {

  assert(start.norm() != 0.0 && "Lanczos start vector must be nonzero");
  assert(maxK >= 1 && "maxK must be >= 1");

  const int n = static_cast<int>(start.size());
  MatrixXd Q(n, maxK);

  VectorXd q = start / start.norm();
  int k = 0;

  for (; k < maxK; ++k) {
    Q.col(k) = q;

    VectorXd z = A * q;

    // Two-pass modified Gram-Schmidt against all current basis vectors
    MatrixXd Qcur = Q.leftCols(k + 1);

    VectorXd h1 = Qcur.transpose() * z;
    z.noalias() -= Qcur * h1;

    VectorXd h2 = Qcur.transpose() * z;
    z.noalias() -= Qcur * h2;

    const double nz = z.norm();
    if (k + 1 == maxK || nz <= breakdownTol) {
      ++k; // actual number of columns
      break;
    }

    q = z / nz;
  }

  Q.conservativeResize(NoChange, k);

  // Rebuild the small projected matrix robustly.
  MatrixXd AQ = A * Q;
  MatrixXd T = Q.transpose() * AQ;
  T = 0.5 * (T + T.transpose());

  return {Q, SymmetricTridiagonalization(T)};
}

} // namespace Eigen