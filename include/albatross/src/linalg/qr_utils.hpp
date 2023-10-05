/*
 * Copyright (C) 2019 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_SRC_LINALG_QR_UTILS_HPP
#define ALBATROSS_SRC_LINALG_QR_UTILS_HPP

namespace albatross {

inline Eigen::MatrixXd
get_R(const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> &qr) {
  // Unfortunately the matrixR() method in Eigen's QR decomposition isn't
  // actually the R matrix, it's tall skinny matrix whose lower trapezoid
  // contains internal data, only the upper triangular portion is useful
  return qr.matrixR()
      .topRows(qr.matrixR().cols())
      .template triangularView<Eigen::Upper>();
}

/*
 * Computes R^-T P^T rhs given R and P from a QR decomposition.
 */
template <typename MatrixType, typename PermutationIndicesType>
inline Eigen::MatrixXd
sqrt_solve(const Eigen::MatrixXd &R,
           const PermutationIndicesType &permutation_indices,
           const MatrixType &rhs) {

  Eigen::MatrixXd sqrt(rhs.rows(), rhs.cols());
  for (Eigen::Index i = 0; i < permutation_indices.size(); ++i) {
    sqrt.row(i) = rhs.row(permutation_indices.coeff(i));
  }
  sqrt = R.template triangularView<Eigen::Upper>().transpose().solve(sqrt);
  return sqrt;
}

template <typename MatrixType>
inline Eigen::MatrixXd
sqrt_solve(const Eigen::ColPivHouseholderQR<Eigen::MatrixXd> &qr,
           const MatrixType &rhs) {
  const Eigen::MatrixXd R = get_R(qr);
  return sqrt_solve(R, qr.colsPermutation().indices(), rhs);
}

} // namespace albatross

#endif // ALBATROSS_SRC_LINALG_QR_UTILS_HPP
