/*
 * Copyright (C) 2018 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_EIGEN_UTILS_H
#define ALBATROSS_EIGEN_UTILS_H

namespace albatross {

/*
 * Stacks matrices on top of each other, for example:
 *
 * A = [[1 2]      B = [[5 6]
 *       3 4]]          [7 8]]
 *
 * C = vertical_stack({A, B}) = [[1 2]
 *                               [3 4]
 *                               [5 6]
 *                               [7 8]]
 */
template <typename _Scalar, int _Rows, int _Cols>
inline auto vertical_stack(
    const std::vector<Eigen::Matrix<_Scalar, _Rows, _Cols>> &blocks) {
  Eigen::Index rows = 0;
  const Eigen::Index cols = blocks[0].cols();
  for (const auto &block : blocks) {
    rows += block.rows();
    ALBATROSS_ASSERT(block.cols() == cols);
  }

  using MatrixType = Eigen::Matrix<_Scalar, Eigen::Dynamic, _Cols>;
  MatrixType output = MatrixType::Zero(rows, cols);
  Eigen::Index row = 0;
  for (const auto &this_block : blocks) {
    output.block(row, 0, this_block.rows(), cols) = this_block;
    row += this_block.rows();
  }
  return output;
}

/*
 * This helper function performs a matrix solve using the eigen
 * decomposition of a positive semi-definite matrix.
 *
 * Ie it solves for x in:
 *
 *    A x = rhs
 *
 * With A = A^T, eigenvalues(A) >= 0
 *
 * This is done by using the truncated SVD (which for symmetric
 * matrices is equivalent to the truncated eigen decomposition).
 *
 * https://en.wikipedia.org/wiki/Singular_value_decomposition#Truncated_SVD
 */
template <typename _Scalar, int _Rows, int _Cols>
inline auto truncated_psd_solve(
    const Eigen::SelfAdjointEigenSolver<
        Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic>> &lhs_evd,
    const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs, double threshold = 1e-8) {
  const auto V = lhs_evd.eigenvectors();
  auto d = lhs_evd.eigenvalues();

  std::vector<Eigen::Index> inds;
  for (Eigen::Index i = 0; i < d.size(); ++i) {
    if (d[i] >= threshold) {
      inds.push_back(i);
    }
  }

  const auto V_sub = subset_cols(V, inds);
  const auto d_sub = subset(d, inds);

  Eigen::Matrix<_Scalar, _Rows, _Cols> output =
      V_sub * d_sub.asDiagonal().inverse() * V_sub.transpose() * rhs;
  return output;
}

} // namespace albatross

#endif
