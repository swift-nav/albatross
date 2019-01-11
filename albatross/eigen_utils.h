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

#include <functional>
#include <map>
#include <unordered_map>
#include <vector>

namespace albatross {

/*
 * Builds a new DiagonalMatrix from a vector of other DiagonalMatrices.
 * Equivalent to concatenating the diagonal vectors.
 */
template <typename _Scalar, int SizeAtCompileTime>
inline Eigen::DiagonalMatrix<_Scalar, Eigen::Dynamic> block_diagonal(
    const std::vector<Eigen::DiagonalMatrix<_Scalar, SizeAtCompileTime>>
        &blocks) {
  Eigen::Index n = 0;
  for (const auto &block : blocks) {
    n += block.rows();
  }

  Eigen::Matrix<_Scalar, Eigen::Dynamic, 1> diag(n);
  Eigen::Index i = 0;
  for (const auto &block : blocks) {
    const auto d = block.diagonal();
    diag.block(i, 0, d.size(), 1) = d;
    i += d.size();
  }
  return diag.asDiagonal();
}

/*
 * Builds a dense block diagonal matrix for a vector of matrices.  These
 * blocks need not be square.
 *
 * A = [[1 2]      B = [[5 6 7 ]
 *       3 4]]          [8 9 10]]
 *
 * C = block_diagonal({A, B}) = [[1 2 0 0 0 ]
 *                               [3 4 0 0 0 ]
 *                               [0 0 5 6 7 ]
 *                               [0 0 8 9 10]]
 */
template <typename _Scalar, int _Rows, int _Cols>
inline auto block_diagonal(
    const std::vector<Eigen::Matrix<_Scalar, _Rows, _Cols>> &blocks) {
  Eigen::Index rows = 0;
  Eigen::Index cols = 0;
  for (const auto &block : blocks) {
    rows += block.rows();
    cols += block.cols();
  }

  using MatrixType = Eigen::Matrix<_Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  MatrixType output = MatrixType::Zero(rows, cols);
  Eigen::Index row = 0;
  Eigen::Index col = 0;
  for (const auto &block : blocks) {
    output.block(row, col, block.rows(), block.cols()) = block;
    row += block.rows();
    col += block.cols();
  }
  return output;
}

/*
 * Builds a dense block diagonal matrix for a vector of matrices.  These
 * blocks need not be square.
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
    assert(block.cols() == cols);
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

} // namespace albatross

#endif
