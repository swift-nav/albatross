/*
 * Copyright (C) 2023 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_SRC_LINALG_BLOCK_DIA_HPP
#define ALBATROSS_SRC_LINALG_BLOCK_DIA_HPP

namespace albatross {

using PermutationIndices = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;

struct BlockSymmetricArrowLDLT {
  BlockDiagonalLDLT upper_left;
  Eigen::MatrixXd lower_left;
  Eigen::SerializableLDLT lower_right;

  template <class _Scalar, int _Rows, int _Cols>
  Eigen::Matrix<_Scalar, _Rows, _Cols>
  sqrt_solve(const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const;

  Eigen::Index rows() const;

  Eigen::Index cols() const;
};

inline Eigen::Index BlockSymmetricArrowLDLT::cols() const {
  return upper_left.cols() + lower_right.cols();
}

inline Eigen::Index BlockSymmetricArrowLDLT::rows() const {
  return upper_left.rows() + lower_right.rows();
}

template <class _Scalar, int _Rows, int _Cols>
inline Eigen::Matrix<_Scalar, _Rows, _Cols> BlockSymmetricArrowLDLT::sqrt_solve(
    const Eigen::Matrix<_Scalar, _Rows, _Cols> &rhs) const {
  ALBATROSS_ASSERT(cols() == rhs.rows());
  Eigen::Index i = 0;
  Eigen::Matrix<_Scalar, _Rows, _Cols> output(rows(), rhs.cols());

  output.topRows(upper_left.rows()) =
      upper_left.sqrt_solve(rhs.topRows(upper_left.rows()).eval());
  output.bottomRows(lower_right.rows()) =
      lower_right.sqrt_solve(rhs.bottomRows(lower_right.rows()) -
                             lower_left * output.topRows(upper_left.rows()));
  return output;
}

// Constructs a BlockSymmetricArrowLDLT
inline BlockSymmetricArrowLDLT
block_symmetric_arrow_ldlt(const BlockDiagonal &upper_left,
                           const Eigen::MatrixXd &upper_right,
                           const Eigen::MatrixXd &lower_right) {
  assert(upper_right.rows() == upper_right.rows());
  BlockSymmetricArrowLDLT output;
  output.upper_left = upper_left.ldlt();
  output.lower_left = output.upper_left.sqrt_solve(upper_right).transpose();
  output.lower_right = Eigen::SerializableLDLT(
      lower_right - output.lower_left * output.lower_left.transpose());
  return output;
}

} // namespace albatross

#endif // ALBATROSS_SRC_LINALG_BLOCK_DIAGONAL_HPP