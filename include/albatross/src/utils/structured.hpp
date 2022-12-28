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

#ifndef INCLUDE_ALBATROSS_UTILS_STRUCTURED_H_
#define INCLUDE_ALBATROSS_UTILS_STRUCTURED_H_

namespace albatross {

/*
 * Holds a structured representation of a matrix which often requires
 * permuting rows and columns;
 */
template <typename MatrixType> struct Structured {

  Eigen::Index rows() const { return matrix.rows(); }

  Eigen::Index cols() const { return matrix.cols(); }

  Eigen::MatrixXd toDense() const { return P_rows * matrix.toDense() * P_cols; }

  MatrixType matrix;
  Eigen::PermutationMatrix<Eigen::Dynamic> P_rows;
  Eigen::PermutationMatrix<Eigen::Dynamic> P_cols;
};

template <typename MatrixType, typename RhsType>
auto operator*(const Structured<MatrixType> &lhs, const RhsType &rhs) {
  return lhs.P_rows * (lhs.matrix * (lhs.P_cols * rhs));
}

namespace structured {

template <typename MatrixType>
auto make_structured(MatrixType &&x,
                     const Eigen::PermutationMatrix<Eigen::Dynamic> &P_rows,
                     const Eigen::PermutationMatrix<Eigen::Dynamic> &P_cols) {
  return Structured<MatrixType>{x, P_rows, P_cols};
}

template <typename MatrixType> auto ldlt(const Structured<MatrixType> &x) {
  return make_structured(x.matrix.ldlt(), x.P_rows, x.P_cols);
}

template <typename MatrixType, typename RhsType>
auto sqrt_solve(const Structured<MatrixType> &x, const RhsType &rhs) {
  assert(x.P_rows == x.P_cols);
  return x.matrix.sqrt_solve(x.P_rows.transpose() * rhs);
}

} // namespace structured

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_UTILS_BLOCK_UTILS_H_ */
