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
template <typename MatrixType>
stuct StructuredMatrix {

  MatrixType matrix;
  Eigen::PermutationMatrix<Eigen::Dynamic> P_rows;
  Eigen::PermutationMatrix<Eigen::Dynamic> P_cols;
};

namespace structured {

template <typename MatrixType>
auto ldlt(const StructuredMatrix<MatrixType> &x) {
  return StructuredMatrix{x.matrix.ldlt(), x.P_rows, x.P_cols};
}

template <typename MatrixType, typename RhsType>
auto sqrt_solve(const StructuredMatrix<MatrixType> &x,
                const RhsType &rhs) {
  assert(x.P_rows == x.P_cols);
  return x.matrix.sqrt_solve(x.P_rows.transpose() * rhs);
}

}

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_UTILS_BLOCK_UTILS_H_ */
