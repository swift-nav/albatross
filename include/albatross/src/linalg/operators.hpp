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

#ifndef ALBATROSS_SRC_LINALG_OPERATORS_HPP
#define ALBATROSS_SRC_LINALG_OPERATORS_HPP

namespace albatross {

template <typename MatrixType>
inline Eigen::MatrixXd to_dense(const MatrixType &x) {
  return x.to_dense();
}

template <typename Derived>
inline Eigen::MatrixXd to_dense(const Eigen::DenseBase<Derived> &x) {
  return x.derived();
}

template <int Size, typename X>
inline auto operator*(const Eigen::PermutationMatrix<Size> &matrix,
                      const std::vector<X> &features) {
  std::vector<X> output(features.size());
  const auto &indices = matrix.indices();
  for (Eigen::Index i = 0; i < indices.size(); ++i) {
    const std::size_t si = cast::to_size(i);
    const std::size_t sj = cast::to_size(indices[i]);
    output[sj] = features[si];
  }
  return output;
}

} // namespace albatross

#endif // ALBATROSS_SRC_LINALG_OPERATORS_HPP
