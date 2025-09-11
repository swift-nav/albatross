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

#ifndef ALBATROSS_SRC_LINALG_BLOCK_UTILS_HPP
#define ALBATROSS_SRC_LINALG_BLOCK_UTILS_HPP

namespace albatross {

/*
 * One approach to dealing with block linear algebra is to cluster everything
 * into groups these subsequent methods make those representations easier to
 * work with.
 */
template <typename MatrixType>
inline Eigen::MatrixXd block_sum(const std::vector<MatrixType> &xs) {
  MatrixType output = xs[0];
  for (std::size_t i = 1; i < xs.size(); ++i) {
    // Eigen internally asserts that the results are the same size.
    output += xs[i];
  }
  return output;
}

template <typename GroupKey, typename MatrixType>
inline MatrixType block_sum(const Grouped<GroupKey, MatrixType> &xs) {
  return block_sum(xs.values());
}

/*
 * Patchwork GP works by clustering all the data into groups which
 * results in several Grouped objects containing block matrix representations.
 *
 * These subsequent methods make those representations easier to work with.
 */
template <typename GroupKey, typename X, typename Y, typename ApplyFunction>
inline Eigen::MatrixXd block_accumulate(const Grouped<GroupKey, X> &lhs,
                                        const Grouped<GroupKey, Y> &rhs,
                                        const ApplyFunction &apply_function) {
  // block_accumulate takes two different grouped objects and returns
  // the sum of that function applied to each pair of values.  Another
  // way of writing this could be something like:
  //
  //   sum_i ( apply_function(lhs.at(key_i), rhs.at(key_i)) )
  //
  // The result of apply_function is expected to be an Eigen::MatrixXd
  static_assert(
      std::is_same<Eigen::MatrixXd,
                   typename invoke_result<ApplyFunction, X, Y>::type>::value,
      "apply_function needs to return an Eigen::MatrixXd type");

  ALBATROSS_ASSERT(lhs.size() == rhs.size());
  ALBATROSS_ASSERT(lhs.size() > 0);

  auto one_group = [&](const GroupKey &key) {
    ALBATROSS_ASSERT(map_contains(lhs, key) && map_contains(rhs, key));
    return apply_function(lhs.at(key), rhs.at(key));
  };

  return block_sum(apply(lhs.keys(), one_group));
}

template <typename GroupKey, typename ApplyFunction>
inline Eigen::MatrixXd block_product(
    const Grouped<GroupKey, Eigen::MatrixXd> &lhs,
    const Grouped<GroupKey, Eigen::MatrixXd> &rhs) {
  // This performs a block matrix product operation where if you aligned the
  // lhs into horizontal blocks and the right into vertical blocks by ordering
  // their keys:
  //
  //   lhs = [x_0, ..., x_n]
  //   rhs = [y_0, ..., y_n]
  //
  // the output corresponds to:
  //
  //   lhs * rhs = [x_0, ..., x_2] * [y_0
  //                                  ...
  //                                  y_2]
  //
  auto matrix_product = [&](const auto &x, const auto &y) {
    return (x * y).eval();
  };

  return block_accumulate(lhs, rhs, matrix_product);
}

template <typename GroupKey>
inline Eigen::MatrixXd block_inner_product(
    const Grouped<GroupKey, Eigen::MatrixXd> &lhs,
    const Grouped<GroupKey, Eigen::MatrixXd> &rhs) {
  // This performs a block matrix inner product operation where if you aligned
  // the lhs into horizontal blocks and the right into vertical blocks by
  // ordering their keys:
  //
  //   lhs = [x_0, ..., x_n]
  //   rhs = [y_0, ..., y_n]
  //
  // the output corresponds to:
  //
  //   lhs.T * rhs = [x_0.T, ..., x_n.T] * [y_0
  //                                        ...
  //                                        y_n]
  //
  auto matrix_inner_product = [&](const auto &x, const auto &y) {
    return (x.transpose() * y).eval();
  };

  return block_accumulate(lhs, rhs, matrix_inner_product);
}

template <typename GroupKey, typename Solver, typename Rhs>
inline auto block_diag_solve(const Grouped<GroupKey, Solver> &lhs,
                             const Grouped<GroupKey, Rhs> &rhs) {
  // Here we have the equivalent to a block diagonal matrix solve
  // in which the inverse of each group in the lhs is applied to
  // the corresponding group in rhs.
  //
  //   lhs = [x_0, ..., x_n]
  //   rhs = [y_0, ..., y_n]
  //
  // the output corresponds to:
  //
  //   lhs.T * rhs = [x_0^-1, ..., x_n^-1] * [y_0
  //                                          ...
  //                                          y_n]
  //
  auto solve_one_block = [&](const auto &key, const auto &x) {
    return Eigen::MatrixXd(lhs.at(key).solve(x));
  };

  return rhs.apply(solve_one_block);
};

template <typename GroupKey>
inline Grouped<GroupKey, Eigen::MatrixXd> block_subtract(
    const Grouped<GroupKey, Eigen::MatrixXd> &lhs,
    const Grouped<GroupKey, Eigen::MatrixXd> &rhs) {
  ALBATROSS_ASSERT(lhs.size() == rhs.size());
  auto matrix_subtract = [&](const auto &key_i, const auto &rhs_i) {
    return (lhs.at(key_i) - rhs_i).eval();
  };

  return rhs.apply(matrix_subtract);
}

}  // namespace albatross

#endif  // ALBATROSS_SRC_LINALG_BLOCK_UTILS_HPP
