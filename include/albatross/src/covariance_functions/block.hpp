/*
 * Copyright (C) 2022 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_COVARIANCE_FUNCTIONS_BLOCK_HPP_
#define ALBATROSS_COVARIANCE_FUNCTIONS_BLOCK_HPP_

#include <albatross/Core>

namespace albatross {

namespace detail {

template <typename CovFunc, typename X, typename... Ts>
struct is_independent_from {

  template <typename Y> struct is_independent_one_type {
    static constexpr bool value = !has_valid_caller<CovFunc, X, Y>::value;
  };

  static constexpr bool value =
      variant_all<is_independent_one_type, variant<Ts...>>::value;
};

template <typename CovFunc, typename... Ts> struct are_all_independent {

  template <typename Y> struct is_independent_one_type {
    static constexpr bool value = !has_valid_caller<CovFunc, X, Y>::value;
  };

  static constexpr bool value =
      variant_all<is_independent_one_type, variant<Ts...>>::value;
};

} // namespace detail

/*
 * Cross covariance between all elements of a vector.
 */
template <typename CovFuncCaller, typename X, typename Grouper>
inline Eigen::MatrixXd compute_block_covariance_matrix(CovFuncCaller caller,
                                                       const std::vector<X> &xs,
                                                       Grouper grouper) {
  static_assert(is_invocable<CovFuncCaller, X, X>::value,
                "caller does not support the required arguments");
  static_assert(is_invocable_with_result<CovFuncCaller, double, X, X>::value,
                "caller does not return a double");

  const auto grouped = group_by_with_type(xs, grouper);

  Eigen::Index n = cast::to_index(xs.size());
  Eigen::MatrixXd C(n, n);

  Eigen::Index i, j;
  std::size_t si, sj;
  for (i = 0; i < n; i++) {
    si = cast::to_size(i);
    for (j = 0; j <= i; j++) {
      sj = cast::to_size(j);
      C(i, j) = caller(xs[si], xs[sj]);
      C(j, i) = C(i, j);
    }
  }
  return C;
}

} // namespace albatross

#endif
