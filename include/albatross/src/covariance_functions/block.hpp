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

#include "../indexing/reorder.hpp"
#include "../utils/block_sparse_matrix.hpp"
#include "../utils/block_utils.hpp"

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

template <typename CovFunc, typename X>
struct are_all_independent : public std::true_type {};

// Checks whether all types provided as arguments are
// independent of each other according to a covariance
// function
template <typename CovFunc, typename X, typename... Ts>
struct are_all_independent<CovFunc, variant<X, Ts...>> {
  static constexpr bool value =
      is_independent_from<CovFunc, X, Ts...>::value &&
      are_all_independent<CovFunc, variant<Ts...>>::value;
};

} // namespace detail

/*
 * Covariance between all elements of a vector.
 */
template <typename CovFuncCaller, typename X>
inline Structured<BlockDiagonal>
compute_block_covariance(CovFuncCaller caller, const std::vector<X> &xs,
                         ThreadPool *pool = nullptr) {
  static_assert(is_invocable<CovFuncCaller, X, X>::value,
                "caller does not support the required arguments");
  static_assert(is_invocable_with_result<CovFuncCaller, double, X, X>::value,
                "caller does not return a double");

  auto grouper = [](const auto &) -> bool { return true; };
  const auto grouped = group_by_with_type(xs, grouper);

  const auto P = build_permutation_matrix(grouped.indexers());

  auto compute_block = [&caller](const auto &g) { return caller(g); };

  return Structured<BlockDiagonal>{
      BlockDiagonal{grouped.apply(compute_block, pool).values()}, P, P};
}

namespace detail {

template <typename CovFunc, typename X, typename Y>
inline bool is_zero(const X &, const Y &) {
  return detail::is_independent_from<CovFunc, X, Y>::value;
}

template <typename CovFunc, typename... Xs, typename Y>
inline bool is_zero(const variant<Xs...> &xs, const Y &y) {
  return xs.match([&y](const auto &x) { return is_zero<CovFunc>(x, y); });
}

template <typename CovFunc, typename... Xs, typename... Ys>
inline bool is_zero(const variant<Xs...> &xs, const variant<Ys...> &ys) {
  return xs.match([&ys](const auto &x) {
    return ys.match([&x](const auto &y) { return is_zero<CovFunc>(x, y); });
  });
}

} // namespace detail

template <typename CovFunc, typename X, typename Y>
inline BlockSparseMatrix<Eigen::MatrixXd>
compute_block_cross_covariance_and_reorder(CovFunc caller, std::vector<X> *xs,
                                           std::vector<Y> *ys,
                                           ThreadPool *pool = nullptr) {

  const auto x_grouped =
      group_by_with_type(*xs, [](const auto &) -> bool { return true; });
  *xs = reorder(*xs, x_grouped.indexers());

  const auto y_grouped =
      group_by_with_type(*ys, [](const auto &) -> bool { return true; });
  *ys = reorder(*ys, y_grouped.indexers());

  const auto row_sizes = x_grouped.counts().apply(cast::to_index).values();
  const auto col_sizes = y_grouped.counts().apply(cast::to_index).values();

  BlockSparseMatrix<Eigen::MatrixXd> output(row_sizes, col_sizes);
  Eigen::Index i = 0;
  for (const auto &x_pair : x_grouped.groups()) {
    Eigen::Index j = 0;
    for (const auto &y_pair : y_grouped.groups()) {
      if (!detail::is_zero<CovFunc>(x_pair.second[0], y_pair.second[0])) {
        output.set_block(i, j, caller(x_pair.second, y_pair.second, pool));
      }
      ++j;
    }
    ++i;
  }

  return output;
}

} // namespace albatross

#endif
