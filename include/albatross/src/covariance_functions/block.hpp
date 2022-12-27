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
inline BlockDiagonal
compute_block_covariance_and_reorder(CovFuncCaller caller, std::vector<X> *xs,
                                     ThreadPool *pool = nullptr) {
  static_assert(is_invocable<CovFuncCaller, X, X>::value,
                "caller does not support the required arguments");
  static_assert(is_invocable_with_result<CovFuncCaller, double, X, X>::value,
                "caller does not return a double");

  auto grouper = [](const auto &) -> bool { return true; };
  const auto grouped = group_by_with_type(*xs, grouper);
  *xs = reorder(*xs, grouped.indexers());

  auto compute_block = [&caller](const auto &g) { return caller(g); };

  return BlockDiagonal{grouped.apply(compute_block, pool).values()};
}

} // namespace albatross

#endif
