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

#ifndef ALBATROSS_INDEXING_COVARIANCE_H
#define ALBATROSS_INDEXING_COVARIANCE_H

namespace albatross {

namespace details {

template <typename CovFuncCaller, typename X, typename Y>
inline void fill_covariance_subset(CovFuncCaller caller,
                                   const std::vector<X> &xs,
                                   const std::vector<std::size_t> &x_inds,
                                   const std::vector<Y> &ys,
                                   const std::vector<std::size_t> &y_inds,
                                   Eigen::MatrixXd *C) {
  static_assert(is_invocable<CovFuncCaller, X, Y>::value,
                "caller does not support the required arguments");
  static_assert(is_invocable_with_result<CovFuncCaller, double, X, Y>::value,
                "caller does not return a double");
  Eigen::Index m = static_cast<Eigen::Index>(xs.size());
  Eigen::Index n = static_cast<Eigen::Index>(ys.size());

  Eigen::Index i, j, ci, cj;
  std::size_t si, sj;
  for (i = 0; i < m; i++) {
    si = static_cast<std::size_t>(i);
    ci = static_cast<Eigen::Index>(x_inds[si]);
    for (j = 0; j < n; j++) {
      sj = static_cast<std::size_t>(j);
      cj = static_cast<Eigen::Index>(y_inds[sj]);
      (*C)(ci, cj) = caller(xs[si], ys[sj]);
    }
  }
}

template <typename CovFuncCaller, typename X>
inline void fill_covariance_subset(CovFuncCaller caller,
                                   const std::vector<X> &xs,
                                   const std::vector<std::size_t> &x_inds,
                                   Eigen::MatrixXd *C) {
  static_assert(is_invocable<CovFuncCaller, X, X>::value,
                "caller does not support the required arguments");
  static_assert(is_invocable_with_result<CovFuncCaller, double, X, X>::value,
                "caller does not return a double");
  Eigen::Index n = static_cast<Eigen::Index>(xs.size());

  Eigen::Index i, j, ci, cj;
  std::size_t si, sj;
  for (i = 0; i < n; i++) {
    si = static_cast<std::size_t>(i);
    ci = static_cast<Eigen::Index>(x_inds[si]);
    for (j = 0; j <= i; j++) {
      sj = static_cast<std::size_t>(j);
      cj = static_cast<Eigen::Index>(x_inds[sj]);
      (*C)(ci, cj) = caller(xs[si], xs[sj]);
      (*C)(cj, ci) = (*C)(ci, cj);
    }
  }
}

} // namespace details

template <
    typename CovFuncCaller, typename Grouper, typename X, typename Y,
    typename std::enable_if_t<details::is_valid_grouper<Grouper, X>::value &&
                                  details::is_valid_grouper<Grouper, Y>::value,
                              int> = 0>
inline Eigen::MatrixXd compute_covariance_by_group(CovFuncCaller caller,
                                                   Grouper grouper,
                                                   const std::vector<X> &xs,
                                                   const std::vector<Y> &ys) {
  Eigen::MatrixXd C(xs.size(), ys.size());
  const auto y_indexers = group_by(ys, grouper).indexers();

  auto fill_block = [&caller, &xs, &ys, &C](const auto &x_inds,
                                            const auto &y_inds) {
    details::fill_covariance_subset(caller, subset(xs, x_inds), x_inds,
                                    subset(ys, y_inds), y_inds, &C);
  };

  auto fill_row = [&](const auto &x_key, const auto &x_inds) {
    for (const auto &y_pair : y_indexers) {
      fill_block(x_inds, y_pair.second);
    }
  };

  const auto x_indexers = group_by(xs, grouper).indexers();
  async_apply(x_indexers, fill_row);
  return C;
}

template <typename CovFuncCaller, typename Grouper, typename X,
          typename std::enable_if_t<
              details::is_valid_grouper<Grouper, X>::value, int> = 0>
inline Eigen::MatrixXd compute_covariance_by_group(CovFuncCaller caller,
                                                   Grouper grouper,
                                                   const std::vector<X> &xs) {
  Eigen::MatrixXd C(xs.size(), xs.size());
  const auto x_indexers = group_by(xs, grouper).indexers();

  auto fill_row = [&](const auto &x_key, const auto &x_inds) {
    for (const auto &y_pair : x_indexers) {
      const auto &y_inds = y_pair.second;
      if (x_key < y_pair.first) {
        // Lower triangle case, fill then copy
        details::fill_covariance_subset(caller, subset(xs, x_inds), x_inds,
                                        subset(xs, y_inds), y_inds, &C);
        for (const auto &x_ind : x_inds) {
          for (const auto &y_ind : y_inds) {
            C(y_ind, x_ind) = C(x_ind, y_ind);
          }
        }
      } else if (x_key == y_pair.first) {
        // Diagonal case, fill both triangles directly
        details::fill_covariance_subset(caller, subset(xs, x_inds), x_inds, &C);
      }
    }
  };

  async_apply(x_indexers, fill_row);

  return C;
}

} // namespace albatross

#endif
