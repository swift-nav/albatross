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
#include <gtest/gtest.h>

#include <albatross/Core>
#include <albatross/CovarianceFunctions>

#include "../include/albatross/src/covariance_functions/block.hpp"

#include "test_covariance_utils.h"

namespace albatross {

class HasIndependent : public CovarianceFunction<HasIndependent> {
public:
  // Y and Z are independent of each other given X
  // W is independent of everything
  double _call_impl(const X &, const X &) const { return 1.; };

  double _call_impl(const X &, const Y &) const { return 3.; };

  double _call_impl(const Y &, const Y &) const { return 5.; };

  double _call_impl(const Z &, const X &) const { return 7.; };

  double _call_impl(const Z &, const Z &) const { return 9.; };

  double _call_impl(const W &, const W &) const { return 11.; };
};

TEST(test_covariance_functions_block, test_independent) {
  EXPECT_TRUE(bool(detail::is_independent_from<HasIndependent, W, X>::value));
  EXPECT_TRUE(bool(detail::is_independent_from<HasIndependent, Y, W>::value));
  EXPECT_TRUE(bool(detail::is_independent_from<HasIndependent, Z, W>::value));

  EXPECT_FALSE(bool(detail::is_independent_from<HasIndependent, X, Y>::value));
  EXPECT_FALSE(bool(detail::is_independent_from<HasIndependent, Y, X>::value));
  EXPECT_FALSE(bool(detail::is_independent_from<HasIndependent, X, Z>::value));
  EXPECT_FALSE(bool(detail::is_independent_from<HasIndependent, Z, X>::value));
  EXPECT_FALSE(bool(detail::is_independent_from<HasIndependent, X, X>::value));
  EXPECT_FALSE(bool(detail::is_independent_from<HasIndependent, Y, Y>::value));
  EXPECT_FALSE(bool(detail::is_independent_from<HasIndependent, Z, Z>::value));

  EXPECT_TRUE(bool(detail::is_independent_from<HasIndependent, Y, Z>::value));
  EXPECT_TRUE(bool(detail::is_independent_from<HasIndependent, Z, Y>::value));
  EXPECT_TRUE(bool(detail::is_independent_from<HasIndependent, Z, Y>::value));
  EXPECT_TRUE(bool(detail::is_independent_from<HasIndependent, Z, Y>::value));

  // Adding an independent type shouldnt change the result
  EXPECT_FALSE(
      bool(detail::is_independent_from<HasIndependent, X, Y, W>::value));
  EXPECT_FALSE(
      bool(detail::is_independent_from<HasIndependent, Y, X, W>::value));
  EXPECT_FALSE(
      bool(detail::is_independent_from<HasIndependent, X, Z, W>::value));
  EXPECT_FALSE(
      bool(detail::is_independent_from<HasIndependent, Z, X, W>::value));
  EXPECT_FALSE(
      bool(detail::is_independent_from<HasIndependent, X, X, W>::value));
  EXPECT_FALSE(
      bool(detail::is_independent_from<HasIndependent, Y, Y, W>::value));
  EXPECT_FALSE(
      bool(detail::is_independent_from<HasIndependent, Z, Z, W>::value));

  EXPECT_TRUE(
      bool(detail::is_independent_from<HasIndependent, Y, Z, W>::value));
  EXPECT_TRUE(
      bool(detail::is_independent_from<HasIndependent, Z, Y, W>::value));
  EXPECT_TRUE(
      bool(detail::is_independent_from<HasIndependent, Z, Y, W>::value));
  EXPECT_TRUE(
      bool(detail::is_independent_from<HasIndependent, Z, Y, W>::value));
}

TEST(test_covariance_functions_block, test_are_all_independent) {
  EXPECT_TRUE(
      bool(detail::are_all_independent<HasIndependent, variant<Y, Z>>::value));
  EXPECT_TRUE(bool(
      detail::are_all_independent<HasIndependent, variant<Y, Z, W>>::value));
  EXPECT_TRUE(
      bool(detail::are_all_independent<HasIndependent, variant<X, W>>::value));

  EXPECT_FALSE(bool(
      detail::are_all_independent<HasIndependent, variant<X, Y, Z>>::value));
  EXPECT_FALSE(
      bool(detail::are_all_independent<HasIndependent, variant<X, Y>>::value));
  EXPECT_FALSE(
      bool(detail::are_all_independent<HasIndependent, variant<X, Z>>::value));
}

TEST(test_covariance_functions_block,
     test_compute_block_covariance_matrix_one_block) {
  std::vector<variant<Y, Z>> ys = {Y(), Y(), Y()};

  HasIndependent cov;

  const auto before = ys;
  const auto cov_matrix = compute_block_covariance_and_reorder(cov, &ys);

  EXPECT_EQ(cov_matrix.blocks.size(), 1);
  EXPECT_EQ(cov_matrix.rows(), cast::to_index(ys.size()));
  EXPECT_EQ(cov_matrix.cols(), cast::to_index(ys.size()));

  const Eigen::MatrixXd actual = cov(before);
  EXPECT_EQ(actual, cov_matrix.toDense());
}

TEST(test_covariance_functions_block, test_compute_block_covariance_matrix) {
  std::vector<variant<Y, Z>> yzs = {Y(), Z(), Y(), Z(), Z()};

  HasIndependent cov;

  const auto cov_matrix = compute_block_covariance_and_reorder(cov, &yzs);

  EXPECT_EQ(cov_matrix.blocks.size(), 2);
  EXPECT_EQ(cov_matrix.rows(), cast::to_index(yzs.size()));
  EXPECT_EQ(cov_matrix.cols(), cast::to_index(yzs.size()));

  const Eigen::MatrixXd actual = cov(yzs);

  EXPECT_EQ(actual, cov_matrix.toDense());
}

} // namespace albatross
