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

#include <albatross/CovarianceFunctions>

#include "test_covariance_utils.h"
#include <gtest/gtest.h>

namespace albatross {

std::vector<X> test_xs() {
  std::vector<X> xs;
  xs.emplace_back(X());
  xs.emplace_back(X());
  xs.emplace_back(X());
  return xs;
}

std::vector<Y> test_ys() {
  std::vector<Y> ys;
  ys.emplace_back(Y());
  ys.emplace_back(Y());
  return ys;
}

TEST(test_covariance_function, test_covariance_matrix) {
  HasMultiple cov;

  std::vector<X> xs = test_xs();
  std::vector<Y> ys = test_ys();

  EXPECT_EQ(cov(xs).size(), 9);
  EXPECT_EQ(cov(ys).size(), 4);
  EXPECT_EQ(cov(xs, ys).size(), 6);

  const std::vector<X> const_xs = test_xs();
  const std::vector<Y> const_ys = test_ys();

  EXPECT_EQ(cov(const_xs).size(), 9);
  EXPECT_EQ(cov(const_ys).size(), 4);
  EXPECT_EQ(cov(const_xs, const_ys).size(), 6);

  EXPECT_EQ(cov(test_xs()).size(), 9);
  EXPECT_EQ(cov(test_ys()).size(), 4);
  EXPECT_EQ(cov(test_xs(), test_ys()).size(), 6);
}

TEST(test_covariance_function, test_works_with_two_variants) {
  HasMultiple cov;

  X x;
  Y y;
  W w;

  EXPECT_EQ(cov(x, x), 1.);
  EXPECT_EQ(cov(x, y), 3.);
  EXPECT_EQ(cov(y, x), 3.);
  EXPECT_EQ(cov(y, y), 5.);
  EXPECT_EQ(cov(w, w), 7.);

  variant<X, Y> vxy_x = x;
  variant<X, Y> vxy_y = y;

  EXPECT_EQ(cov(vxy_x, x), cov(x, x));
  EXPECT_EQ(cov(vxy_x, vxy_x), cov(x, x));
  EXPECT_EQ(cov(x, vxy_x), cov(x, x));

  EXPECT_EQ(cov(vxy_x, y), cov(x, y));
  EXPECT_EQ(cov(vxy_x, vxy_y), cov(x, y));
  EXPECT_EQ(cov(x, vxy_y), cov(x, y));

  EXPECT_EQ(cov(vxy_y, x), cov(x, y));
  EXPECT_EQ(cov(vxy_y, vxy_x), cov(x, y));
  EXPECT_EQ(cov(y, vxy_x), cov(x, y));

  EXPECT_EQ(cov(vxy_y, y), cov(y, y));
  EXPECT_EQ(cov(vxy_y, vxy_y), cov(y, y));
  EXPECT_EQ(cov(y, vxy_y), cov(y, y));

  variant<X, W> vxw_w = w;
  variant<X, W> vxw_x = x;

  EXPECT_EQ(cov(vxw_w, w), cov(w, w));
  EXPECT_EQ(cov(vxw_w, vxw_w), cov(w, w));
  EXPECT_EQ(cov(w, vxw_w), cov(w, w));

  EXPECT_EQ(cov(vxw_w, x), 0.);
  EXPECT_EQ(cov(vxw_x, w), 0.);
}

TEST(test_covariance_function, test_works_with_three_variants) {
  HasMultiple cov;

  W w;
  X x;
  Y y;

  variant<X, Y, W> vxyw_x = x;
  variant<X, Y, W> vxyw_y = y;
  variant<X, Y, W> vxyw_w = w;

  EXPECT_EQ(cov(vxyw_x, vxyw_x), cov(x, x));
  EXPECT_EQ(cov(vxyw_x, vxyw_y), cov(x, y));
  EXPECT_EQ(cov(vxyw_y, vxyw_x), cov(x, y));
  EXPECT_EQ(cov(vxyw_y, vxyw_y), cov(y, y));
  EXPECT_EQ(cov(vxyw_w, vxyw_w), cov(w, w));

  EXPECT_EQ(cov(vxyw_x, x), cov(x, x));
  EXPECT_EQ(cov(vxyw_x, y), cov(x, y));
  EXPECT_EQ(cov(vxyw_y, x), cov(x, y));
  EXPECT_EQ(cov(vxyw_y, y), cov(y, y));
  EXPECT_EQ(cov(vxyw_w, w), cov(w, w));

  EXPECT_EQ(cov(x, vxyw_x), cov(x, x));
  EXPECT_EQ(cov(x, vxyw_y), cov(x, y));
  EXPECT_EQ(cov(y, vxyw_x), cov(x, y));
  EXPECT_EQ(cov(y, vxyw_y), cov(y, y));
  EXPECT_EQ(cov(w, vxyw_w), cov(w, w));

  EXPECT_EQ(cov(vxyw_x, vxyw_w), 0.);
  EXPECT_EQ(cov(vxyw_y, vxyw_w), 0.);
  EXPECT_EQ(cov(vxyw_w, vxyw_x), 0.);
  EXPECT_EQ(cov(vxyw_w, vxyw_y), 0.);
}

TEST(test_covariance_function, test_variant_recurssion_bug) {
  // This tests a bug in which the variant forwarder would recurse down the tree
  // of covariance functions until it found one defined for all types involved
  // and in turn ignored some terms.
  HasMultiple has_multiple;
  HasXX has_xx;

  auto cov = has_xx + has_multiple;

  X x;
  Y y;

  EXPECT_EQ(cov(x, x), has_xx(x, x) + has_multiple(x, x));
  EXPECT_EQ(cov(x, y), has_multiple(x, y));
  EXPECT_EQ(cov(y, x), has_multiple(y, x));
  EXPECT_EQ(cov(y, y), has_multiple(y, y));

  variant<X, Y> vx = x;
  variant<X, Y> vy = y;

  EXPECT_EQ(cov(vx, x), cov(x, x));
  EXPECT_EQ(cov(vx, vx), cov(x, x));
  EXPECT_EQ(cov(x, vx), cov(x, x));

  EXPECT_EQ(cov(vx, y), cov(x, y));
  EXPECT_EQ(cov(vx, vy), cov(x, y));
  EXPECT_EQ(cov(x, vy), cov(x, y));

  EXPECT_EQ(cov(vy, x), cov(x, y));
  EXPECT_EQ(cov(vy, vx), cov(x, y));
  EXPECT_EQ(cov(y, vx), cov(x, y));

  EXPECT_EQ(cov(vy, y), cov(y, y));
  EXPECT_EQ(cov(vy, vy), cov(y, y));
  EXPECT_EQ(cov(y, vy), cov(y, y));
}

TEST(test_covariance_function, test_caller_ordering) {
  using ComplicatedFeature =
      variant<X, Y, Measurement<Y>, Measurement<LinearCombination<Y>>>;

  const HasMultiple cov;

  const ComplicatedFeature x = X();

  const Y y;
  const Measurement<Y> meas_y = as_measurement(y);
  const std::vector<Y> two_ys = {Y(), Y()};
  const LinearCombination<Y> sum_of_two_ys(two_ys);
  const ComplicatedFeature meas_lin_combination_y =
      as_measurement(sum_of_two_ys);

  const double x_y = cov(x, y);
  const double x_sum_of_two_ys = 2 * cov(x, y);

  EXPECT_EQ(cov(x, meas_y), x_y);
  EXPECT_EQ(cov(x, sum_of_two_ys), x_sum_of_two_ys);
}

TEST(test_covariance_function, test_linear_combinations) {
  HasMultiple cov_func;
  X x;
  Y y;

  const auto one_x = cov_func(x, x);
  const auto one_xy = cov_func(y, x);

  const auto sum_x = sum(x, x);
  EXPECT_EQ(cov_func(sum_x, x), 2. * one_x);
  EXPECT_EQ(cov_func(x, sum_x), 2. * one_x);
  EXPECT_EQ(cov_func(sum_x, sum_x), 4. * one_x);
  const auto sum_xy = sum(x, y);
  EXPECT_EQ(cov_func(x, sum_xy), one_x + one_xy);

  const auto diff_y = difference(y, y);
  EXPECT_EQ(cov_func(diff_y, diff_y), 0.);
  const auto diff_xy = difference(x, y);
  EXPECT_EQ(cov_func(x, diff_xy), one_x - one_xy);

  std::vector<X> xs = {X(), X()};
  const auto mean_x = mean(xs);
  EXPECT_EQ(cov_func(mean_x, mean_x), 0.25 * cov_func(sum_x, sum_x));
}

} // namespace albatross
