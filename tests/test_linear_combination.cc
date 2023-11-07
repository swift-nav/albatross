/*
 * Copyright (C) 2023 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <albatross/Core>
#include <gtest/gtest.h>

namespace albatross {

TEST(test_linear_combination, test_sum) {
  int x = 1;
  int y = 2;
  const auto combo = linear_combination::sum(x, y);
  EXPECT_EQ(combo.values.size(), 2);
  EXPECT_EQ(combo.values[0], x);
  EXPECT_EQ(combo.values[1], y);
  EXPECT_EQ(combo.coefficients.size(), 2);
  EXPECT_EQ(combo.coefficients[0], 1);
  EXPECT_EQ(combo.coefficients[1], 1);
}

TEST(test_linear_combination, test_sum_variant) {
  int x = 1;
  double y = 3.14159;
  const auto combo = linear_combination::sum(x, y);
  EXPECT_EQ(combo.values.size(), 2);
  EXPECT_EQ(combo.values[0].get<int>(), x);
  EXPECT_EQ(combo.values[1].get<double>(), y);
  EXPECT_EQ(combo.coefficients.size(), 2);
  EXPECT_EQ(combo.coefficients[0], 1);
  EXPECT_EQ(combo.coefficients[1], 1);
}

TEST(test_linear_combination, test_difference) {
  int x = 1;
  int y = 2;
  const auto combo = linear_combination::difference(x, y);
  EXPECT_EQ(combo.values.size(), 2);
  EXPECT_EQ(combo.values[0], x);
  EXPECT_EQ(combo.values[1], y);
  EXPECT_EQ(combo.coefficients.size(), 2);
  EXPECT_EQ(combo.coefficients[0], 1);
  EXPECT_EQ(combo.coefficients[1], -1);
}

TEST(test_linear_combination, test_difference_variant) {
  int x = 1;
  double y = 3.14159;
  const auto combo = linear_combination::difference(x, y);
  EXPECT_EQ(combo.values.size(), 2);
  EXPECT_EQ(combo.values[0].get<int>(), x);
  EXPECT_EQ(combo.values[1].get<double>(), y);
  EXPECT_EQ(combo.coefficients.size(), 2);
  EXPECT_EQ(combo.coefficients[0], 1);
  EXPECT_EQ(combo.coefficients[1], -1);
}

TEST(test_linear_combination, test_mean) {
  for (std::size_t i = 1; i < 12; ++i) {
    std::vector<std::size_t> xs;
    for (std::size_t j = 0; j < i; ++j) {
      xs.emplace_back(j);
    }
    const auto combo = linear_combination::mean(xs);
    EXPECT_EQ(combo.values.size(), i);
    EXPECT_EQ(combo.values, xs);
    EXPECT_EQ(combo.coefficients.size(), i);
    const double expected_coef = 1.0 / static_cast<double>(i);
    for (Eigen::Index j = 0; j < combo.coefficients.size(); ++j) {
      EXPECT_EQ(combo.coefficients[j], expected_coef);
    }
  }
}

TEST(test_linear_combination, test_to_linear_combination) {
  int x = 1;
  const auto combo = linear_combination::to_linear_combination(x);
  EXPECT_EQ(combo.values.size(), 1);
  EXPECT_EQ(combo.values[0], x);
  EXPECT_EQ(combo.coefficients.size(), 1);
  EXPECT_EQ(combo.coefficients[0], 1);
}

TEST(test_linear_combination, test_to_combo) {
  int x = 1;
  int y = 2;
  const auto diff = linear_combination::difference(x, y);
  // diff is already a linear combination
  const auto combo = linear_combination::to_linear_combination(diff);
  EXPECT_EQ(combo.values.size(), 2);
  EXPECT_EQ(combo.values[0], x);
  EXPECT_EQ(combo.values[1], y);
  EXPECT_EQ(combo.coefficients.size(), 2);
  EXPECT_EQ(combo.coefficients[0], 1);
  EXPECT_EQ(combo.coefficients[1], -1);
}

} // namespace albatross
