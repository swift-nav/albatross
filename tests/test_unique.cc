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

#include <albatross/Indexing>
#include <gtest/gtest.h>

namespace albatross {

// Empty input

TEST(test_unique, unique_values_empty) {
  std::vector<int> empty = {};
  EXPECT_EQ(unique_values(empty).size(), 0);
}

TEST(test_unique, unique_value_empty) {
  std::vector<int> empty = {};
  EXPECT_DEATH({ unique_value(empty); }, "Assertion");
}

TEST(test_unique, unique_values_function_empty) {
  std::vector<int> empty = {};
  auto foo = [](const int &x) { return x + 1; };
  EXPECT_EQ(unique_values(empty, foo).size(), 0);
}

// Identical inputs

TEST(test_unique, unique_value_identical) {
  std::vector<int> values = {3, 3, 3};
  EXPECT_EQ(unique_value(values), 3);
}

// Mixed input

TEST(test_unique, unique_values_mixed) {
  std::vector<int> values = {3, 1, 5, 1, 3};
  std::set<int> expected = {1, 3, 5};
  EXPECT_EQ(unique_values(values), expected);
}

TEST(test_unique, unique_value_function_mixed) {
  std::vector<int> values = {3, 1, 5, 2};
  auto foo = [](const auto &) -> double { return 4.; };
  EXPECT_EQ(unique_value(values, foo), 4.);
}

TEST(test_unique, unique_value_mixed) {
  std::vector<int> values = {3, 1, 5, 1, 3};
  EXPECT_DEATH({ unique_value(values); }, "Assertion");
}

TEST(test_unique, unique_values_function_mixed) {
  std::vector<int> values = {3, 1, 5, 1, 3};
  auto foo = [](const int &x) { return x + 1; };

  const auto applied = albatross::apply(values, foo);
  std::set<int> expected(applied.begin(), applied.end());

  EXPECT_EQ(unique_values(values, foo), expected);
}

// With sets

TEST(test_unique, unique_values_set_function_empty) {
  std::set<int> values = {};
  auto foo = [](const auto &) -> double { return 4.; };
  std::set<double> expected = {};
  EXPECT_EQ(unique_values(values, foo), expected);
}

TEST(test_unique, unique_values_set_function) {
  std::set<int> values = {3, 1, 5, 2};
  auto foo = [](const auto &) -> double { return 4.; };
  std::set<double> expected = {4.};
  EXPECT_EQ(unique_values(values, foo), expected);
}

TEST(test_unique, unique_values_set_function_non_trival) {
  std::set<int> values = {0, 1, 2, 3, 4, 5, 6, 7};
  auto foo = [](const auto &x) -> double { return x % 2; };
  std::set<double> expected = {0, 1};
  EXPECT_EQ(unique_values(values, foo), expected);
}

} // namespace albatross
