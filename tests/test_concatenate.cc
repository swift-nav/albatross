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

#include <albatross/Core>
#include <gtest/gtest.h>

namespace albatross {

TEST(test_concatenate, test_concatenation_type) {
  EXPECT_TRUE(bool(
      std::is_same<int, internal::concatenation_type<int, int>::type>::value));
  EXPECT_TRUE(bool(
      std::is_same<double,
                   internal::concatenation_type<double, double>::type>::value));
  EXPECT_FALSE(bool(
      std::is_same<int,
                   internal::concatenation_type<double, double>::type>::value));

  EXPECT_TRUE(bool(
      std::is_same<variant<int, double>,
                   internal::concatenation_type<int, double>::type>::value));
  EXPECT_TRUE(
      bool(std::is_same<variant<int, double>,
                        internal::concatenation_type<variant<int, double>,
                                                     double>::type>::value));
  EXPECT_TRUE(
      bool(std::is_same<variant<int, double>,
                        internal::concatenation_type<
                            double, variant<int, double>>::type>::value));
  EXPECT_TRUE(
      bool(std::is_same<variant<int, double>,
                        internal::concatenation_type<variant<int, double>,
                                                     double>::type>::value));
  EXPECT_TRUE(
      bool(std::is_same<
           variant<int, double>,
           internal::concatenation_type<variant<int, double>,
                                        variant<int, double>>::type>::value));
}

TEST(test_concatenate, test_same_types) {
  std::vector<int> first = {1, 2, 3};
  std::vector<int> second = {4, 5, 6};
  std::vector<int> expected = {1, 2, 3, 4, 5, 6};

  EXPECT_NE(first, expected);
  EXPECT_EQ(concatenate(first, second), expected);
}

TEST(test_concatenate, test_different_types) {
  std::vector<int> first = {1, 2, 3};
  std::vector<double> second = {4., 5., 6.};
  std::vector<variant<int, double>> expected;

  for (const auto &x : first) {
    expected.push_back(x);
  }

  for (const auto &x : second) {
    expected.push_back(x);
  }

  EXPECT_EQ(concatenate(first, second), expected);
}

TEST(test_concatenate, test_different_types_repeated) {
  std::vector<int> first = {1, 2, 3};
  std::vector<double> second = {4., 5., 6.};
  std::vector<variant<int, double>> expected;

  for (const auto &x : first) {
    expected.push_back(x);
  }
  for (const auto &x : second) {
    expected.push_back(x);
  }
  for (const auto &x : second) {
    expected.push_back(x);
  }

  const auto once = concatenate(first, second);
  const auto actual = concatenate(once, second);

  EXPECT_EQ(actual, expected);
}

struct ConcatenateTest {
  ConcatenateTest(const int &x_) : x(x_){};

  bool operator==(const ConcatenateTest &other) const { return x == other.x; }

  int x;
};

TEST(test_concatenate, test_different_types_twice) {
  std::vector<int> first = {1, 2, 3};
  std::vector<double> second = {4., 5., 6.};
  std::vector<ConcatenateTest> third = {ConcatenateTest(10),
                                        ConcatenateTest(11)};
  std::vector<variant<int, double, ConcatenateTest>> expected;

  for (const auto &x : first) {
    expected.push_back(x);
  }
  for (const auto &x : second) {
    expected.push_back(x);
  }
  for (const auto &y : third) {
    expected.push_back(y);
  }

  const auto once = concatenate(first, second);
  EXPECT_EQ(concatenate(once, third), expected);
}

} // namespace albatross
