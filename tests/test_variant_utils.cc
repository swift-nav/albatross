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

#include <albatross/Common>
#include <albatross/utils/VariantUtils>
#include <gtest/gtest.h>

namespace albatross {

TEST(test_variant_utils, test_set_variant_two_types) {

  variant<int, double> foo;
  const int one = 1;
  const double two = 2.;

  variant<double, int> bar;

  set_variant(one, &foo);
  EXPECT_TRUE(foo.is<int>());
  EXPECT_EQ(foo.get<int>(), one);

  set_variant(two, &foo);
  EXPECT_TRUE(foo.is<double>());
  EXPECT_EQ(foo.get<double>(), two);

  bar = one;
  set_variant(bar, &foo);
  EXPECT_TRUE(foo.is<int>());
  EXPECT_EQ(foo.get<int>(), one);

  bar = two;
  set_variant(bar, &foo);
  EXPECT_TRUE(foo.is<double>());
  EXPECT_EQ(foo.get<double>(), two);
}

struct X {
  double value;
  bool operator==(const X &other) const { return value == other.value; }
};

TEST(test_variant_utils, test_set_variant_three_types) {

  variant<int, double, X> foo;
  const int one = 1;
  const double two = 2.;
  const X x = {3.};

  variant<double, int> double_int;

  double_int = one;
  set_variant(double_int, &foo);
  EXPECT_TRUE(foo.is<int>());
  EXPECT_EQ(foo.get<int>(), one);

  double_int = two;
  set_variant(double_int, &foo);
  EXPECT_TRUE(foo.is<double>());
  EXPECT_EQ(foo.get<double>(), two);

  variant<double, X> double_x;

  double_x = x;
  set_variant(double_x, &foo);
  EXPECT_TRUE(foo.is<X>());
  EXPECT_EQ(foo.get<X>(), x);

  double_x = two;
  set_variant(double_x, &foo);
  EXPECT_TRUE(foo.is<double>());
  EXPECT_EQ(foo.get<double>(), two);
}

TEST(test_variant_utils, test_to_variant_vector) {
  const auto doubles = linspace(0., 10., 11);
  const auto variants = to_variant_vector<variant<int, double, X>>(doubles);
  EXPECT_EQ(variants.size(), doubles.size());

  for (std::size_t i = 0; i < variants.size(); ++i) {
    EXPECT_TRUE(variants[i].is<double>());
    EXPECT_EQ(variants[i].get<double>(), doubles[i]);
  }

  double a = 1.;
  double b = 2.;
  X x_a = {a};
  X x_b = {b};

  std::vector<variant<double, X>> mixed;
  mixed.emplace_back(a);
  mixed.emplace_back(b);
  mixed.emplace_back(x_a);
  mixed.emplace_back(x_b);

  auto from_mixed = to_variant_vector<variant<int, double, X>>(mixed);
  for (std::size_t i = 0; i < mixed.size(); ++i) {
    mixed[i].match([&](auto v) {
      EXPECT_TRUE(from_mixed[i].is<decltype(v)>());
      EXPECT_EQ(from_mixed[i].get<decltype(v)>(), v);
    });
  }
}

} // namespace albatross
