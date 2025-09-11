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

#include <gtest/gtest.h>
#include <albatross/Core>

namespace albatross {

TEST(test_traits_core, test_is_vector) {
  EXPECT_TRUE(bool(is_vector<std::vector<double>>::value));
  EXPECT_FALSE(bool(is_vector<double>::value));
}

TEST(test_traits_core, test_variant_size) {
  const auto one = variant_size<variant<int>>::value;
  EXPECT_EQ(one, 1);

  const auto two = variant_size<variant<int, double>>::value;
  EXPECT_EQ(two, 2);

  struct X {};
  const auto three = variant_size<variant<int, double, X>>::value;
  EXPECT_EQ(three, 3);
}

TEST(test_traits_core, test_is_variant) {
  EXPECT_TRUE(is_variant<variant<int>>::value);
  EXPECT_TRUE(bool(is_variant<variant<int, double>>::value));
  EXPECT_FALSE(bool(is_variant<Measurement<int>>::value));
  EXPECT_FALSE(is_variant<int>::value);
  EXPECT_FALSE(is_variant<double>::value);
}

TEST(test_traits_core, test_is_in_variant) {
  struct W {};
  struct X {};
  struct Y {};
  struct Z {};

  EXPECT_TRUE(bool(is_in_variant<X, variant<X>>::value));
  EXPECT_FALSE(bool(is_in_variant<Y, variant<X>>::value));

  EXPECT_TRUE(bool(is_in_variant<X, variant<X, Y>>::value));
  EXPECT_TRUE(bool(is_in_variant<Y, variant<X, Y>>::value));
  EXPECT_FALSE(bool(is_in_variant<Z, variant<X, Y>>::value));

  EXPECT_TRUE(bool(is_in_variant<X, variant<X, Y, Z>>::value));
  EXPECT_TRUE(bool(is_in_variant<Y, variant<X, Y, Z>>::value));
  EXPECT_TRUE(bool(is_in_variant<Z, variant<X, Y, Z>>::value));
  EXPECT_FALSE(bool(is_in_variant<W, variant<X, Y, Z>>::value));

  EXPECT_FALSE(bool(is_in_variant<variant<X>, variant<X, Y, Z>>::value));
}

struct Foo {};

template <typename T>
struct is_foo : public std::is_same<T, Foo> {};

TEST(test_traits_core, test_variant_any) {
  struct X {};
  struct Y {};
  struct Z {};

  EXPECT_FALSE(bool(variant_any<is_foo, variant<>>::value));

  EXPECT_TRUE(bool(variant_any<is_foo, variant<Foo>>::value));
  EXPECT_FALSE(bool(variant_any<is_foo, variant<X>>::value));

  EXPECT_TRUE(bool(variant_any<is_foo, variant<Foo, X>>::value));
  EXPECT_TRUE(bool(variant_any<is_foo, variant<X, Foo>>::value));
  EXPECT_TRUE(bool(variant_any<is_foo, variant<Y, Foo>>::value));
  EXPECT_FALSE(bool(variant_any<is_foo, variant<X, Y>>::value));

  EXPECT_TRUE(bool(variant_any<is_foo, variant<Foo, X, Y>>::value));
  EXPECT_TRUE(bool(variant_any<is_foo, variant<X, Y, Foo>>::value));
  EXPECT_TRUE(bool(variant_any<is_foo, variant<Y, Foo, X>>::value));
  EXPECT_FALSE(bool(variant_any<is_foo, variant<X, Y, Z>>::value));
}

struct Bar {};

template <typename T>
struct is_foo_or_bar {
  static constexpr bool value =
      std::is_same<T, Foo>::value || std::is_same<T, Bar>::value;
};

TEST(test_traits_core, test_variant_all) {
  struct X {};
  struct Y {};
  struct Z {};

  EXPECT_TRUE(bool(variant_all<is_foo_or_bar, variant<>>::value));

  EXPECT_TRUE(bool(variant_all<is_foo_or_bar, variant<Foo>>::value));
  EXPECT_TRUE(bool(variant_all<is_foo_or_bar, variant<Bar>>::value));
  EXPECT_FALSE(bool(variant_all<is_foo_or_bar, variant<Y>>::value));

  EXPECT_TRUE(bool(variant_all<is_foo_or_bar, variant<Foo, Bar>>::value));
  EXPECT_TRUE(bool(variant_all<is_foo_or_bar, variant<Bar, Foo>>::value));
  EXPECT_FALSE(bool(variant_all<is_foo_or_bar, variant<Y, Foo>>::value));
  EXPECT_FALSE(bool(variant_all<is_foo_or_bar, variant<X, Y>>::value));

  EXPECT_FALSE(bool(variant_all<is_foo_or_bar, variant<Foo, Bar, Y>>::value));
  EXPECT_FALSE(bool(variant_all<is_foo_or_bar, variant<X, Bar, Foo>>::value));
  EXPECT_FALSE(bool(variant_all<is_foo_or_bar, variant<Y, Foo, X>>::value));
  EXPECT_FALSE(bool(variant_all<is_foo_or_bar, variant<X, Y, Z>>::value));
}

TEST(test_traits_core, test_is_sub_variant) {
  struct X {};
  struct Y {};
  struct Z {};

  EXPECT_FALSE(bool(is_sub_variant<X, variant<X>>::value));
  EXPECT_FALSE(bool(is_sub_variant<variant<X>, X>::value));
  EXPECT_FALSE(bool(is_sub_variant<X, variant<>>::value));

  EXPECT_TRUE(bool(is_sub_variant<variant<>, variant<>>::value));

  EXPECT_TRUE(bool(is_sub_variant<variant<>, variant<X>>::value));
  EXPECT_TRUE(bool(is_sub_variant<variant<X>, variant<X>>::value));
  EXPECT_FALSE(bool(is_sub_variant<variant<Y>, variant<X>>::value));

  EXPECT_TRUE(bool(is_sub_variant<variant<X>, variant<X, Y>>::value));
  EXPECT_TRUE(bool(is_sub_variant<variant<X>, variant<Y, X>>::value));
  EXPECT_TRUE(bool(is_sub_variant<variant<X, Y>, variant<X, Y>>::value));
  EXPECT_TRUE(bool(is_sub_variant<variant<Y, X>, variant<Y, X>>::value));
  EXPECT_FALSE(bool(is_sub_variant<variant<Y, X>, variant<Y, Z>>::value));
  EXPECT_FALSE(bool(is_sub_variant<variant<X, Y>, variant<Y, Z>>::value));
  EXPECT_FALSE(bool(is_sub_variant<variant<X>, variant<Y, Z>>::value));
}

TEST(test_traits_core, test_is_measurement) {
  EXPECT_TRUE(is_measurement<Measurement<int>>::value);
  EXPECT_TRUE(bool(is_measurement<Measurement<double>>::value));
  EXPECT_TRUE(bool(is_measurement<Measurement<variant<int, double>>>::value));
  EXPECT_FALSE(is_measurement<int>::value);
  EXPECT_FALSE(is_measurement<double>::value);
  EXPECT_FALSE(is_measurement<variant<double>>::value);
}

class Complete {};

class Incomplete;

TEST(test_traits_covariance, test_is_complete) {
  EXPECT_TRUE(bool(is_complete<Complete>::value));
  EXPECT_FALSE(bool(is_complete<Incomplete>::value));
}

}  // namespace albatross
