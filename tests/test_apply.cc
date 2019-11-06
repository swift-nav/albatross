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

#include <albatross/Indexing>
#include <gtest/gtest.h>

namespace albatross {

struct Foo {
  Foo() : value(){};
  Foo(const double &x) : value(x){};

  bool operator==(const Foo &other) const {
    return fabs(other.value - value) < std::numeric_limits<double>::epsilon();
  }

  double value;
};

TEST(test_apply, test_vector_apply) {

  const auto xs = linspace(0., 10., 11);

  const auto make_foo = [](const double &x) { return Foo(x); };

  const auto applied = apply(xs, make_foo);

  std::vector<Foo> expected;
  for (const auto &x : xs) {
    expected.emplace_back(Foo(x));
  }

  EXPECT_EQ(expected, applied);
}

TEST(test_apply, test_vector_apply_void) {

  const auto xs = linspace(0., 10., 11);

  std::size_t call_count = 0;

  const auto count_calls = [&](const double &x) { ++call_count; };

  apply(xs, count_calls);

  EXPECT_EQ(call_count, xs.size());
}

TEST(test_apply, test_vector_apply_all) {

  std::vector<std::vector<bool>> input;
  std::vector<bool> expected;

  std::vector<bool> empty = {};
  input.push_back(empty);
  expected.push_back(false);

  input.push_back({true});
  expected.push_back(true);

  input.push_back({false});
  expected.push_back(false);

  input.push_back({true, true});
  expected.push_back(true);

  input.push_back({true, false});
  expected.push_back(false);

  input.push_back({false, true});
  expected.push_back(false);

  input.push_back({false, false});
  expected.push_back(false);

  input.push_back({true, true, true});
  expected.push_back(true);

  input.push_back({true, false, true});
  expected.push_back(false);

  const auto actual = apply(input, all);

  EXPECT_EQ(actual, expected);
}

TEST(test_apply, test_vector_apply_any) {

  std::vector<std::vector<bool>> input;
  std::vector<bool> expected;

  std::vector<bool> empty = {};
  input.push_back(empty);
  expected.push_back(false);

  input.push_back({true});
  expected.push_back(true);

  input.push_back({false});
  expected.push_back(false);

  input.push_back({true, true});
  expected.push_back(true);

  input.push_back({true, false});
  expected.push_back(true);

  input.push_back({false, true});
  expected.push_back(true);

  input.push_back({false, false});
  expected.push_back(false);

  input.push_back({true, true, true});
  expected.push_back(true);

  input.push_back({true, false, true});
  expected.push_back(true);

  const auto actual = apply(input, any);

  EXPECT_EQ(actual, expected);
}

} // namespace albatross
