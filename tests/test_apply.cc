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

std::vector<double> test_double_vector() { return linspace(0., 10., 11); }

double square(double x) { return x * x; }

struct Foo {
  Foo() : value(){};
  Foo(const double &x) : value(x){};

  bool operator==(const Foo &other) const {
    return fabs(other.value - value) < std::numeric_limits<double>::epsilon();
  }

  double value;
};

Foo make_foo(double x) { return Foo(x); }

class Square {
public:
  double operator()(double x) const { return square(x); }
};

/*
 * Test Cases
 */

struct SquareClassMethodApply {

  auto get_parent() const { return test_double_vector(); }

  auto get_function() const { return Square(); }
};

struct SquareFunctionPointerApply {

  auto get_parent() const { return test_double_vector(); }

  auto get_function() const { return &square; }
};

struct SquareFunctionApply {

  auto get_parent() const { return test_double_vector(); }

  auto get_function() const { return square; }
};

struct SquareLambdaApply {

  auto get_parent() const { return test_double_vector(); }

  auto get_function() const {
    const auto lambda_square = [](double x) { return square(x); };
    return lambda_square;
  }
};

struct MakeFooFunctionApply {

  auto get_parent() const { return test_double_vector(); }

  auto get_function() const { return make_foo; }
};

template <typename CaseType> class ApplyTester : public ::testing::Test {
public:
  CaseType test_case;
};

typedef ::testing::Types<SquareClassMethodApply, SquareFunctionPointerApply,
                         SquareFunctionApply, SquareLambdaApply>
    ApplyTestCases;

TYPED_TEST_CASE_P(ApplyTester);

TYPED_TEST_P(ApplyTester, test_apply_sanity) {
  auto parent = this->test_case.get_parent();
  const auto actual = apply(parent, this->test_case.get_function());

  typename std::remove_const<decltype(actual)>::type expected;
  for (const auto &x : parent) {
    expected.emplace_back(this->test_case.get_function()(x));
  }

  EXPECT_EQ(expected, actual);
}

REGISTER_TYPED_TEST_CASE_P(ApplyTester, test_apply_sanity);

INSTANTIATE_TYPED_TEST_CASE_P(test_apply, ApplyTester, ApplyTestCases);

TEST(test_apply, test_vector_apply_free_function) {

  const auto xs = linspace(0., 10., 11);
  const auto actual = apply(xs, square);

  std::vector<double> expected;
  for (const auto &x : xs) {
    expected.push_back(x * x);
  }

  EXPECT_EQ(expected.size(), actual.size());
  EXPECT_EQ(expected, actual);
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
