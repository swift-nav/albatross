/*
 * Copyright (C) 2019 Swift Navigation Inc.
 * Contact: Swift Na vigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "../include/albatross/Indexing"
#include <gtest/gtest.h>

namespace albatross {

struct X {
  int something;
  std::string other;
};

TEST(test_traits_indexing, test_is_valid_map_key) {
  EXPECT_TRUE(bool(details::is_valid_map_key<int>::value));
  EXPECT_FALSE(bool(details::is_valid_map_key<X>::value));
  EXPECT_FALSE(bool(details::is_valid_map_key<void>::value));
}

std::string free_string_const_ref_int(const int &x) { return "int"; }

std::string free_string_ref_int(int &x) { return free_string_const_ref_int(x); }

std::string free_string_int(int x) { return free_string_const_ref_int(x); }

std::string free_string_X(const X &x) { return "x"; }

const auto lambda_string_const_ref_int = [](const int &x) {
  return free_string_const_ref_int(x);
};

struct CallOperatorStringConstRefInt {
  std::string operator()(const int &x) const {
    return free_string_const_ref_int(x);
  }
};

CallOperatorStringConstRefInt call_operator_string_const_ref_int;

template <typename T> class TestCanBeCalledWithInt : public ::testing::Test {};

template <typename _FunctionType, typename _ReturnType> struct WithReturnType {
  using FunctionType = _FunctionType;
  using ReturnType = _ReturnType;
};

typedef ::testing::Types<
    WithReturnType<decltype(free_string_const_ref_int), std::string>,
    //    decltype(free_string_ref_int),  WHY DOESN'T THIS ONE WORK??
    WithReturnType<decltype(free_string_int), std::string>,
    WithReturnType<decltype(lambda_string_const_ref_int), std::string>,
    WithReturnType<decltype(call_operator_string_const_ref_int), std::string>,
    WithReturnType<CallOperatorStringConstRefInt, std::string>>
    TestFunctions;

TYPED_TEST_CASE(TestCanBeCalledWithInt, TestFunctions);

TYPED_TEST(TestCanBeCalledWithInt, test_callable_traits) {
  using Expected = typename TypeParam::ReturnType;

  using FunctionType = typename TypeParam::FunctionType;
  EXPECT_TRUE(bool(details::callable_traits<FunctionType, int>::is_defined));
  EXPECT_TRUE(
      bool(std::is_same<Expected, typename details::callable_traits<
                                      FunctionType, int>::return_type>::value));
  // HOW SHOULD IT BEHAVE IF YOU CAN CONVERT AN ARGUMENT AND CALL???
  // EXPECT_TRUE(bool(details::can_be_called_with<ToTest, double>::value));
  EXPECT_FALSE(bool(details::callable_traits<FunctionType, X>::is_defined));
}

TYPED_TEST(TestCanBeCalledWithInt, test_can_be_called_with) {
  using FunctionType = typename TypeParam::FunctionType;
  EXPECT_TRUE(bool(details::can_be_called_with<FunctionType, int>::value));
}

TYPED_TEST(TestCanBeCalledWithInt, test_can_be_called_with_return_type) {
  using Expected = typename TypeParam::ReturnType;
  using FunctionType = typename TypeParam::FunctionType;
  EXPECT_TRUE(bool(
      std::is_same<Expected, typename details::return_type_when_called_with<
                                 FunctionType, int>::type>::value));
}

TYPED_TEST(TestCanBeCalledWithInt, test_is_valid_grouper) {
  using FunctionType = typename TypeParam::FunctionType;
  EXPECT_TRUE(bool(details::is_valid_grouper<FunctionType, int>::value));
  EXPECT_FALSE(bool(details::is_valid_grouper<FunctionType, X>::value));
}

template <typename... Args> struct Tester {
  template <typename FunctionType> bool test(const FunctionType &) const {
    return details::can_be_called_with<FunctionType, Args...>::value;
  }
};

// Here we make sure can_be_called_with works when the type of the
// functions is deduced by the compiler.
TEST(test_traits_indexing, test_can_be_called_with_deduction) {
  const Tester<int> int_tester;
  const Tester<X> x_tester;

  EXPECT_TRUE(int_tester.test(free_string_const_ref_int));
  //  EXPECT_TRUE(int_tester.test(free_string_ref_int));
  EXPECT_TRUE(int_tester.test(free_string_int));
  EXPECT_TRUE(int_tester.test(lambda_string_const_ref_int));
  EXPECT_TRUE(int_tester.test(call_operator_string_const_ref_int));
  EXPECT_TRUE(int_tester.test(CallOperatorStringConstRefInt()));
  EXPECT_TRUE(int_tester.test(
      [](const int &x) { return free_string_const_ref_int(x); }));
  EXPECT_TRUE(int_tester.test(
      [](const auto &x) { return free_string_const_ref_int(x); }));

  EXPECT_FALSE(x_tester.test(free_string_const_ref_int));
  EXPECT_FALSE(x_tester.test(free_string_ref_int));
  EXPECT_FALSE(x_tester.test(free_string_int));
  EXPECT_FALSE(x_tester.test(lambda_string_const_ref_int));
  EXPECT_FALSE(x_tester.test(call_operator_string_const_ref_int));
  EXPECT_FALSE(x_tester.test(CallOperatorStringConstRefInt()));
  EXPECT_FALSE(
      x_tester.test([](const int &x) { return free_string_const_ref_int(x); }));
}

/*
 * Test Filter Function Traits
 */
bool free_bool_const_ref_int(const int &x) { return true; }

bool free_bool_ref_int(int &x) { return free_bool_const_ref_int(x); }

bool free_bool_int(int x) { return free_bool_const_ref_int(x); }

bool free_bool_X(const X &x) { return true; }

const auto lambda_bool_const_ref_int = [](const int &x) {
  return free_bool_const_ref_int(x);
};

const bool need_this_or_clang_complains_about_unused_variables =
    lambda_bool_const_ref_int(0);

struct CallOperatorBoolConstRefInt {
  bool operator()(const int &x) const { return free_bool_const_ref_int(x); }
};

CallOperatorBoolConstRefInt call_operator_bool_const_ref_int;

template <typename T>
class TestIsValueFilterFunction : public ::testing::Test {};

typedef ::testing::Types<
    decltype(free_bool_const_ref_int), decltype(free_bool_int),
    decltype(lambda_bool_const_ref_int), CallOperatorBoolConstRefInt>
    ValueFilterFunctionTestCases;

TYPED_TEST_CASE(TestIsValueFilterFunction, ValueFilterFunctionTestCases);

TYPED_TEST(TestIsValueFilterFunction, test_is_value_filter_function) {
  EXPECT_TRUE(bool(
      details::is_valid_value_only_filter_function<TypeParam, int>::value));
}

} // namespace albatross
