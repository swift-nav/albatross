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

#include <gtest/gtest.h>
#include "../include/albatross/Indexing"

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

std::string free_string_const_ref_int(const int &x ALBATROSS_UNUSED) {
  return "int";
}

std::string free_string_ref_int(int &x) { return free_string_const_ref_int(x); }

std::string free_string_int(int x) { return free_string_const_ref_int(x); }

std::string free_string_X(const X &x ALBATROSS_UNUSED) { return "x"; }

const auto lambda_string_const_ref_int = [](const int &x) {
  return free_string_const_ref_int(x);
};

const auto lambda_string_ref_int = [](int &x) {
  return free_string_const_ref_int(x);
};

struct CallOperatorStringConstRefInt {
  std::string operator()(const int &x) const {
    return free_string_const_ref_int(x);
  }
};

CallOperatorStringConstRefInt call_operator_string_const_ref_int;

template <typename T>
class TestCanBeCalledWithInt : public ::testing::Test {};

template <typename _FunctionType, typename _ReturnType>
struct WithReturnType {
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

TYPED_TEST_SUITE(TestCanBeCalledWithInt, TestFunctions);

TYPED_TEST(TestCanBeCalledWithInt, test_invocalbe) {
  using Expected = typename TypeParam::ReturnType;

  using FunctionType = typename TypeParam::FunctionType;
  EXPECT_TRUE(bool(is_invocable<FunctionType, int>::value));
  EXPECT_TRUE(bool(
      std::is_same<Expected,
                   typename invoke_result<FunctionType, int>::type>::value));

  // HOW SHOULD IT BEHAVE IF YOU CAN CONVERT AN ARGUMENT AND CALL???
  EXPECT_TRUE(bool(is_invocable<FunctionType, double>::value));
  EXPECT_TRUE(bool(
      std::is_same<Expected,
                   typename invoke_result<FunctionType, double>::type>::value));

  EXPECT_FALSE(bool(is_invocable<FunctionType, X>::value));
}

TYPED_TEST(TestCanBeCalledWithInt, test_is_valid_grouper) {
  using FunctionType = typename TypeParam::FunctionType;
  EXPECT_TRUE(bool(details::is_valid_grouper<FunctionType, int>::value));
  EXPECT_FALSE(bool(details::is_valid_grouper<FunctionType, X>::value));
}

template <typename FunctionType>
bool test_can_be_called_with_int(FunctionType &&) {
  return is_invocable<FunctionType, int>::value ||
         is_invocable<FunctionType, int &>::value;
}

template <typename FunctionType>
bool test_can_be_called_with_x(FunctionType &&) {
  return is_invocable<FunctionType, X>::value ||
         is_invocable<FunctionType, X &>::value;
}

// Here we make sure can_be_called_with works when the type of the
// functions is deduced by the compiler.
TEST(test_traits_indexing, test_can_be_called_with_deduction) {
  EXPECT_TRUE(test_can_be_called_with_int(free_string_const_ref_int));
  EXPECT_TRUE(test_can_be_called_with_int(free_string_ref_int));
  EXPECT_TRUE(test_can_be_called_with_int(free_string_int));
  EXPECT_TRUE(test_can_be_called_with_int(lambda_string_const_ref_int));
  EXPECT_TRUE(test_can_be_called_with_int(lambda_string_ref_int));

  EXPECT_TRUE(test_can_be_called_with_int(call_operator_string_const_ref_int));
  EXPECT_TRUE(test_can_be_called_with_int(CallOperatorStringConstRefInt()));
  EXPECT_TRUE(test_can_be_called_with_int(
      [](const int &x) { return free_string_const_ref_int(x); }));
  EXPECT_TRUE(test_can_be_called_with_int(
      [](const auto &x) { return free_string_const_ref_int(x); }));

  EXPECT_FALSE(test_can_be_called_with_x(free_string_const_ref_int));
  EXPECT_FALSE(test_can_be_called_with_x(free_string_ref_int));
  EXPECT_FALSE(test_can_be_called_with_x(free_string_int));
  EXPECT_FALSE(test_can_be_called_with_x(lambda_string_const_ref_int));
  EXPECT_FALSE(test_can_be_called_with_x(call_operator_string_const_ref_int));
  EXPECT_FALSE(test_can_be_called_with_x(CallOperatorStringConstRefInt()));
  EXPECT_FALSE(test_can_be_called_with_x(
      [](const int &x) { return free_string_const_ref_int(x); }));
}

/*
 * Test Filter Function Traits
 */
bool free_bool_const_ref_int(const int &x ALBATROSS_UNUSED) { return true; }

bool free_bool_ref_int(int &x) { return free_bool_const_ref_int(x); }

bool free_bool_int(int x) { return free_bool_const_ref_int(x); }

bool free_bool_X(const X &x ALBATROSS_UNUSED) { return true; }

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

TYPED_TEST_SUITE(TestIsValueFilterFunction, ValueFilterFunctionTestCases);

TYPED_TEST(TestIsValueFilterFunction, test_is_value_filter_function) {
  EXPECT_TRUE(bool(
      details::is_valid_value_only_filter_function<TypeParam, int>::value));
}

}  // namespace albatross
