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

#include <albatross/GroupBy>

#include <gtest/gtest.h>

namespace albatross {

struct X {};

std::string free_string_const_ref_int(const int &x) { return "int"; }

std::string free_string_ref_int(int &x) { return free_string_const_ref_int(x); }

std::string free_string_int(int x) { return free_string_const_ref_int(x); }

std::string free_string_X(const X &x) { return "x"; }

const auto lambda_string_const_ref_int = [](const int &x) {
  return free_string_const_ref_int(x);
};

struct CallOperatorStringConstRefInt {
  std::string operator()(const int &x) { return free_string_const_ref_int(x); }
};

CallOperatorStringConstRefInt call_operator_string_const_ref_int;

template <typename T> class TestCanBeCalledWithInt : public ::testing::Test {};

typedef ::testing::Types<
    decltype(free_string_const_ref_int),
    //    decltype(free_string_ref_int),  WHY DOESN'T THIS ONE WORK??
    decltype(free_string_int), decltype(lambda_string_const_ref_int),
    decltype(call_operator_string_const_ref_int), CallOperatorStringConstRefInt>
    TestFunctions;

TYPED_TEST_CASE(TestCanBeCalledWithInt, TestFunctions);

TYPED_TEST(TestCanBeCalledWithInt, test_values) {
  EXPECT_TRUE(bool(details::can_be_called_with<TypeParam, int>::value));
  // HOW SHOULD IT BEHAVE IF YOU CAN CONVERT AN ARGUMENT AND CALL???
  // EXPECT_TRUE(bool(details::can_be_called_with<ToTest, double>::value));
  EXPECT_FALSE(bool(details::can_be_called_with<TypeParam, X>::value));
}

TYPED_TEST(TestCanBeCalledWithInt, test_return_types) {
  EXPECT_TRUE(bool(
      std::is_same<std::string, typename details::return_type_when_called_with<
                                    TypeParam, int>::type>::value));
}

TYPED_TEST(TestCanBeCalledWithInt, test_is_valid_grouper) {
  EXPECT_TRUE(bool(details::is_valid_grouper<TypeParam, int>::value));
  EXPECT_FALSE(bool(details::is_valid_grouper<TypeParam, X>::value));
}

template <typename... Args> struct Tester {

  //  template <typename FunctionType>
  //  bool test(const FunctionType &&) const {
  //    return details::can_be_called_with<FunctionType, Args...>::value;
  //  }

  template <typename FunctionType> bool test(const FunctionType &) const {
    return details::can_be_called_with<FunctionType, Args...>::value;
  }
};

// Here we make sure can_be_called_with works when the type of the
// functions is deduced by the compiler.
TEST(test_traits_group_by, test_can_be_called_with_deduction) {
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
  EXPECT_FALSE(x_tester.test(
      [](const int &x) { return free_string_const_ref_int(x); }));
}

} // namespace albatross
