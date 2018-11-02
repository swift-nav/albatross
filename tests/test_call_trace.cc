/*
 * Copyright (C) 2018 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

// clang-format off
#include "covariance_functions/covariance_functions.h"
#include "covariance_functions/call_trace.h"
// clang-format on
#include <gtest/gtest.h>
#include <iostream>

namespace albatross {

struct X {};
struct Y {};

class DefinedForX : public CovarianceFunction<DefinedForX> {
public:
  double call_impl_(const X &x, const X &y) const { return 1.; }
  std::string name_ = "defined_for_x";
};

class DefinedForY : public CovarianceFunction<DefinedForY> {
public:
  double call_impl_(const Y &x, const Y &y) const { return 3.; }
  std::string name_ = "defined_for_y";
};

class DefinedForXY : public CovarianceFunction<DefinedForXY> {
public:
  double call_impl_(const X &x, const X &y) const { return 5.; }

  double call_impl_(const X &x, const Y &y) const { return 7.; }

  double call_impl_(const Y &x, const Y &y) const { return 9.; }
  std::string name_ = "defined_for_xy";
};

template <typename T> class CallTraceTest {
public:
  virtual int expected_number_of_calls() = 0;

  virtual void check_expected_values(const X &x, const Y &y) = 0;

  T covariance_function;
};

class SumXandXY : public CallTraceTest<
                      SumOfCovarianceFunctions<DefinedForX, DefinedForXY>> {
public:
  int expected_number_of_calls() override { return 3; }

  void check_expected_values(const X &x, const Y &y) override {
    EXPECT_DOUBLE_EQ(this->covariance_function(x, x), 6.);
    EXPECT_DOUBLE_EQ(this->covariance_function(x, y), 7.);
    EXPECT_DOUBLE_EQ(this->covariance_function(y, x), 7.);
    EXPECT_DOUBLE_EQ(this->covariance_function(y, y), 9.);
  }
};

class SumXandY
    : public CallTraceTest<SumOfCovarianceFunctions<DefinedForX, DefinedForY>> {
public:
  int expected_number_of_calls() override { return 3; }

  void check_expected_values(const X &x, const Y &y) override {
    EXPECT_DOUBLE_EQ(this->covariance_function(x, x), 1.);
    EXPECT_DOUBLE_EQ(this->covariance_function(y, y), 3.);
  }
};

class SumSumXandYandXY
    : public CallTraceTest<SumOfCovarianceFunctions<
          SumOfCovarianceFunctions<DefinedForX, DefinedForY>, DefinedForXY>> {
public:
  int expected_number_of_calls() override { return 5; }

  void check_expected_values(const X &x, const Y &y) override {
    EXPECT_DOUBLE_EQ(this->covariance_function(x, x), 6.);
    EXPECT_DOUBLE_EQ(this->covariance_function(y, x), 7.);
    EXPECT_DOUBLE_EQ(this->covariance_function(x, y), 7.);
    EXPECT_DOUBLE_EQ(this->covariance_function(y, y), 12.);
  }
};

class ProdXandXY
    : public CallTraceTest<
          ProductOfCovarianceFunctions<DefinedForX, DefinedForXY>> {
public:
  int expected_number_of_calls() override { return 3; };

  void check_expected_values(const X &x, const Y &y) override {
    EXPECT_DOUBLE_EQ(this->covariance_function(x, x), 5.);
    EXPECT_DOUBLE_EQ(this->covariance_function(y, x), 7.);
    EXPECT_DOUBLE_EQ(this->covariance_function(x, y), 7.);
    EXPECT_DOUBLE_EQ(this->covariance_function(y, y), 9.);
  }
};

class ProdSumXandXYProdXandXY
    : public CallTraceTest<ProductOfCovarianceFunctions<
          SumOfCovarianceFunctions<DefinedForX, DefinedForXY>,
          ProductOfCovarianceFunctions<DefinedForX, DefinedForXY>>> {
public:
  int expected_number_of_calls() override { return 7; };

  void check_expected_values(const X &x, const Y &y) override {
    EXPECT_DOUBLE_EQ(this->covariance_function(x, x), 30.);
    EXPECT_DOUBLE_EQ(this->covariance_function(y, x), 49.);
    EXPECT_DOUBLE_EQ(this->covariance_function(x, y), 49.);
    EXPECT_DOUBLE_EQ(this->covariance_function(y, y), 81.);
  }
};

/*
 * In the following we test any covariance functions which should support
 * Eigen::Vector feature vectors.
 */
template <typename T>
class TestCallTreeCovarianceFunctions : public ::testing::Test {

public:
  T test_case;
};

typedef ::testing::Types<SumXandXY, SumXandY, SumSumXandYandXY, ProdXandXY,
                         ProdSumXandXYProdXandXY>
    TestFunctions;

TYPED_TEST_CASE(TestCallTreeCovarianceFunctions, TestFunctions);

TYPED_TEST(TestCallTreeCovarianceFunctions, prints_call_trace) {
  X x;
  Y y;

  this->test_case.check_expected_values(x, y);

  const auto calls_xx =
      this->test_case.covariance_function.call_trace().get_trace(x, x);
  EXPECT_EQ(calls_xx.size(), this->test_case.expected_number_of_calls());

  const auto calls_xy =
      this->test_case.covariance_function.call_trace().get_trace(x, y);
  EXPECT_EQ(calls_xy.size(), this->test_case.expected_number_of_calls());

  const auto calls_yy =
      this->test_case.covariance_function.call_trace().get_trace(y, y);
  EXPECT_EQ(calls_yy.size(), this->test_case.expected_number_of_calls());
}

} // namespace albatross
