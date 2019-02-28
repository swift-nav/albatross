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

#include "CovarianceFunctions"

#include <gtest/gtest.h>

namespace albatross {

class Complete {};

class Incomplete;

TEST(test_traits_covariance, test_is_complete) {
  EXPECT_TRUE(bool(is_complete<Complete>::value));
  EXPECT_FALSE(bool(is_complete<Incomplete>::value));
}

class HasName {
public:
  std::string name_;
};

class HasNoName {};

TEST(test_traits_covariance, test_has_name) {
  EXPECT_TRUE(bool(has_name_<HasName>::value));
  EXPECT_FALSE(bool(has_name_<HasNoName>::value));
}

struct X {};
struct Y {};
struct Z {};

class HasPublicCallOperator {
public:
  double operator()(const X &, const Y &) const { return 1.; };
};

class HasProtectedCallOperator {
protected:
  double operator()(const X &, const Y &) const { return 1.; };
};

class HasPrivateCallOperator {
  double operator()(const X &, const Y &) const { return 1.; };
};

class HasNoCallOperator {};

TEST(test_traits_covariance, test_has_call_operator) {
  EXPECT_TRUE(bool(has_call_operator<HasPublicCallOperator, X, Y>::value));
  EXPECT_FALSE(bool(has_call_operator<HasPrivateCallOperator, X, Y>::value));
  EXPECT_FALSE(bool(has_call_operator<HasProtectedCallOperator, X, Y>::value));
  EXPECT_FALSE(bool(has_call_operator<HasNoCallOperator, X, Y>::value));
}

class HasPublicCallImpl {
public:
  double call_impl_(const X &, const Y &) const { return 1.; };
};

class HasProtectedCallImpl {
protected:
  double call_impl_(const X &, const Y &) const { return 1.; };
};

class HasPrivateCallImpl {
  double call_impl_(const X &, const Y &) const { return 1.; };
};

class HasNoCallImpl {};

TEST(test_traits_covariance, test_has_any_call_impl_) {
  EXPECT_TRUE(bool(has_any_call_impl<HasPublicCallImpl>::value));
  EXPECT_TRUE(bool(has_any_call_impl<HasProtectedCallImpl>::value));
  EXPECT_TRUE(bool(has_any_call_impl<HasPrivateCallImpl>::value));
  EXPECT_FALSE(bool(has_any_call_impl<HasNoCallImpl>::value));
}

class HasMultiplePublicCallImpl {
public:
  double call_impl_(const X &, const Y &) const { return 1.; };

  double call_impl_(const X &, const X &) const { return 1.; };

  double call_impl_(const Y &, const Y &) const { return 1.; };

  // These are all invalid:
  double call_impl_(const Z &, const X &) { return 1.; };

  double call_impl_(Z &, const Y &) const { return 1.; };

  int call_impl_(const Z &, const Z &) const { return 1.; };
};

TEST(test_traits_covariance, test_has_valid_call_impl) {
  EXPECT_TRUE(bool(has_valid_call_impl<HasPublicCallImpl, X, Y>::value));
  EXPECT_FALSE(bool(has_valid_call_impl<HasPublicCallImpl, Y, X>::value));
  EXPECT_TRUE(
      bool(has_valid_call_impl<HasMultiplePublicCallImpl, X, X>::value));
  EXPECT_TRUE(
      bool(has_valid_call_impl<HasMultiplePublicCallImpl, Y, Y>::value));
  EXPECT_FALSE(
      bool(has_valid_call_impl<HasMultiplePublicCallImpl, Z, X>::value));
  EXPECT_FALSE(
      bool(has_valid_call_impl<HasMultiplePublicCallImpl, Z, Y>::value));
  EXPECT_FALSE(
      bool(has_valid_call_impl<HasMultiplePublicCallImpl, Z, Z>::value));
}

/*
 * Here we test to make sure we can identify situations where
 * call_impl_ has been defined but not necessarily properly.
 */
TEST(test_traits_covariance, test_has_possible_call_impl) {
  EXPECT_TRUE(bool(has_possible_call_impl<HasPublicCallImpl, X, Y>::value));
  EXPECT_FALSE(bool(has_possible_call_impl<HasPublicCallImpl, Y, X>::value));
  EXPECT_TRUE(
      bool(has_possible_call_impl<HasMultiplePublicCallImpl, X, X>::value));
  EXPECT_TRUE(
      bool(has_possible_call_impl<HasMultiplePublicCallImpl, Y, Y>::value));
  EXPECT_TRUE(
      bool(has_possible_call_impl<HasMultiplePublicCallImpl, Z, X>::value));
  EXPECT_TRUE(
      bool(has_possible_call_impl<HasMultiplePublicCallImpl, Z, Y>::value));
  EXPECT_TRUE(
      bool(has_possible_call_impl<HasMultiplePublicCallImpl, Z, Z>::value));
}

TEST(test_traits_covariance, test_has_invalid_call_impl) {
  EXPECT_FALSE(bool(has_invalid_call_impl<HasPublicCallImpl, X, Y>::value));
  EXPECT_FALSE(bool(has_invalid_call_impl<HasPublicCallImpl, Y, X>::value));
  EXPECT_FALSE(
      bool(has_invalid_call_impl<HasMultiplePublicCallImpl, X, X>::value));
  EXPECT_FALSE(
      bool(has_invalid_call_impl<HasMultiplePublicCallImpl, Y, Y>::value));
  EXPECT_TRUE(
      bool(has_invalid_call_impl<HasMultiplePublicCallImpl, Z, X>::value));
  EXPECT_TRUE(
      bool(has_invalid_call_impl<HasMultiplePublicCallImpl, Z, Y>::value));
  EXPECT_TRUE(
      bool(has_invalid_call_impl<HasMultiplePublicCallImpl, Z, Z>::value));
}

} // namespace albatross
