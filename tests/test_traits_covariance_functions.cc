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

#include <albatross/CovarianceFunctions>
#include <gtest/gtest.h>

#include "test_covariance_utils.h"

namespace albatross {

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

TEST(test_traits_covariance, test_has_any_call_impl) {
  EXPECT_TRUE(bool(has_any_call_impl<HasPublicCallImpl>::value));
  EXPECT_TRUE(bool(has_any_call_impl<HasProtectedCallImpl>::value));
  EXPECT_TRUE(bool(has_any_call_impl<HasPrivateCallImpl>::value));
  EXPECT_FALSE(bool(has_any_call_impl<HasNoCallImpl>::value));
}

TEST(test_traits_covariance, test_has_valid_call_impl) {
  EXPECT_TRUE(bool(has_valid_call_impl<HasPublicCallImpl, X, Y>::value));
  EXPECT_FALSE(bool(has_valid_call_impl<HasPublicCallImpl, Y, X>::value));
  EXPECT_TRUE(bool(has_valid_call_impl<HasMultiple, X, X>::value));
  EXPECT_TRUE(bool(has_valid_call_impl<HasMultiple, Y, Y>::value));
  EXPECT_FALSE(bool(has_valid_call_impl<HasMultiple, Z, X>::value));
  EXPECT_FALSE(bool(has_valid_call_impl<HasMultiple, Z, Y>::value));
  EXPECT_FALSE(bool(has_valid_call_impl<HasMultiple, Z, Z>::value));
}

/*
 * Here we test to make sure we can identify situations where
 * _call_impl( has been defined but not necessarily properly.
 */
TEST(test_traits_covariance, test_has_possible_call_impl) {
  EXPECT_TRUE(bool(has_possible_call_impl<HasPublicCallImpl, X, Y>::value));
  EXPECT_FALSE(bool(has_possible_call_impl<HasPublicCallImpl, Y, X>::value));
  EXPECT_TRUE(bool(has_possible_call_impl<HasMultiple, X, X>::value));
  EXPECT_TRUE(bool(has_possible_call_impl<HasMultiple, Y, Y>::value));
  EXPECT_TRUE(bool(has_possible_call_impl<HasMultiple, Z, X>::value));
  EXPECT_TRUE(bool(has_possible_call_impl<HasMultiple, Z, Y>::value));
  EXPECT_TRUE(bool(has_possible_call_impl<HasMultiple, Z, Z>::value));
}

TEST(test_traits_covariance, test_has_invalid_call_impl) {
  EXPECT_FALSE(bool(has_invalid_call_impl<HasPublicCallImpl, X, Y>::value));
  EXPECT_FALSE(bool(has_invalid_call_impl<HasPublicCallImpl, Y, X>::value));
  EXPECT_FALSE(bool(has_invalid_call_impl<HasMultiple, X, X>::value));
  EXPECT_FALSE(bool(has_invalid_call_impl<HasMultiple, Y, Y>::value));
  EXPECT_TRUE(bool(has_invalid_call_impl<HasMultiple, Z, X>::value));
  EXPECT_TRUE(bool(has_invalid_call_impl<HasMultiple, Z, Y>::value));
  EXPECT_TRUE(bool(has_invalid_call_impl<HasMultiple, Z, Z>::value));
}

TEST(test_traits_covariance_function, test_operator_resolution) {

  EXPECT_TRUE(bool(has_call_operator<HasXY, X, Y>::value));
  EXPECT_TRUE(bool(has_call_operator<HasXY, Y, X>::value));
  EXPECT_FALSE(bool(has_call_operator<HasXY, X, X>::value));
  EXPECT_FALSE(bool(has_call_operator<HasXY, Y, Y>::value));
  EXPECT_FALSE(bool(has_call_operator<HasXY, Z, Z>::value));

  EXPECT_FALSE(bool(has_call_operator<HasNone, X, Y>::value));
  EXPECT_FALSE(bool(has_call_operator<HasNone, Y, X>::value));
  EXPECT_FALSE(bool(has_call_operator<HasNone, X, X>::value));
  EXPECT_FALSE(bool(has_call_operator<HasNone, Y, Y>::value));
  EXPECT_FALSE(bool(has_call_operator<HasNone, Z, Z>::value));

  EXPECT_TRUE(bool(has_call_operator<HasMultiple, X, Y>::value));
  EXPECT_TRUE(bool(has_call_operator<HasMultiple, Y, X>::value));
  EXPECT_TRUE(bool(has_call_operator<HasMultiple, X, X>::value));
  EXPECT_TRUE(bool(has_call_operator<HasMultiple, Y, Y>::value));
  EXPECT_FALSE(bool(has_call_operator<HasMultiple, Z, Z>::value));
}

TEST(test_traits_covariance_function, test_vector_operator_inspection) {
  EXPECT_TRUE(
      bool(has_call_operator<HasXY, std::vector<X>, std::vector<Y>>::value));
  EXPECT_TRUE(
      bool(has_call_operator<HasXY, std::vector<Y>, std::vector<X>>::value));
  EXPECT_FALSE(
      bool(has_call_operator<HasXY, std::vector<X>, std::vector<X>>::value));
  EXPECT_FALSE(
      bool(has_call_operator<HasXY, std::vector<Y>, std::vector<Y>>::value));
  EXPECT_FALSE(
      bool(has_call_operator<HasXY, std::vector<Z>, std::vector<Z>>::value));

  EXPECT_TRUE(bool(
      has_call_operator<HasMultiple, std::vector<X>, std::vector<Y>>::value));
  EXPECT_TRUE(bool(
      has_call_operator<HasMultiple, std::vector<Y>, std::vector<X>>::value));
  EXPECT_TRUE(bool(
      has_call_operator<HasMultiple, std::vector<X>, std::vector<X>>::value));
  EXPECT_TRUE(bool(
      has_call_operator<HasMultiple, std::vector<Y>, std::vector<Y>>::value));
  EXPECT_FALSE(bool(
      has_call_operator<HasMultiple, std::vector<Z>, std::vector<Z>>::value));
}

TEST(test_traits_covariance_function, test_has_valid_caller_for_all_variants) {

  // With one type
  EXPECT_TRUE(bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                                     variant<W>>::value));
  EXPECT_TRUE(bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                                     variant<X>>::value));
  EXPECT_TRUE(bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                                     variant<Y>>::value));

  EXPECT_FALSE(
      bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                             variant<Z>>::value));

  // With two types
  EXPECT_TRUE(bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                                     variant<X, W>>::value));
  EXPECT_TRUE(bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                                     variant<W, X>>::value));
  EXPECT_TRUE(bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                                     variant<X, Y>>::value));
  EXPECT_TRUE(bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                                     variant<Y, X>>::value));

  EXPECT_FALSE(
      bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                             variant<X, Z>>::value));
  EXPECT_FALSE(
      bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                             variant<Z, X>>::value));
  EXPECT_FALSE(
      bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                             variant<W, Z>>::value));

  // With three types
  EXPECT_TRUE(bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                                     variant<W, X, Y>>::value));
  EXPECT_TRUE(bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                                     variant<W, Y, X>>::value));
  EXPECT_TRUE(bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                                     variant<X, W, Y>>::value));
  EXPECT_TRUE(bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                                     variant<X, Y, W>>::value));
  EXPECT_TRUE(bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                                     variant<Y, X, W>>::value));
  EXPECT_TRUE(bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                                     variant<Y, W, X>>::value));

  EXPECT_FALSE(
      bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                             variant<X, Y, Z>>::value));
  EXPECT_FALSE(
      bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                             variant<X, Z, Y>>::value));
  EXPECT_FALSE(
      bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                             variant<Y, X, Z>>::value));
  EXPECT_FALSE(
      bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                             variant<Y, Z, X>>::value));
  EXPECT_FALSE(
      bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                             variant<Z, X, Y>>::value));
  EXPECT_FALSE(
      bool(has_valid_caller_for_all_variants<HasMultiple, DefaultCaller,
                                             variant<Z, Y, X>>::value));
}

TEST(test_traits_covariance_function,
     test_has_valid_cross_cov_caller_for_variant) {
  // With one type
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, X,
                                              variant<X>>::value));
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, X,
                                              variant<Y>>::value));
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Y,
                                              variant<X>>::value));
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Y,
                                              variant<Y>>::value));

  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, X,
                                               variant<W>>::value));
  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, W,
                                               variant<X>>::value));
  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, X,
                                               variant<Z>>::value));
  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Y,
                                               variant<W>>::value));
  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Y,
                                               variant<Z>>::value));
  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, W,
                                               variant<Y>>::value));
  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, W,
                                               variant<Z>>::value));

  // With two types
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, X,
                                              variant<X, Y>>::value));
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, X,
                                              variant<Y, W>>::value));
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Y,
                                              variant<X, Y>>::value));
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Y,
                                              variant<X, W>>::value));
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Y,
                                              variant<Y, W>>::value));

  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, X,
                                               variant<Y, Z>>::value));
  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Y,
                                               variant<X, Z>>::value));
  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, X,
                                               variant<W, Z>>::value));
  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, X,
                                               variant<Z, W>>::value));
  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Y,
                                               variant<W, Z>>::value));
  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Y,
                                               variant<Z, W>>::value));
  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Z,
                                               variant<W, Y>>::value));
  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, W,
                                               variant<Y, Z>>::value));

  // With three types
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, X,
                                              variant<Y, W, V>>::value));
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, X,
                                              variant<Y, V, W>>::value));
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, X,
                                              variant<V, Y, W>>::value));
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, X,
                                              variant<V, W, Y>>::value));
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, X,
                                              variant<W, Y, V>>::value));
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, X,
                                              variant<W, V, Y>>::value));

  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Y,
                                              variant<X, W, V>>::value));
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Y,
                                              variant<X, V, W>>::value));
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Y,
                                              variant<V, X, W>>::value));
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Y,
                                              variant<V, W, X>>::value));
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Y,
                                              variant<W, X, V>>::value));
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Y,
                                              variant<W, V, X>>::value));

  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, W,
                                              variant<X, W, V>>::value));
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, W,
                                              variant<X, V, W>>::value));
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, W,
                                              variant<V, X, W>>::value));
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, W,
                                              variant<V, W, X>>::value));
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, W,
                                              variant<W, X, V>>::value));
  EXPECT_TRUE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, W,
                                              variant<W, V, X>>::value));

  // Since Z isn't valid all variants including it are invalid.
  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, X,
                                               variant<Z, X, Y>>::value));
  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Y,
                                               variant<Z, X, Y>>::value));
  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, W,
                                               variant<Z, X, Y>>::value));
  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, V,
                                               variant<Z, X, Y>>::value));

  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Z,
                                               variant<W, X, Y>>::value));
  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Z,
                                               variant<W, Y, X>>::value));
  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Z,
                                               variant<X, W, Y>>::value));
  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Z,
                                               variant<X, Y, W>>::value));
  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Z,
                                               variant<Y, W, X>>::value));
  EXPECT_FALSE(bool(has_valid_cross_cov_caller<HasMultiple, DefaultCaller, Z,
                                               variant<Y, X, W>>::value));
}

TEST(test_traits_covariance_function, test_has_valid_variant_cov_call) {
  EXPECT_TRUE(bool(has_valid_variant_cov_caller<HasMultiple, DefaultCaller, X,
                                                variant<X, Y>>::value));
  EXPECT_TRUE(bool(has_valid_variant_cov_caller<HasMultiple, DefaultCaller, X,
                                                variant<X, Y, Z>>::value));
  EXPECT_TRUE(bool(has_valid_variant_cov_caller<HasMultiple, DefaultCaller,
                                                variant<X, Y>, X>::value));
  EXPECT_TRUE(
      bool(has_valid_variant_cov_caller<HasMultiple, DefaultCaller,
                                        variant<X, Y>, variant<X, Y>>::value));

  EXPECT_FALSE(bool(has_valid_variant_cov_caller<HasMultiple, DefaultCaller, Z,
                                                 variant<X, Y>>::value));
}

} // namespace albatross
