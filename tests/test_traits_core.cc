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

#include "core/declarations.h"

#include "core/traits.h"
#include <gtest/gtest.h>

namespace albatross {

struct X {};
struct Y {};
struct Z {};

class HasValidFitImpl {
public:
  Fit<HasValidFitImpl> fit_impl_(const std::vector<X> &,
                                 const MarginalDistribution &) const {
    return {};
  };
};

class HasWrongReturnTypeFitImpl {
public:
  Fit<X> fit_impl_(const std::vector<X> &, const MarginalDistribution &) const {
    return {};
  };
};

class HasNonConstFitImpl {
public:
  Fit<HasNonConstFitImpl> fit_impl_(const std::vector<X> &,
                                    const MarginalDistribution &) {
    return {};
  };
};

class HasNonConstArgsFitImpl {
public:
  Fit<HasNonConstArgsFitImpl> fit_impl_(std::vector<X> &,
                                        const MarginalDistribution &) const {
    return {};
  };

  Fit<HasNonConstArgsFitImpl> fit_impl_(const std::vector<X> &,
                                        MarginalDistribution &) const {
    return {};
  };

  Fit<HasNonConstArgsFitImpl> fit_impl_(std::vector<X> &,
                                        MarginalDistribution &) const {
    return {};
  };
};

class HasProtectedValidFitImpl {
protected:
  Fit<HasProtectedValidFitImpl> fit_impl_(const std::vector<X> &,
                                          const MarginalDistribution &) const {
    return {};
  };
};

class HasPrivateValidFitImpl {
private:
  Fit<HasPrivateValidFitImpl> fit_impl_(const std::vector<X> &,
                                        const MarginalDistribution &) const {
    return {};
  };
};

class HasValidAndInvalidFitImpl {
public:
  Fit<HasValidAndInvalidFitImpl> fit_impl_(const std::vector<X> &,
                                           const MarginalDistribution &) const {
    return {};
  };

  Fit<HasValidAndInvalidFitImpl> fit_impl_(const std::vector<X> &,
                                           const MarginalDistribution &) {
    return {};
  };
};

class HasValidXYFitImpl {
public:
  Fit<HasValidXYFitImpl> fit_impl_(const std::vector<X> &,
                                   const MarginalDistribution &) const {
    return {};
  };

  Fit<HasValidXYFitImpl> fit_impl_(const std::vector<Y> &,
                                   const MarginalDistribution &) const {
    return {};
  };
};

class HasNoFitImpl {};

TEST(test_traits_core, test_has_valid_fit_impl_) {
  EXPECT_TRUE(bool(has_valid_fit_impl<HasValidFitImpl, X>::value));
  EXPECT_FALSE(bool(has_valid_fit_impl<HasWrongReturnTypeFitImpl, X>::value));
  EXPECT_FALSE(bool(has_valid_fit_impl<HasNonConstFitImpl, X>::value));
  EXPECT_FALSE(bool(has_valid_fit_impl<HasNonConstArgsFitImpl, X>::value));
  EXPECT_FALSE(bool(has_valid_fit_impl<HasProtectedValidFitImpl, X>::value));
  EXPECT_FALSE(bool(has_valid_fit_impl<HasPrivateValidFitImpl, X>::value));
  EXPECT_TRUE(bool(has_valid_fit_impl<HasValidAndInvalidFitImpl, X>::value));
  EXPECT_TRUE(bool(has_valid_fit_impl<HasValidXYFitImpl, X>::value));
  EXPECT_TRUE(bool(has_valid_fit_impl<HasValidXYFitImpl, Y>::value));
  EXPECT_FALSE(bool(has_valid_fit_impl<HasNoFitImpl, X>::value));
}

TEST(test_traits_core, test_has_possible_fit_impl) {
  EXPECT_TRUE(bool(has_possible_fit_impl<HasValidFitImpl, X>::value));
  EXPECT_TRUE(bool(has_possible_fit_impl<HasWrongReturnTypeFitImpl, X>::value));
  EXPECT_TRUE(bool(has_possible_fit_impl<HasNonConstFitImpl, X>::value));
  EXPECT_TRUE(bool(has_possible_fit_impl<HasNonConstArgsFitImpl, X>::value));
  EXPECT_FALSE(bool(has_possible_fit_impl<HasProtectedValidFitImpl, X>::value));
  EXPECT_FALSE(bool(has_possible_fit_impl<HasPrivateValidFitImpl, X>::value));
  EXPECT_TRUE(bool(has_possible_fit_impl<HasValidAndInvalidFitImpl, X>::value));
  EXPECT_TRUE(bool(has_possible_fit_impl<HasValidXYFitImpl, X>::value));
  EXPECT_TRUE(bool(has_possible_fit_impl<HasValidXYFitImpl, Y>::value));
  EXPECT_FALSE(bool(has_possible_fit_impl<HasNoFitImpl, X>::value));
}

//
// TEST(test_traits, test_has_valid_call_impl) {
//  EXPECT_TRUE(bool(has_valid_call_impl<HasPublicCallImpl, X, Y>::value));
//  EXPECT_FALSE(bool(has_valid_call_impl<HasPublicCallImpl, Y, X>::value));
//  EXPECT_TRUE(
//      bool(has_valid_call_impl<HasMultiplePublicCallImpl, X, X>::value));
//  EXPECT_TRUE(
//      bool(has_valid_call_impl<HasMultiplePublicCallImpl, Y, Y>::value));
//  EXPECT_FALSE(
//      bool(has_valid_call_impl<HasMultiplePublicCallImpl, Z, X>::value));
//  EXPECT_FALSE(
//      bool(has_valid_call_impl<HasMultiplePublicCallImpl, Z, Y>::value));
//  EXPECT_FALSE(
//      bool(has_valid_call_impl<HasMultiplePublicCallImpl, Z, Z>::value));
//}
//
///*
// * Here we test to make sure we can identify situations where
// * call_impl_ has been defined but not necessarily properly.
// */
// TEST(test_traits, test_has_possible_call_impl) {
//  EXPECT_TRUE(bool(has_possible_call_impl<HasPublicCallImpl, X, Y>::value));
//  EXPECT_FALSE(bool(has_possible_call_impl<HasPublicCallImpl, Y, X>::value));
//  EXPECT_TRUE(
//      bool(has_possible_call_impl<HasMultiplePublicCallImpl, X, X>::value));
//  EXPECT_TRUE(
//      bool(has_possible_call_impl<HasMultiplePublicCallImpl, Y, Y>::value));
//  EXPECT_TRUE(
//      bool(has_possible_call_impl<HasMultiplePublicCallImpl, Z, X>::value));
//  EXPECT_TRUE(
//      bool(has_possible_call_impl<HasMultiplePublicCallImpl, Z, Y>::value));
//  EXPECT_TRUE(
//      bool(has_possible_call_impl<HasMultiplePublicCallImpl, Z, Z>::value));
//}
//
// TEST(test_traits, test_has_invalid_call_impl) {
//  EXPECT_FALSE(bool(has_invalid_call_impl<HasPublicCallImpl, X, Y>::value));
//  EXPECT_FALSE(bool(has_invalid_call_impl<HasPublicCallImpl, Y, X>::value));
//  EXPECT_FALSE(
//      bool(has_invalid_call_impl<HasMultiplePublicCallImpl, X, X>::value));
//  EXPECT_FALSE(
//      bool(has_invalid_call_impl<HasMultiplePublicCallImpl, Y, Y>::value));
//  EXPECT_TRUE(
//      bool(has_invalid_call_impl<HasMultiplePublicCallImpl, Z, X>::value));
//  EXPECT_TRUE(
//      bool(has_invalid_call_impl<HasMultiplePublicCallImpl, Z, Y>::value));
//  EXPECT_TRUE(
//      bool(has_invalid_call_impl<HasMultiplePublicCallImpl, Z, Z>::value));
//}

} // namespace albatross
