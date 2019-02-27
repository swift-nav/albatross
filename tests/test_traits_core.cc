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

#include "Core"

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

TEST(test_traits_core, test_has_valid_fit_impl) {
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
  //  EXPECT_TRUE(bool(has_valid_fit_impl<HasInheritedFitImpl, Y>::value));
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

template <typename T>
struct Base {};

struct Derived : public Base<Derived> {};

TEST(test_traits_core, test_is_valid_fit_type) {
  EXPECT_TRUE(bool(is_valid_fit_type<Derived, Fit<Derived>>::value));
  EXPECT_TRUE(bool(is_valid_fit_type<Base<Derived>, Fit<Base<Derived>>>::value));
  // If a Derived class which inherits from Base<Derived> has a fit_impl_
  // which returns Fit<Base<Derived>> consider that a valid fit type.
  EXPECT_TRUE(bool(is_valid_fit_type<Derived, Fit<Base<Derived>>>::value));
  // However a fit_impl which returns
  EXPECT_FALSE(bool(is_valid_fit_type<Base<Derived>, Fit<Derived>>::value));
}

/*
 * Predict Traits
 */
// TEST(test_traits_core, test_is_valid_predict_type) {
//  EXPECT_TRUE(bool(is_valid_predict_type<Eigen::VectorXd>::value));
//  EXPECT_TRUE(bool(is_valid_predict_type<MarginalDistribution>::value));
//  EXPECT_TRUE(bool(is_valid_predict_type<JointDistribution>::value));
//  EXPECT_FALSE(bool(is_valid_predict_type<Eigen::MatrixXd>::value));
//  EXPECT_FALSE(bool(is_valid_predict_type<double>::value));
//}

class HasMeanPredictImpl {
public:
  Eigen::VectorXd predict_(const std::vector<X> &,
                           PredictTypeIdentity<Eigen::VectorXd> &&) const {
    return Eigen::VectorXd::Zero(0);
  }
};

class HasMarginalPredictImpl {
public:
  MarginalDistribution
  predict_(const std::vector<X> &,
           PredictTypeIdentity<MarginalDistribution> &&) const {
    const auto mean = Eigen::VectorXd::Zero(0);
    return MarginalDistribution(mean);
  }
};

class HasJointPredictImpl {
public:
  JointDistribution predict_(const std::vector<X> &,
                             PredictTypeIdentity<JointDistribution> &&) const {
    const auto mean = Eigen::VectorXd::Zero(0);
    return JointDistribution(mean);
  }
};

class HasAllPredictImpls {
public:
  Eigen::VectorXd predict_(const std::vector<X> &,
                           PredictTypeIdentity<Eigen::VectorXd> &&) const {
    return Eigen::VectorXd::Zero(0);
  }

  MarginalDistribution
  predict_(const std::vector<X> &,
           PredictTypeIdentity<MarginalDistribution> &&) const {
    const auto mean = Eigen::VectorXd::Zero(0);
    return MarginalDistribution(mean);
  }

  JointDistribution predict_(const std::vector<X> &,
                             PredictTypeIdentity<JointDistribution> &&) const {
    const auto mean = Eigen::VectorXd::Zero(0);
    return JointDistribution(mean);
  }
};

TEST(test_traits_core, test_has_valid_predict_impl) {
  EXPECT_TRUE(bool(has_valid_predict_mean<HasMeanPredictImpl, X>::value));
  EXPECT_TRUE(
      bool(has_valid_predict_marginal<HasMarginalPredictImpl, X>::value));
  EXPECT_TRUE(bool(has_valid_predict_joint<HasJointPredictImpl, X>::value));
  EXPECT_TRUE(bool(has_valid_predict_mean<HasAllPredictImpls, X>::value));
  EXPECT_TRUE(bool(has_valid_predict_marginal<HasAllPredictImpls, X>::value));
  EXPECT_TRUE(bool(has_valid_predict_joint<HasAllPredictImpls, X>::value));
}

} // namespace albatross
