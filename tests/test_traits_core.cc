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

template <typename T>
struct Adaptable;

template <typename T>
struct Fit<Adaptable<T>> {};

template <typename ImplType>
struct Adaptable : public ModelBase<Adaptable<ImplType>> {

  Fit<Adaptable<ImplType>> fit_impl_(const std::vector<X> &,
                                   const MarginalDistribution &) const {
    return Fit<Adaptable<ImplType>>();
  }

  /*
   * This forwards on `fit_impl_` definitions from the implementing class
   * to this class so it can be picked up by ModelBase.
   */
  template <typename FeatureType,
            typename std::enable_if<
                       has_valid_fit_impl<ImplType, FeatureType>::value,
                                   int>::type = 0>
  Fit<Adaptable<ImplType>> fit_impl_(const std::vector<FeatureType> &features,
                                   const MarginalDistribution &targets) const {
    return impl().fit_impl_(features, targets);
  }

  /*
   * CRTP Helpers
   */
  ImplType &impl() { return *static_cast<ImplType *>(this); }
  const ImplType &impl() const { return *static_cast<const ImplType *>(this); }

};

struct Extended : public Adaptable<Extended> {

  using Base = Adaptable<Extended>;

  auto fit_impl_(const std::vector<Y> &,
                 const MarginalDistribution &targets) const {
    std::vector<X> xs = {{}};
    return Base::fit_impl_(xs, targets);
  }

};


TEST(test_traits_core, test_adaptable_has_valid_fit_impl) {
  EXPECT_TRUE(bool(has_valid_fit<Extended, X>::value));
  EXPECT_TRUE(bool(has_valid_fit<Extended, Y>::value));
  EXPECT_TRUE(bool(is_valid_fit_type<Extended, Fit<Extended>>::value));
  EXPECT_TRUE(bool(is_valid_fit_type<Extended, Fit<Adaptable<Extended>>>::value));
}

/*
 * Predict Traits
 */
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
