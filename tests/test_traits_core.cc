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

TEST(test_traits_core, test_is_vector) {
  EXPECT_TRUE(bool(is_vector<std::vector<double>>::value));
  EXPECT_FALSE(bool(is_vector<double>::value));
}

struct X {};
struct Y {};
struct Z {};

class HasValidFitImpl : public ModelBase<HasValidFitImpl> {
public:
  Fit<HasValidFitImpl, X> fit(const std::vector<X> &,
                              const MarginalDistribution &) const {
    return {};
  };
};

class HasWrongReturnTypeFitImpl : public ModelBase<HasWrongReturnTypeFitImpl> {
public:
  Fit<X, X> fit(const std::vector<X> &, const MarginalDistribution &) const {
    return {};
  };
};

class HasNonConstFitImpl : public ModelBase<HasNonConstFitImpl> {
public:
  Fit<HasNonConstFitImpl, X> fit(const std::vector<X> &,
                                 const MarginalDistribution &) {
    return {};
  };
};

class HasNonConstArgsFitImpl : public ModelBase<HasNonConstFitImpl> {
public:
  Fit<HasNonConstArgsFitImpl, X> fit(std::vector<X> &,
                                     const MarginalDistribution &) const {
    return {};
  };

  Fit<HasNonConstArgsFitImpl, X> fit(const std::vector<X> &,
                                     MarginalDistribution &) const {
    return {};
  };

  Fit<HasNonConstArgsFitImpl, X> fit(std::vector<X> &,
                                     MarginalDistribution &) const {
    return {};
  };
};

class HasProtectedValidFitImpl : public ModelBase<HasNonConstFitImpl> {
protected:
  Fit<HasProtectedValidFitImpl, X> fit(const std::vector<X> &,
                                       const MarginalDistribution &) const {
    return {};
  };
};

class HasPrivateValidFitImpl : public ModelBase<HasPrivateValidFitImpl> {
private:
  Fit<HasPrivateValidFitImpl, X> fit(const std::vector<X> &,
                                     const MarginalDistribution &) const {
    return {};
  };
};

class HasValidAndInvalidFitImpl : public ModelBase<HasValidAndInvalidFitImpl> {
public:
  Fit<HasValidAndInvalidFitImpl, X> fit(const std::vector<X> &,
                                        const MarginalDistribution &) const {
    return {};
  };

  Fit<HasValidAndInvalidFitImpl, X> fit(const std::vector<X> &,
                                        const MarginalDistribution &) {
    return {};
  };
};

class HasValidXYFitImpl : public ModelBase<HasValidXYFitImpl> {
public:
  Fit<HasValidXYFitImpl, X> fit(const std::vector<X> &,
                                const MarginalDistribution &) const {
    return {};
  };

  Fit<HasValidXYFitImpl, Y> fit(const std::vector<Y> &,
                                const MarginalDistribution &) const {
    return {};
  };
};

class HasNoFitImpl {};

TEST(test_traits_core, test_has_valid_fit) {
  EXPECT_TRUE(bool(has_valid_fit<HasValidFitImpl, X>::value));
  EXPECT_FALSE(bool(has_valid_fit<HasWrongReturnTypeFitImpl, X>::value));
  EXPECT_FALSE(bool(has_valid_fit<HasNonConstFitImpl, X>::value));
  EXPECT_FALSE(bool(has_valid_fit<HasNonConstArgsFitImpl, X>::value));
  EXPECT_FALSE(bool(has_valid_fit<HasProtectedValidFitImpl, X>::value));
  EXPECT_FALSE(bool(has_valid_fit<HasPrivateValidFitImpl, X>::value));
  EXPECT_TRUE(bool(has_valid_fit<HasValidAndInvalidFitImpl, X>::value));
  EXPECT_TRUE(bool(has_valid_fit<HasValidXYFitImpl, X>::value));
  EXPECT_TRUE(bool(has_valid_fit<HasValidXYFitImpl, Y>::value));
  EXPECT_FALSE(bool(has_valid_fit<HasNoFitImpl, X>::value));
  //  EXPECT_TRUE(bool(has_valid_fit<HasInheritedFitImpl, Y>::value));
}

TEST(test_traits_core, test_has_possible_fit_impl) {
  EXPECT_TRUE(bool(has_possible_fit<HasValidFitImpl, X>::value));
  EXPECT_TRUE(bool(has_possible_fit<HasWrongReturnTypeFitImpl, X>::value));
  EXPECT_TRUE(bool(has_possible_fit<HasNonConstFitImpl, X>::value));
  EXPECT_TRUE(bool(has_possible_fit<HasNonConstArgsFitImpl, X>::value));
  EXPECT_FALSE(bool(has_possible_fit<HasProtectedValidFitImpl, X>::value));
  EXPECT_FALSE(bool(has_possible_fit<HasPrivateValidFitImpl, X>::value));
  EXPECT_TRUE(bool(has_possible_fit<HasValidAndInvalidFitImpl, X>::value));
  EXPECT_TRUE(bool(has_possible_fit<HasValidXYFitImpl, X>::value));
  EXPECT_TRUE(bool(has_possible_fit<HasValidXYFitImpl, Y>::value));
  EXPECT_FALSE(bool(has_possible_fit<HasNoFitImpl, X>::value));
}

TEST(test_traits_core, test_fit_type) {
  EXPECT_TRUE(bool(std::is_same<Z, fit_type<FitModel<X, Z>, Y>::type>::value));
  EXPECT_TRUE(bool(std::is_same<Fit<HasValidFitImpl, X>,
                                fit_type<HasValidFitImpl, X>::type>::value));
  EXPECT_TRUE(bool(std::is_same<Fit<HasValidXYFitImpl, X>,
                                fit_type<HasValidXYFitImpl, X>::type>::value));
  EXPECT_TRUE(bool(std::is_same<Fit<HasValidXYFitImpl, Y>,
                                fit_type<HasValidXYFitImpl, Y>::type>::value));
  EXPECT_TRUE(
      bool(std::is_same<Fit<HasValidAndInvalidFitImpl, X>,
                        fit_type<HasValidAndInvalidFitImpl, X>::type>::value));
  //  EXPECT_FALSE(
  //      bool(std::is_same<Fit<HasPrivateValidFitImpl, X>,
  //                        fit_type<HasPrivateValidFitImpl, X>::type>::value));
}

template <typename T> struct Base {};

struct Derived : public Base<Derived> {};

TEST(test_traits_core, test_is_valid_fit_type) {
  EXPECT_TRUE(bool(is_valid_fit_type<Derived, Fit<Derived, X>>::value));
  EXPECT_TRUE(bool(is_valid_fit_type<Derived, Fit<Derived, Y>>::value));
  EXPECT_TRUE(
      bool(is_valid_fit_type<Base<Derived>, Fit<Base<Derived>, X>>::value));
  EXPECT_TRUE(
      bool(is_valid_fit_type<Base<Derived>, Fit<Base<Derived>, Y>>::value));
  // If a Derived class which inherits from Base<Derived> has a fit
  // which returns Fit<Base<Derived>> consider that a valid fit type.
  EXPECT_TRUE(bool(is_valid_fit_type<Derived, Fit<Base<Derived>, X>>::value));
  EXPECT_TRUE(bool(is_valid_fit_type<Derived, Fit<Base<Derived>, Y>>::value));
  // However a fit_impl which returns
  EXPECT_FALSE(bool(is_valid_fit_type<Base<Derived>, Fit<Derived, X>>::value));
  EXPECT_FALSE(bool(is_valid_fit_type<Base<Derived>, Fit<Derived, Y>>::value));
}

template <typename T> struct Adaptable;

template <typename T, typename FeatureType>
struct Fit<Adaptable<T>, FeatureType> {};

template <typename ImplType>
struct Adaptable : public ModelBase<Adaptable<ImplType>> {

  Fit<Adaptable<ImplType>, X> fit(const std::vector<X> &,
                                  const MarginalDistribution &) const {
    return Fit<Adaptable<ImplType>, X>();
  }

  /*
   * This forwards on `fit` definitions from the implementing class
   * to this class so it can be picked up by ModelBase.
   */
  template <typename FeatureType,
            typename std::enable_if<has_valid_fit<ImplType, FeatureType>::value,
                                    int>::type = 0>
  auto fit(const std::vector<FeatureType> &features,
           const MarginalDistribution &targets) const {
    return impl().fit(features, targets);
  }

  /*
   * CRTP Helpers
   */
  ImplType &impl() { return *static_cast<ImplType *>(this); }
  const ImplType &impl() const { return *static_cast<const ImplType *>(this); }
};

struct Extended : public Adaptable<Extended> {

  using Base = Adaptable<Extended>;

  auto fit(const std::vector<Y> &, const MarginalDistribution &targets) const {
    std::vector<X> xs = {{}};
    return Base::fit(xs, targets);
  }

  Z predict(const std::vector<Y> &, const Fit<Adaptable<Extended>, X> &,
            PredictTypeIdentity<Z>) const {
    return {};
  }
};

struct OtherExtended : public Adaptable<OtherExtended> {};

TEST(test_traits_core, test_adaptable_fit_type) {
  EXPECT_TRUE(bool(std::is_base_of<Fit<Adaptable<Extended>, X>,
                                   fit_type<Extended, Y>::type>::value));
  EXPECT_TRUE(bool(std::is_base_of<Fit<Adaptable<Extended>, X>,
                                   fit_type<Extended, X>::type>::value));
  EXPECT_TRUE(bool(
      has_valid_predict<Extended, Y, Fit<Adaptable<Extended>, X>, Z>::value));
}

TEST(test_traits_core, test_adaptable_has_valid_fit) {
  EXPECT_FALSE(bool(has_valid_fit<Extended, X>::value));
  EXPECT_TRUE(bool(has_valid_fit<Extended, Y>::value));
  EXPECT_TRUE(bool(is_valid_fit_type<Extended, Fit<Extended, X>>::value));
  EXPECT_TRUE(bool(is_valid_fit_type<Extended, Fit<Extended, Y>>::value));
  EXPECT_TRUE(
      bool(is_valid_fit_type<Extended, Fit<Adaptable<Extended>, X>>::value));
  EXPECT_TRUE(
      bool(is_valid_fit_type<Extended, Fit<Adaptable<Extended>, Y>>::value));
  EXPECT_FALSE(bool(
      is_valid_fit_type<OtherExtended, Fit<Adaptable<Extended>, Y>>::value));
  EXPECT_FALSE(bool(
      is_valid_fit_type<Extended, Fit<Adaptable<OtherExtended>, Y>>::value));
}

/*
 * Predict Traits
 */
class HasMeanPredictImpl {
public:
  Eigen::VectorXd predict(const std::vector<X> &,
                          const Fit<HasMeanPredictImpl> &,
                          PredictTypeIdentity<Eigen::VectorXd>) const {
    return Eigen::VectorXd::Zero(0);
  }
};

class HasMarginalPredictImpl {
public:
  MarginalDistribution
  predict(const std::vector<X> &, const Fit<HasMarginalPredictImpl> &,
          PredictTypeIdentity<MarginalDistribution>) const {
    const auto mean = Eigen::VectorXd::Zero(0);
    return MarginalDistribution(mean);
  }
};

class HasJointPredictImpl {
public:
  JointDistribution predict(const std::vector<X> &,
                            const Fit<HasJointPredictImpl> &,
                            PredictTypeIdentity<JointDistribution>) const {
    const auto mean = Eigen::VectorXd::Zero(0);
    return JointDistribution(mean);
  }
};

class HasAllPredictImpls {
public:
  Eigen::VectorXd predict(const std::vector<X> &,
                          const Fit<HasAllPredictImpls> &,
                          PredictTypeIdentity<Eigen::VectorXd>) const {
    return Eigen::VectorXd::Zero(0);
  }

  MarginalDistribution
  predict(const std::vector<X> &, const Fit<HasAllPredictImpls> &,
          PredictTypeIdentity<MarginalDistribution>) const {
    const auto mean = Eigen::VectorXd::Zero(0);
    return MarginalDistribution(mean);
  }

  JointDistribution predict(const std::vector<X> &,
                            const Fit<HasAllPredictImpls> &,
                            PredictTypeIdentity<JointDistribution>) const {
    const auto mean = Eigen::VectorXd::Zero(0);
    return JointDistribution(mean);
  }
};

TEST(test_traits_core, test_has_valid_predict_impl) {

  EXPECT_TRUE(bool(has_valid_predict_mean<HasMeanPredictImpl, X,
                                          Fit<HasMeanPredictImpl>>::value));
  EXPECT_FALSE(
      bool(has_valid_predict_marginal<HasMeanPredictImpl, X,
                                      Fit<HasMeanPredictImpl>>::value));
  EXPECT_FALSE(bool(has_valid_predict_joint<HasMeanPredictImpl, X,
                                            Fit<HasMeanPredictImpl>>::value));

  EXPECT_TRUE(
      bool(has_valid_predict_marginal<HasMarginalPredictImpl, X,
                                      Fit<HasMarginalPredictImpl>>::value));
  EXPECT_FALSE(
      bool(has_valid_predict_mean<HasMarginalPredictImpl, X,
                                  Fit<HasMarginalPredictImpl>>::value));
  EXPECT_FALSE(
      bool(has_valid_predict_joint<HasMarginalPredictImpl, X,
                                   Fit<HasMarginalPredictImpl>>::value));

  EXPECT_TRUE(bool(has_valid_predict_joint<HasJointPredictImpl, X,
                                           Fit<HasJointPredictImpl>>::value));
  EXPECT_FALSE(bool(has_valid_predict_mean<HasJointPredictImpl, X,
                                           Fit<HasJointPredictImpl>>::value));
  EXPECT_FALSE(
      bool(has_valid_predict_marginal<HasJointPredictImpl, X,
                                      Fit<HasJointPredictImpl>>::value));

  EXPECT_TRUE(bool(has_valid_predict_mean<HasAllPredictImpls, X,
                                          Fit<HasAllPredictImpls>>::value));
  EXPECT_TRUE(bool(has_valid_predict_marginal<HasAllPredictImpls, X,
                                              Fit<HasAllPredictImpls>>::value));
  EXPECT_TRUE(bool(has_valid_predict_joint<HasAllPredictImpls, X,
                                           Fit<HasAllPredictImpls>>::value));
}

class HasName {
public:
  std::string name() const { return "name"; };
};

class HasNoName {};

TEST(test_traits_covariance, test_has_name) {
  EXPECT_TRUE(bool(has_name<HasName>::value));
  EXPECT_FALSE(bool(has_name<HasNoName>::value));
}

} // namespace albatross
