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

#include <albatross/Core>
#include <gtest/gtest.h>

namespace albatross {

struct X {};
struct Y {};
struct Z {};

class HasValidFitImpl : public ModelBase<HasValidFitImpl> {
public:
  Fit<HasValidFitImpl> _fit_impl(const std::vector<X> &,
                                 const MarginalDistribution &) const {
    return {};
  };
};

class HasWrongReturnTypeFitImpl : public ModelBase<HasWrongReturnTypeFitImpl> {
public:
  double _fit_impl(const std::vector<X> &, const MarginalDistribution &) const {
    return {};
  };
};

class HasNonConstFitImpl : public ModelBase<HasNonConstFitImpl> {
public:
  Fit<HasNonConstFitImpl> _fit_impl(const std::vector<X> &,
                                    const MarginalDistribution &) {
    return {};
  };
};

class HasNonConstArgsFitImpl : public ModelBase<HasNonConstFitImpl> {
public:
  Fit<HasNonConstArgsFitImpl> _fit_impl(std::vector<X> &,
                                        const MarginalDistribution &) const {
    return {};
  };

  Fit<HasNonConstArgsFitImpl> _fit_impl(const std::vector<X> &,
                                        MarginalDistribution &) const {
    return {};
  };

  Fit<HasNonConstArgsFitImpl> _fit_impl(std::vector<X> &,
                                        MarginalDistribution &) const {
    return {};
  };
};

class HasProtectedValidFitImpl : public ModelBase<HasNonConstFitImpl> {
protected:
  Fit<HasProtectedValidFitImpl> _fit_impl(const std::vector<X> &,
                                          const MarginalDistribution &) const {
    return {};
  };
};

class HasPrivateValidFitImpl : public ModelBase<HasPrivateValidFitImpl> {
private:
  Fit<HasPrivateValidFitImpl> _fit_impl(const std::vector<X> &,
                                        const MarginalDistribution &) const {
    return {};
  };
};

class HasValidAndInvalidFitImpl : public ModelBase<HasValidAndInvalidFitImpl> {
public:
  Fit<HasValidAndInvalidFitImpl> _fit_impl(const std::vector<X> &,
                                           const MarginalDistribution &) const {
    return {};
  };

  Fit<HasValidAndInvalidFitImpl> _fit_impl(const std::vector<X> &,
                                           const MarginalDistribution &) {
    return {};
  };
};

class HasValidXYFitImpl : public ModelBase<HasValidXYFitImpl> {
public:
  Fit<HasValidXYFitImpl> _fit_impl(const std::vector<X> &,
                                   const MarginalDistribution &) const {
    return {};
  };

  Fit<HasValidXYFitImpl> _fit_impl(const std::vector<Y> &,
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
  EXPECT_TRUE(bool(std::is_same<Fit<HasValidFitImpl>,
                                fit_type<HasValidFitImpl, X>::type>::value));
  EXPECT_TRUE(bool(std::is_same<Fit<HasValidXYFitImpl>,
                                fit_type<HasValidXYFitImpl, X>::type>::value));
  EXPECT_TRUE(bool(std::is_same<Fit<HasValidXYFitImpl>,
                                fit_type<HasValidXYFitImpl, Y>::type>::value));
  EXPECT_TRUE(
      bool(std::is_same<Fit<HasValidAndInvalidFitImpl>,
                        fit_type<HasValidAndInvalidFitImpl, X>::type>::value));
}

TEST(test_traits_core, test_fit_model_type) {
  EXPECT_TRUE(
      bool(std::is_same<FitModel<HasValidXYFitImpl, Fit<HasValidXYFitImpl>>,
                        fit_model_type<HasValidXYFitImpl, X>::type>::value));
  EXPECT_FALSE(
      bool(std::is_same<FitModel<HasValidXYFitImpl, Fit<HasValidFitImpl>>,
                        fit_model_type<HasValidXYFitImpl, X>::type>::value));
}

template <typename T> struct Base : public ModelBase<T> {};

struct Derived : public Base<Derived> {};

TEST(test_traits_core, test_is_valid_fit_type) {
  EXPECT_TRUE(bool(is_valid_fit_type<Fit<Derived>>::value));
  EXPECT_TRUE(bool(is_valid_fit_type<Fit<Derived>>::value));

  EXPECT_TRUE(bool(is_valid_fit_type<Fit<Base<Derived>>>::value));
  EXPECT_TRUE(bool(is_valid_fit_type<Fit<Base<Derived>>>::value));

  EXPECT_FALSE(bool(is_valid_fit_type<Derived>::value));
  EXPECT_FALSE(bool(is_valid_fit_type<Base<Derived>>::value));
}

template <typename T, typename FeatureType> struct AdaptableFit;

template <typename T, typename FeatureType>
struct Fit<AdaptableFit<T, FeatureType>> {};

template <typename ImplType> struct Adaptable : public ModelBase<ImplType> {

  Fit<AdaptableFit<ImplType, X>> _fit_impl(const std::vector<X> &,
                                           const MarginalDistribution &) const {
    return Fit<AdaptableFit<ImplType, X>>();
  }
};

struct Extended : public Adaptable<Extended> {

  using Base = Adaptable<Extended>;
  using Base::_fit_impl;

  auto _fit_impl(const std::vector<Y> &,
                 const MarginalDistribution &targets) const {
    std::vector<X> xs = {{}};
    return Base::_fit_impl(xs, targets);
  }

  Z _predict_impl(const std::vector<Y> &,
                  const Fit<AdaptableFit<Extended, X>> &,
                  PredictTypeIdentity<Z>) const {
    return {};
  }
};

struct OtherExtended : public Adaptable<OtherExtended> {};

TEST(test_traits_core, test_adaptable_has_valid_fit) {
  EXPECT_TRUE(bool(has_valid_fit<Extended, X>::value));
  EXPECT_TRUE(bool(has_valid_fit<Extended, Y>::value));

  EXPECT_TRUE(bool(has_valid_fit<OtherExtended, X>::value));
  EXPECT_FALSE(bool(has_valid_fit<OtherExtended, Y>::value));
}

/*
 * Predict Traits
 */
class HasMeanPredictImpl {
public:
  Eigen::VectorXd _predict_impl(const std::vector<X> &,
                                const Fit<HasMeanPredictImpl> &,
                                PredictTypeIdentity<Eigen::VectorXd>) const {
    return Eigen::VectorXd::Zero(0);
  }
};

class HasMarginalPredictImpl {
public:
  MarginalDistribution
  _predict_impl(const std::vector<X> &, const Fit<HasMarginalPredictImpl> &,
                PredictTypeIdentity<MarginalDistribution>) const {
    const auto mean = Eigen::VectorXd::Zero(0);
    return MarginalDistribution(mean);
  }
};

class HasJointPredictImpl {
public:
  JointDistribution
  _predict_impl(const std::vector<X> &, const Fit<HasJointPredictImpl> &,
                PredictTypeIdentity<JointDistribution>) const {
    const auto mean = Eigen::VectorXd::Zero(0);
    const auto cov = Eigen::MatrixXd::Zero(0, 0);
    return JointDistribution(mean, cov);
  }
};

class HasAllPredictImpls {
public:
  Eigen::VectorXd _predict_impl(const std::vector<X> &,
                                const Fit<HasAllPredictImpls> &,
                                PredictTypeIdentity<Eigen::VectorXd>) const {
    return Eigen::VectorXd::Zero(0);
  }

  MarginalDistribution
  _predict_impl(const std::vector<X> &, const Fit<HasAllPredictImpls> &,
                PredictTypeIdentity<MarginalDistribution>) const {
    const auto mean = Eigen::VectorXd::Zero(0);
    return MarginalDistribution(mean);
  }

  JointDistribution
  _predict_impl(const std::vector<X> &, const Fit<HasAllPredictImpls> &,
                PredictTypeIdentity<JointDistribution>) const {
    const auto mean = Eigen::VectorXd::Zero(0);
    const auto cov = Eigen::MatrixXd::Zero(0, 0);
    return JointDistribution(mean, cov);
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

TEST(test_traits_core, test_has_name) {
  EXPECT_TRUE(bool(has_name<HasName>::value));
  EXPECT_FALSE(bool(has_name<HasNoName>::value));
}

TEST(test_traits_core, test_eigen_plain_objects) {
  EXPECT_TRUE(bool(is_eigen_plain_object<Eigen::MatrixXd>::value));
  EXPECT_FALSE(bool(is_eigen_xpr<Eigen::MatrixXd>::value));

  EXPECT_TRUE(bool(is_eigen_plain_object<Eigen::VectorXd>::value));
  EXPECT_FALSE(bool(is_eigen_xpr<Eigen::VectorXd>::value));

  EXPECT_TRUE(bool(is_eigen_plain_object<Eigen::Matrix<int, 3, 4>>::value));
  EXPECT_FALSE(bool(is_eigen_xpr<Eigen::Matrix<int, 3, 4>>::value));
}

TEST(test_traits_core, test_eigen_expressions) {
  Eigen::MatrixXd a;
  Eigen::MatrixXd b;

  using ProductType = decltype(a * b);
  EXPECT_FALSE(bool(is_eigen_plain_object<ProductType>::value));
  EXPECT_TRUE(bool(is_eigen_xpr<ProductType>::value));

  using LDLTType = Eigen::LDLT<Eigen::MatrixXd>;
  EXPECT_FALSE(bool(is_eigen_plain_object<LDLTType>::value));
  EXPECT_FALSE(bool(is_eigen_xpr<LDLTType>::value));
}

} // namespace albatross
