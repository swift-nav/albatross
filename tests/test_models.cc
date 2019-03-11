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

#include <gtest/gtest.h>

#include "test_utils.h"

#include "GP"
#include "models/least_squares.h"

namespace albatross {

auto make_simple_covariance_function() {
  SquaredExponential<EuclideanDistance> squared_exponential(100., 100.);
  IndependentNoise<double> noise(0.1);
  return squared_exponential + noise;
}

class MakeGaussianProcess {
public:
  auto get_model() const {
    auto covariance = make_simple_covariance_function();
    return gp_from_covariance(covariance);
  }

  RegressionDataset<double> get_dataset() const {
    return make_toy_linear_data();
  }
};


template <typename CovarianceFunc>
class AdaptedGaussianProcess
    : public GaussianProcessBase<CovarianceFunc,
                                 AdaptedGaussianProcess<CovarianceFunc>> {
public:
  using Base = GaussianProcessBase<CovarianceFunc,
                                   AdaptedGaussianProcess<CovarianceFunc>>;

  template <typename FitFeatureType>
  using GPFitType = Fit<Base, FitFeatureType>;

  auto fit(const std::vector<AdaptedFeature> &features,
           const MarginalDistribution &targets) const {
    std::vector<double> converted;
    for (const auto &f : features) {
      converted.push_back(f.value);
    }
    return Base::fit(converted, targets);
  }

  template <typename FitFeatureType>
  JointDistribution predict(const std::vector<AdaptedFeature> &features,
                            const GPFitType<FitFeatureType> &gp_fit,
                            PredictTypeIdentity<JointDistribution> &&) const {
    std::vector<double> converted;
    for (const auto &f : features) {
      converted.push_back(f.value);
    }
    return Base::predict(converted, gp_fit,
                         PredictTypeIdentity<JointDistribution>());
  }
};

class MakeAdaptedGaussianProcess {
public:
  auto get_model() const {
    auto covariance = make_simple_covariance_function();
    AdaptedGaussianProcess<decltype(covariance)> gp;

    return gp;
  }

  RegressionDataset<AdaptedFeature> get_dataset() const {
    return make_adapted_toy_linear_data();
  }
};

class MakeLinearRegression {
public:
  LinearRegression get_model() const { return LinearRegression(); }

  RegressionDataset<double> get_dataset() const {
    return make_toy_linear_data();
  }
};

template <typename ModelTestCase>
class RegressionModelTester : public ::testing::Test {
public:
  ModelTestCase test_case;
};

typedef ::testing::Types<MakeLinearRegression, MakeGaussianProcess,
                         MakeAdaptedGaussianProcess>
    ModelCreators;
TYPED_TEST_CASE(RegressionModelTester, ModelCreators);

Eigen::Index silly_function_to_increment_stack_pointer() {
  Eigen::VectorXd x(10);
  return x.size();
}

TYPED_TEST(RegressionModelTester, performs_reasonably_on_linear_data) {
  auto dataset = this->test_case.get_dataset();
  auto model = this->test_case.get_model();

  const auto fit_model = model.get_fit_model(dataset.features, dataset.targets);
  const auto pred = fit_model.get_prediction(dataset.features);
  const auto pred_mean = pred.mean();

  double rmse = sqrt((pred_mean - dataset.targets.mean).norm());
  EXPECT_LE(rmse, 0.5);
}

TYPED_TEST(RegressionModelTester, test_predict_variants) {
  auto dataset = this->test_case.get_dataset();
  auto model = this->test_case.get_model();

  const auto fit_model = model.get_fit_model(dataset.features, dataset.targets);
  silly_function_to_increment_stack_pointer();
  const auto pred = fit_model.get_prediction(dataset.features);
  silly_function_to_increment_stack_pointer();

  const Eigen::VectorXd pred_mean = pred.mean();

  const MarginalDistribution marginal = pred.marginal();
  EXPECT_LE((pred_mean - marginal.mean).norm(), 1e-8);

  const JointDistribution joint = pred.joint();
  EXPECT_LE((pred_mean - joint.mean).norm(), 1e-8);

  if (joint.has_covariance()) {
    EXPECT_LE(
        (marginal.covariance.diagonal() - joint.covariance.diagonal()).norm(),
        1e-8);
  }
}

} // namespace albatross
