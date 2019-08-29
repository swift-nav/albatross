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

namespace albatross {

template <typename CovFunc>
class TestAdaptedModel
    : public GaussianProcessBase<CovFunc, TestAdaptedModel<CovFunc>> {
public:
  using Base = GaussianProcessBase<CovFunc, TestAdaptedModel<CovFunc>>;

  template <typename FeatureType>
  using FitType = typename Base::template CholeskyFit<FeatureType>;

  TestAdaptedModel() {
    this->params_["center"] = {1., UniformPrior(-10., 10.)};
  }

  std::vector<double>
  convert(const std::vector<AdaptedFeature> &features) const {
    std::vector<double> converted;
    for (const auto &f : features) {
      converted.push_back(f.value - this->get_param_value("center"));
    }
    return converted;
  }

  auto _fit_impl(const std::vector<AdaptedFeature> &features,
                 const MarginalDistribution &targets) const {
    return Base::_fit_impl(convert(features), targets);
  }

  JointDistribution
  _predict_impl(const std::vector<AdaptedFeature> &features,
                const FitType<double> &fit_,
                PredictTypeIdentity<JointDistribution> &&) const {
    return Base::_predict_impl(convert(features), fit_,
                               PredictTypeIdentity<JointDistribution>());
  }
};

template <typename ModelType>
void test_get_set(ModelType &model, const std::string &key) {
  // Make sure a key exists, then modify it and make sure it
  // takes on the new value.
  const auto orig = model.get_param_value(key);
  model.set_param(key, orig + 1.);
  EXPECT_EQ(model.get_params().at(key), orig + 1.);
}

TEST(test_model_adapter, test_get_set_params) {
  // An adapted model should contain both higher level parameters,
  // and the sub model parameters.
  using SqrExp = SquaredExponential<EuclideanDistance>;
  TestAdaptedModel<SqrExp> model;
  auto sqr_exp_params = SqrExp().get_params();
  auto params = model.get_params();
  // Make sure all the sub model params are in the adapted params
  for (const auto &pair : sqr_exp_params) {
    test_get_set(model, pair.first);
  }
  // And the higher level parameter.
  test_get_set(model, "center");
};

TEST(test_model_adapter, test_fit) {
  const auto adpated_dataset = make_adapted_toy_linear_data();

  using CovFunc = decltype(toy_covariance_function());
  TestAdaptedModel<CovFunc> adapted_model;

  const auto adapted_fit_model = adapted_model.fit(adpated_dataset);
  const auto adapted_pred =
      adapted_fit_model.predict(adpated_dataset.features).joint();
  const auto adapted_fit = adapted_fit_model.get_fit();

  GaussianProcessRegression<CovFunc> model;

  auto dataset = make_toy_linear_data();
  const auto fit_model = model.fit(dataset);
  const auto pred = fit_model.predict(dataset.features).joint();
  const auto fit = fit_model.get_fit();

  EXPECT_EQ(adapted_pred, pred);

  // The train_features will actually be different because the adapted model
  // subtracts off a center value.
  EXPECT_EQ(adapted_fit.information, fit.information);
  EXPECT_EQ(adapted_fit.train_covariance, fit.train_covariance);
}
} // namespace albatross
