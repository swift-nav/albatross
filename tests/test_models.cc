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

#include "test_models.h"

namespace albatross {

TYPED_TEST_P(RegressionModelTester, test_performs_reasonably_on_linear_data) {
  auto dataset = this->test_case.get_dataset();
  auto model = this->test_case.get_model();

  if (std::is_same<decltype(model), NullModel>::value) {
    return;
  }

  const auto fit_model = model.fit(dataset.features, dataset.targets);
  const auto pred = fit_model.predict(dataset.features);
  const auto pred_mean = pred.mean();

  double rmse = sqrt((pred_mean - dataset.targets.mean).norm());
  EXPECT_LE(rmse, 0.5);
}

Eigen::Index silly_function_to_increment_stack_pointer() {
  Eigen::VectorXd x(10);
  return x.size();
}

TYPED_TEST_P(RegressionModelTester, test_predict_variants) {
  auto dataset = this->test_case.get_dataset();
  auto model = this->test_case.get_model();

  const auto fit_model = model.fit(dataset.features, dataset.targets);
  silly_function_to_increment_stack_pointer();
  const auto pred = fit_model.predict(dataset.features);
  silly_function_to_increment_stack_pointer();

  expect_predict_variants_consistent(pred);
}

TYPED_TEST_P(RegressionModelTester, test_currect_derived_type) {
  const auto model = this->test_case.get_model();
  const auto derived = model.derived();
  bool same = std::is_same<decltype(model), decltype(derived)>::value;
  EXPECT_TRUE(same);
}

REGISTER_TYPED_TEST_CASE_P(RegressionModelTester,
                           test_performs_reasonably_on_linear_data,
                           test_predict_variants, test_currect_derived_type);

INSTANTIATE_TYPED_TEST_CASE_P(test_models, RegressionModelTester,
                              ExampleModels);

class BadModel : public ModelBase<BadModel> {
public:
  Fit<BadModel> _fit_impl(const std::vector<double> &,
                          const MarginalDistribution &) const {
    return {};
  }

  Eigen::VectorXd
  _predict_impl(const std::vector<double> &features, const Fit<BadModel> &,
                const PredictTypeIdentity<Eigen::VectorXd>) const {
    return Eigen::VectorXd::Ones(static_cast<Eigen::Index>(features.size()));
  }

  MarginalDistribution
  _predict_impl(const std::vector<double> &features, const Fit<BadModel> &,
                const PredictTypeIdentity<MarginalDistribution>) const {
    const auto zeros =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(features.size()));
    return MarginalDistribution(zeros, zeros.asDiagonal());
  }
};

TEST(test_models, test_expect_predict_variants_consistent_fails) {
  const auto dataset = make_toy_linear_data();
  BadModel m;
  const auto pred = m.fit(dataset).predict(dataset.features);
  expect_predict_variants_inconsistent(pred);
}

TEST(test_models, test_model_from_prediction) {
  MakeGaussianProcess test_case;
  auto dataset = test_case.get_dataset();
  auto model = test_case.get_model();
  std::vector<double> test_features = {0.1, 1.1, 2.2};
  auto joint_prediction = model.fit(dataset).predict(test_features).joint();
  auto joint_prediction_from_prediction =
      model.fit_from_prediction(test_features, joint_prediction)
          .predict(test_features)
          .joint();
  EXPECT_TRUE(joint_prediction_from_prediction.mean.isApprox(
      joint_prediction.mean, 1e-12));
  EXPECT_TRUE(joint_prediction_from_prediction.covariance.isApprox(
      joint_prediction.covariance, 1e-8));
}

} // namespace albatross
