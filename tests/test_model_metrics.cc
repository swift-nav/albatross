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
#include <albatross/GP>
#include <albatross/Tune>
#include <gtest/gtest.h>

namespace albatross {

template <typename TestMetric>
class ModelMetricTester : public ::testing::Test {
public:
  TestMetric test_metric;
};

/*
 * Add any new model metrics here:
 */
typedef ::testing::Types<LeaveOneOutLikelihood<JointDistribution>,
                         LeaveOneOutLikelihood<MarginalDistribution>,
                         LeaveOneOutRMSE, GaussianProcessLikelihood>
    MetricsToTest;

TYPED_TEST_CASE(ModelMetricTester, MetricsToTest);

TYPED_TEST(ModelMetricTester, test_sanity) {
  MakeGaussianProcess test_case;
  auto dataset = test_case.get_dataset();
  auto model = test_case.get_model();
  const auto metric = this->test_metric(dataset, model);
  EXPECT_FALSE(std::isnan(metric));
}

TEST(test_gp_from_prediction, test_model_from_prediction) {
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
      joint_prediction.mean, 1e-16));
  EXPECT_TRUE(joint_prediction_from_prediction.covariance.isApprox(
      joint_prediction.covariance, 1e-8));
}

} // namespace albatross
