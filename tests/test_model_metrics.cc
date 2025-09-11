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
#include <albatross/GP>
#include <albatross/Tune>
#include "test_models.h"

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
                         LeaveOneOutRMSE, GaussianProcessNegativeLogLikelihood>
    MetricsToTest;

TYPED_TEST_SUITE(ModelMetricTester, MetricsToTest);

TYPED_TEST(ModelMetricTester, test_sanity) {
  MakeGaussianProcess test_case;
  auto dataset = test_case.get_dataset();
  auto model = test_case.get_model();
  const auto metric = this->test_metric(dataset, model);
  EXPECT_FALSE(std::isnan(metric));
}

}  // namespace albatross
