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

#include "covariance_functions/covariance_functions.h"
#include "evaluate.h"
#include "models/gp.h"
#include "test_utils.h"
#include "tune.h"
#include <gtest/gtest.h>

namespace albatross {

TEST(test_tuning_metrics, test_fast_loo_equals_slow) {
  auto dataset = make_toy_linear_data();

  auto model_creator = toy_gaussian_process;
  auto model = model_creator();

  double fast_loo_nll = gp_fast_loo_nll(dataset, model.get());

  double slow_loo_nll = loo_nll(dataset, model.get());

  EXPECT_NEAR(fast_loo_nll, slow_loo_nll, 1e-6);
}

/*
 * Here we setup a typed test to make it easy to add new
 * tuning metrics and make sure they run fine.
 */

using DoubleTuningMetric = double(const RegressionDataset<double> &,
                                  RegressionModel<double> *);

template <DoubleTuningMetric Metric_> struct TestMetric {
  TuningMetric<double> function = Metric_;
};

template <typename TestMetric>
class TuningMetricTester : public ::testing::Test {
public:
  TestMetric test_metric;
};

/*
 * Add any new tuning metrics here:
 */
typedef ::testing::Types<TestMetric<loo_nll>, TestMetric<loo_rmse>,
                         TestMetric<gp_fast_loo_nll<double>>>
    MetricsToTest;

TYPED_TEST_CASE(TuningMetricTester, MetricsToTest);

TYPED_TEST(TuningMetricTester, test_sanity) {
  const auto dataset = make_toy_linear_data();
  const auto model_creator = toy_gaussian_process;
  const auto model = model_creator();
  const auto metric = this->test_metric.function(dataset, model.get());
  EXPECT_FALSE(std::isnan(metric));
}

TEST(test_tuning_metrics, test_fast_loo_works_on_adapted_model) {
  auto dataset = make_adapted_toy_linear_data();

  auto model_creator = adapted_toy_gaussian_process;
  auto model = model_creator();

  double fast_loo_nll =
      gp_fast_loo_nll<AdaptedFeature, double>(dataset, model.get());
}

} // namespace albatross
