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
                         TestMetric<gp_nll>>
    MetricsToTest;

TYPED_TEST_CASE(TuningMetricTester, MetricsToTest);

TYPED_TEST(TuningMetricTester, test_sanity) {
  const auto dataset = make_toy_linear_data(5., 1., 0.1, 4);
  ;
  const auto model_creator = toy_gaussian_process;
  const auto model = model_creator();
  const auto metric = this->test_metric.function(dataset, model.get());
  EXPECT_FALSE(std::isnan(metric));
}

} // namespace albatross
