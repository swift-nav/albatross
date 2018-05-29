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

TEST(test_tune, test_single_dataset) {
  auto dataset = make_toy_linear_data();

  auto model_creator = one_dimensional_gaussian_process;

  TuningMetric<double> metric = loo_nll;
  std::ostringstream output_stream;
  TuneModelConfg<double> config(model_creator, dataset, metric,
                                albatross::mean_aggregator, output_stream);
  auto params = tune_regression_model(config);
}

TEST(test_tune, test_multiple_datasets) {
  auto one_dataset = make_toy_linear_data(2., 4., 0.2);
  auto another_dataset = make_toy_linear_data(1., 5., 0.1);
  std::vector<RegressionDataset<double>> datasets = {one_dataset,
                                                     another_dataset};
  auto model_creator = one_dimensional_gaussian_process;
  TuningMetric<double> metric = loo_nll;
  std::ostringstream output_stream;
  TuneModelConfg<double> config(model_creator, datasets, metric,
                                albatross::mean_aggregator, output_stream);
  auto params = tune_regression_model(config);
}

} // namespace albatross
