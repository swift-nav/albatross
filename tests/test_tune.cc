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

double loo_nll(const albatross::RegressionDataset<double> &dataset,
               albatross::RegressionModel<double> *model) {
  auto loo_folds = albatross::leave_one_out(dataset);
  return albatross::cross_validated_scores(albatross::negative_log_likelihood,
                                           loo_folds, model)
      .mean();
}

std::unique_ptr<RegressionModel<double>> create_model() {
  using SqrExp = SquaredExponential<ScalarDistance>;
  using Noise = IndependentNoise<double>;
  CovarianceFunction<SqrExp> squared_exponential = {SqrExp(100., 100.)};
  CovarianceFunction<Noise> noise = {Noise(0.1)};
  auto covariance = squared_exponential + noise;
  return gp_pointer_from_covariance<double>(covariance);
}

TEST(test_tune, test_single_dataset) {
  auto dataset = make_toy_linear_data();

  auto model_creator = create_model;

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
  auto model_creator = create_model;
  TuningMetric<double> metric = loo_nll;
  std::ostringstream output_stream;
  TuneModelConfg<double> config(model_creator, datasets, metric,
                                albatross::mean_aggregator, output_stream);
  std::cout << "output" << std::endl;
  std::cout << output_stream.str() << std::endl;
  auto params = tune_regression_model(config);
}
} // namespace albatross
