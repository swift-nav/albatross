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

  auto model_creator = toy_gaussian_process;

  TuningMetric<double> metric = loo_nll;
  std::ostringstream output_stream;
  TuneModelConfig<double> config(model_creator, dataset, metric,
                                 albatross::mean_aggregator, output_stream);
  config.optimizer.set_maxeval(20);
  auto params = tune_regression_model(config);
}

TEST(test_tune, test_with_prior_bounds) {
  // Here we create a situation where tuning should hit a few
  // invalid parameters which will result in a NAN objective
  // function and we want to make sure the tuning recovers.
  auto dataset = make_toy_linear_data();

  auto model_with_prior = [] {
    auto model_creator = toy_gaussian_process;
    auto model = model_creator();
    for (const auto &pair : model->get_params()) {
      Parameter param = {1.e-8, std::make_shared<PositivePrior>()};
      model->set_param(pair.first, param);
    }
    return model;
  };

  TuningMetric<double> metric = loo_nll;
  std::ostringstream output_stream;
  TuneModelConfig<double> config(model_with_prior, dataset, metric,
                                 albatross::mean_aggregator, output_stream);
  config.optimizer.set_maxeval(20);
  auto params = tune_regression_model(config);
  auto m = model_with_prior();
  m->set_params(params);
  EXPECT_TRUE(m->params_are_valid());
}

TEST(test_tune, test_with_prior) {
  // Here we create a situation where tuning should hit a few
  // invalid parameters which will result in a NAN objective
  // function and we want to make sure the tuning recovers.
  auto dataset = make_toy_linear_data();

  auto model_with_prior = [] {
    auto model_creator = toy_gaussian_process;
    auto model = model_creator();
    for (const auto &pair : model->get_params()) {
      model->set_prior(pair.first, std::make_shared<GaussianPrior>(
                                       pair.second.value + 0.1, 0.001));
    }
    auto param_names = map_keys(model->get_params());
    model->set_prior(param_names[0], std::make_shared<FixedPrior>());
    return model;
  };

  // Tune with a prior
  TuningMetric<double> metric = loo_nll;
  std::ostringstream output_stream;
  TuneModelConfig<double> config(model_with_prior, dataset, metric,
                                 albatross::mean_aggregator, output_stream);
  config.optimizer.set_maxeval(20);
  auto params = tune_regression_model(config);

  // Tune without a prior
  TuneModelConfig<double> config_no_prior(toy_gaussian_process, dataset, metric,
                                          albatross::mean_aggregator,
                                          output_stream);
  config_no_prior.optimizer.set_maxeval(20);
  auto params_no_prior = tune_regression_model(config_no_prior);

  // Make sure tuning to the prior results in parameters that are
  // more likely.
  auto m = model_with_prior();
  m->set_params(params);
  double ll_with_prior = m->prior_log_likelihood();

  for (const auto &pair : params_no_prior) {
    m->set_param(pair.first, pair.second.value);
  }
  EXPECT_GT(ll_with_prior, m->prior_log_likelihood());
}

TEST(test_tune, test_multiple_datasets) {
  auto one_dataset = make_toy_linear_data(2., 4., 0.2);
  auto another_dataset = make_toy_linear_data(1., 5., 0.1);
  std::vector<RegressionDataset<double>> datasets = {one_dataset,
                                                     another_dataset};
  auto model_creator = toy_gaussian_process;
  TuningMetric<double> metric = loo_nll;
  std::ostringstream output_stream;
  TuneModelConfig<double> config(model_creator, datasets, metric,
                                 albatross::mean_aggregator, output_stream);
  config.optimizer.set_maxeval(20);
  auto params = tune_regression_model(config);
}

} // namespace albatross
