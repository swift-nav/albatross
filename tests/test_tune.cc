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
#include <albatross/Tune>
#include <gtest/gtest.h>

namespace albatross {

TEST(test_tune, test_single_dataset) {
  const MakeGaussianProcess test_case;
  auto dataset = test_case.get_dataset();
  auto model = test_case.get_model();

  LeaveOneOutLikelihood<> loo_nll;
  std::ostringstream output_stream;
  auto tuner =
      get_tuner(model, loo_nll, dataset, mean_aggregator, output_stream);
  tuner.optimizer.set_maxeval(20);
  auto params = tuner.tune();

  NegativeLogLikelihood<JointDistribution> nll;
  LeaveOneOutGrouper loo;
  const auto scores = model.cross_validate().scores(nll, dataset, loo);

  model.set_params(params);
  const auto scores_post_tuning =
      model.cross_validate().scores(nll, dataset, loo);

  EXPECT_LT(scores_post_tuning.mean(), scores.mean());
}

TEST(test_tune, test_with_prior_bounds) {
  // Here we create a situation where tuning should hit a few
  // invalid parameters which will result in a NAN objective
  // function and we want to make sure the tuning recovers.
  const MakeGaussianProcess test_case;
  auto dataset = test_case.get_dataset();
  auto model = test_case.get_model();

  for (const auto &pair : model.get_params()) {
    Parameter param = {1.e-8, PositivePrior()};
    model.set_param(pair.first, param);
  }

  LeaveOneOutLikelihood<> loo_nll;
  std::ostringstream output_stream;
  auto tuner =
      get_tuner(model, loo_nll, dataset, mean_aggregator, output_stream);
  tuner.optimizer.set_maxeval(20);
  auto params = tuner.tune();

  model.set_params(params);
  EXPECT_TRUE(model.params_are_valid());
}

TEST(test_tune, test_with_prior) {
  const MakeGaussianProcess test_case;
  auto dataset = test_case.get_dataset();
  auto model_no_priors = test_case.get_model();

  // Add priors to the parameters
  auto model_with_priors = test_case.get_model();
  for (const auto &pair : model_with_priors.get_params()) {
    model_with_priors.set_prior(pair.first,
                                GaussianPrior(pair.second.value + 0.1, 0.001));
  }
  auto param_names = map_keys(model_with_priors.get_params());
  model_with_priors.set_prior(param_names[0], FixedPrior());

  // Tune using likelihood which will include the parameter priors
  LeaveOneOutLikelihood<> loo_nll;
  std::ostringstream output_stream;
  auto tuner = get_tuner(model_with_priors, loo_nll, dataset, mean_aggregator,
                         output_stream);
  tuner.optimizer.set_maxeval(50);
  auto params = tuner.tune();
  model_with_priors.set_params(params);
  double ll_with_prior = model_with_priors.prior_log_likelihood();

  // Then tune without
  auto tuner_no_priors = get_tuner(model_no_priors, loo_nll, dataset,
                                   mean_aggregator, output_stream);
  tuner_no_priors.optimizer.set_maxeval(50);
  auto params_no_prior = tuner_no_priors.tune();

  // Set the model with priors to have the parameters from tuning
  // without.  These parameters should be inconsistent with the
  // prior which should lead to a smaller likelihood.
  for (const auto &pair : params_no_prior) {
    model_with_priors.set_param(pair.first, pair.second.value);
  }
  double ll_without_prior = model_with_priors.prior_log_likelihood();

  EXPECT_GT(ll_with_prior, ll_without_prior);
}

TEST(test_tune, test_multiple_datasets) {
  const MakeGaussianProcess test_case;
  auto model_no_priors = test_case.get_model();

  auto one_dataset = make_toy_linear_data(2., 4., 0.2);
  auto another_dataset = make_toy_linear_data(1., 5., 0.1);
  std::vector<RegressionDataset<double>> datasets = {one_dataset,
                                                     another_dataset};

  LeaveOneOutLikelihood<> loo_nll;
  std::ostringstream output_stream;
  auto tuner = get_tuner(model_no_priors, loo_nll, datasets, mean_aggregator,
                         output_stream);
  tuner.optimizer.set_maxeval(20);
  auto params = tuner.tune();
}

} // namespace albatross
