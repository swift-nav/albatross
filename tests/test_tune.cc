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
#include <nlopt.hpp>

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
    model_with_priors.set_param_value(pair.first, pair.second.value);
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

template <typename ObjectiveFunction>
Eigen::VectorXd nlopt_solve(GenericTuner &tuner, ObjectiveFunction &objective) {
  const auto output = tuner.tune(objective);
  auto x = get_tunable_parameters(output).values;
  const Eigen::Map<Eigen::VectorXd> eigen_output(&x[0],
                                                 cast::to_index(x.size()));
  return eigen_output;
}

class TestTuneQuadratic : public ::testing::Test {
public:
  TestTuneQuadratic() {
    Eigen::Index k = 3;
    A.resize(k, k);
    A << 4.5244, 1.43904, 2.24636, 1.43904, 2.26512, 0.985532, 2.24636,
        0.985532, 2.18973;
    truth = Eigen::VectorXd::Ones(k);
    b = A * truth;
  };

  double objective(const Eigen::VectorXd &x) const {
    const Eigen::VectorXd z = A * x - b;
    return z.dot(z);
  }

  double objective(const std::vector<double> &vector_x) const {
    std::vector<double> x(vector_x);
    const Eigen::Map<Eigen::VectorXd> eigen_x(&x[0], cast::to_index(x.size()));
    return objective(eigen_x);
  }

  double objective(const ParameterStore &params) const {
    return objective(get_tunable_parameters(params).values);
  }

  Eigen::VectorXd gradient(const std::vector<double> &vector_x) const {
    std::vector<double> x(vector_x);
    const Eigen::Map<Eigen::VectorXd> eigen_x(&x[0], cast::to_index(x.size()));
    return gradient(eigen_x);
  }

  Eigen::VectorXd gradient(const Eigen::VectorXd &x) const {
    return 2 * (A.transpose() * A * x - A.transpose() * b);
  }

  Eigen::MatrixXd A;
  Eigen::VectorXd truth;
  Eigen::VectorXd b;
};

TEST_F(TestTuneQuadratic, test_generic) {

  auto mahalanobis_distance_eigen = [&](const Eigen::VectorXd &eigen_x) {
    return this->objective(eigen_x);
  };

  auto mahalanobis_distance_vector = [&](const std::vector<double> &vector_x) {
    return this->objective(vector_x);
  };

  auto mahalanobis_distance_params = [&](const ParameterStore &params) {
    return this->objective(params);
  };

  std::ostringstream output_stream;
  std::vector<double> initial_x(cast::to_size(b.size()));
  for (auto &d : initial_x) {
    d = 0.;
  }

  auto test_tuner = [&](GenericTuner &tuner) {
    // Make sure the generic tuner can use any of the different objective
    // function signatures.
    const auto eigen_result = tuner.tune(mahalanobis_distance_eigen);
    EXPECT_LT((eigen_result - truth).array().abs().maxCoeff(), 5e-3);

    auto vector_result = tuner.tune(mahalanobis_distance_vector);
    const Eigen::Map<Eigen::VectorXd> eigen_vector_output(
        &vector_result[0], cast::to_index(vector_result.size()));
    EXPECT_LT((eigen_vector_output - truth).array().abs().maxCoeff(), 5e-3);

    const auto params_result = tuner.tune(mahalanobis_distance_params);
    auto param_vector = get_tunable_parameters(params_result).values;
    const Eigen::Map<Eigen::VectorXd> eigen_param_output(
        &param_vector[0], cast::to_index(param_vector.size()));

    EXPECT_LT((eigen_param_output - truth).array().abs().maxCoeff(), 5e-3);
  };

  GenericTuner default_tuner(initial_x, output_stream);
  test_tuner(default_tuner);

  const auto params = uninformative_params(initial_x);

  GenericTuner gradient_tuner(initial_x, output_stream);
  gradient_tuner.optimizer = default_gradient_optimizer(params);
  test_tuner(gradient_tuner);

  GenericTuner async_gradient_tuner(initial_x, output_stream);
  async_gradient_tuner.optimizer = default_gradient_optimizer(params);
  async_gradient_tuner.threads = make_shared_thread_pool();
  test_tuner(async_gradient_tuner);
}

TEST_F(TestTuneQuadratic, test_greedy_tune) {

  auto mahalanobis_distance_params = [&](const ParameterStore &params) {
    return this->objective(params);
  };

  std::ostringstream output_stream;
  std::vector<double> initial_x(cast::to_size(b.size()));
  for (auto &d : initial_x) {
    d = 0.;
  }

  auto test_tuner = [&]() {
    const std::size_t n_threads = 8;
    const std::size_t n_iterations = 1000;
    ParameterStore params;
    for (std::size_t i = 0; i < initial_x.size(); ++i) {
      params[std::to_string(i)] = {initial_x[i],
                                   albatross::UniformPrior(-2, 2)};
    }
    const auto params_result =
        greedy_tune(mahalanobis_distance_params, params, n_threads,
                    n_iterations, nullptr, &output_stream);

    auto to_eigen = [](const auto &p) {
      auto param_vector = get_tunable_parameters(p).values;
      const Eigen::VectorXd output = Eigen::Map<Eigen::VectorXd>(
          &param_vector[0], cast::to_index(param_vector.size()));
      return output;
    };

    EXPECT_GT((to_eigen(params) - truth).array().abs().maxCoeff(), 0.1);
    EXPECT_LE((to_eigen(params_result) - truth).array().abs().maxCoeff(), 0.1);
  };

  test_tuner();
}

TEST_F(TestTuneQuadratic, test_compute_gradient) {

  auto mahalanobis_distance_vector = [&](const std::vector<double> &vector_x) {
    return this->objective(vector_x);
  };

  auto mahalanobis_distance_params = [&](const ParameterStore &params) {
    return this->objective(params);
  };

  ParameterStore params;
  params["0"] = {0., albatross::UninformativePrior()};
  params["1"] = {1., albatross::PositiveGaussianPrior()};
  params["2"] = {2., albatross::UniformPrior(-10, 10)};

  std::vector<double> x = albatross::get_tunable_parameters(params).values;

  const double f_val = mahalanobis_distance_vector(x);
  const auto vector_grad =
      compute_gradient(mahalanobis_distance_vector, x, f_val);
  const auto param_grad =
      compute_gradient(mahalanobis_distance_params, params, f_val);
  const auto expected_grad = this->gradient(x);

  for (std::size_t i = 0; i < vector_grad.size(); ++i) {
    EXPECT_NEAR(vector_grad[i], expected_grad[cast::to_index(i)], 1e-4);
    EXPECT_NEAR(param_grad[i], expected_grad[cast::to_index(i)], 1e-4);
  }
}

TEST_F(TestTuneQuadratic, test_gradient_based_bounds) {

  auto mahalanobis_distance_params = [&](const ParameterStore &params) {
    return this->objective(params);
  };

  ParameterStore params;
  params["0"] = {0., albatross::UninformativePrior()};
  params["1"] = {1., albatross::PositiveGaussianPrior()};
  params["2"] = {2., albatross::UniformPrior(2, 10)};

  srand(2012);
  std::ostringstream output_stream;
  GenericTuner async_gradient_tuner(params, output_stream);
  async_gradient_tuner.optimizer = default_gradient_optimizer(params);

  const auto params_result =
      async_gradient_tuner.tune(mahalanobis_distance_params);
  auto param_vector = get_tunable_parameters(params_result).values;
  const Eigen::Map<Eigen::VectorXd> optimal(
      &param_vector[0], cast::to_index(param_vector.size()));

  Eigen::VectorXd active_constraint = Eigen::VectorXd::Zero(optimal.size());
  active_constraint[2] = 1.;

  const Eigen::VectorXd grad = this->gradient(optimal);

  // The optimal solution in the presence of a constraint should have a gradient
  // which is perpendicular to the constraint.  In otherwords, to decrease the
  // function value any further we'd need to violate the constraint.
  EXPECT_NEAR(grad.normalized().dot(active_constraint), 1, 1e-4);
}

} // namespace albatross
