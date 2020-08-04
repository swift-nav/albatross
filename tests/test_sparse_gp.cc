/*
 * Copyright (C) 2019 Swift Navigation Inc.
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

#include "test_models.h"
#include <chrono>

#include <albatross/SparseGP>

namespace albatross {

inline long int get_group(const double &f) {
  return static_cast<double>(floor(f / 5.));
}

struct LeaveOneIntervalOut {

  long int operator()(const double &f) const { return get_group(f); }
};

template <typename GrouperFunction>
class SparseGaussianProcessTest : public ::testing::Test {
public:
  GrouperFunction grouper;
};

typedef ::testing::Types<LeaveOneIntervalOut> IndependenceAssumptions;
TYPED_TEST_CASE(SparseGaussianProcessTest, IndependenceAssumptions);

template <typename CovFunc, typename GrouperFunction>
void expect_sparse_gp_performance(const CovFunc &covariance,
                                  const GrouperFunction &grouper,
                                  double sparse_error_threshold,
                                  double really_sparse_error_threshold) {

  auto dataset = make_toy_linear_data();
  auto direct = gp_from_covariance(covariance, "direct");

  UniformlySpacedInducingPoints strategy(8);
  auto sparse =
      sparse_gp_from_covariance(covariance, grouper, strategy, "sparse");
  sparse.set_param(details::inducing_nugget_name(), 1e-3);
  sparse.set_param(details::measurement_nugget_name(), 1e-12);

  UniformlySpacedInducingPoints bad_strategy(3);
  auto really_sparse = sparse_gp_from_covariance(covariance, grouper,
                                                 bad_strategy, "really_sparse");
  really_sparse.set_param(details::inducing_nugget_name(), 1e-3);
  really_sparse.set_param(details::measurement_nugget_name(), 1e-12);

  auto state_space =
      sparse_gp_from_covariance(covariance, grouper, "state_space");
  state_space.set_param(details::inducing_nugget_name(), 1e-3);
  state_space.set_param(details::measurement_nugget_name(), 1e-12);

  auto direct_fit = direct.fit(dataset);
  auto sparse_fit = sparse.fit(dataset);
  auto really_sparse_fit = really_sparse.fit(dataset);
  auto state_space_fit = state_space.fit(dataset);

  auto test_features = linspace(0.01, 9.9, 11);

  auto direct_pred =
      direct_fit.predict_with_measurement_noise(test_features).joint();
  auto sparse_pred =
      sparse_fit.predict_with_measurement_noise(test_features).joint();
  auto really_sparse_pred =
      really_sparse_fit.predict_with_measurement_noise(test_features).joint();
  auto state_space_pred =
      state_space_fit.predict_with_measurement_noise(test_features).joint();

  double sparse_error = (sparse_pred.mean - direct_pred.mean).norm();
  double really_sparse_error =
      (really_sparse_pred.mean - direct_pred.mean).norm();
  double state_space_error = (state_space_pred.mean - direct_pred.mean).norm();
  EXPECT_LT(sparse_error, sparse_error_threshold);
  EXPECT_LT(really_sparse_error, really_sparse_error_threshold);
  EXPECT_GT(really_sparse_error, sparse_error);
  EXPECT_GE(really_sparse_error, state_space_error);

  double sparse_cov_diff =
      (sparse_pred.covariance - direct_pred.covariance).norm();
  double really_sparse_cov_diff =
      (really_sparse_pred.covariance - direct_pred.covariance).norm();
  double state_sparse_cov_diff =
      (state_space_pred.covariance - direct_pred.covariance).norm();

  EXPECT_LT(sparse_cov_diff, sparse_error_threshold);
  EXPECT_LT(really_sparse_cov_diff, really_sparse_error_threshold);
  EXPECT_GT(really_sparse_cov_diff, sparse_cov_diff);
  EXPECT_GE(really_sparse_cov_diff, state_sparse_cov_diff);
}

TYPED_TEST(SparseGaussianProcessTest, test_sanity) {

  auto grouper = this->grouper;
  auto covariance = make_simple_covariance_function();

  // When the length scale is large the model with more inducing points
  // gets very nearly singular.  this checks to make sure that's dealt with
  // gracefully.
  covariance.set_param("squared_exponential_length_scale", 1000.);
  expect_sparse_gp_performance(covariance, grouper, 1e-2, 0.5);

  covariance.set_param("squared_exponential_length_scale", 100.);
  expect_sparse_gp_performance(covariance, grouper, 1e-2, 0.5);

  // Then when the length scale is shorter, the really sparse model
  // should become significantly worse than the sparse one.
  covariance.set_param("squared_exponential_length_scale", 10.);
  expect_sparse_gp_performance(covariance, grouper, 5e-2, 100.);
}

TYPED_TEST(SparseGaussianProcessTest, test_scales) {

  auto grouper = this->grouper;

  auto large_dataset = make_toy_sine_data(5., 10., 0.1, 1000);

  auto covariance = make_simple_covariance_function();

  auto direct = gp_from_covariance(covariance, "direct");
  // We need to make sure the priors on the parameters don't enter
  // into the log_likelihood computation.
  for (const auto &p : direct.get_params()) {
    direct.set_prior(p.first, FixedPrior());
  }

  using namespace std::chrono;
  high_resolution_clock::time_point start = high_resolution_clock::now();
  auto direct_fit = direct.fit(large_dataset);
  high_resolution_clock::time_point end = high_resolution_clock::now();
  auto direct_duration = duration_cast<microseconds>(end - start).count();

  UniformlySpacedInducingPoints strategy(100);
  auto sparse =
      sparse_gp_from_covariance(covariance, grouper, strategy, "sparse");

  sparse.set_param_value("inducing_nugget", 1e-6);

  start = high_resolution_clock::now();
  auto sparse_fit = sparse.fit(large_dataset);
  end = high_resolution_clock::now();
  auto sparse_duration = duration_cast<microseconds>(end - start).count();

  // Make sure the sparse version is a lot faster.
  EXPECT_LT(sparse_duration, 0.3 * direct_duration);
}

TYPED_TEST(SparseGaussianProcessTest, test_likelihood) {

  auto grouper = this->grouper;
  auto dataset = make_toy_sine_data(5., 10., 0.1, 12);
  auto covariance = make_simple_covariance_function();

  UniformlySpacedInducingPoints strategy(2);
  auto sparse =
      sparse_gp_from_covariance(covariance, grouper, strategy, "sparse");
  const auto inducing_points = strategy(covariance, dataset.features);
  // We need to make sure the priors on the parameters don't enter
  // into the log_likelihood computation.
  for (const auto &p : sparse.get_params()) {
    sparse.set_prior(p.first, FixedPrior());
  }

  // Here we build up the dense equivalent to the sparse GP covariance matrix
  // which we can then use to sanity check the likelihood computation
  const auto measurements = as_measurements(dataset.features);
  Eigen::MatrixXd K_uu = covariance(inducing_points);

  const double inducing_nugget = sparse.get_params()["inducing_nugget"].value;
  const double measurement_nugget =
      sparse.get_params()["measurement_nugget_"].value;
  K_uu.diagonal() += inducing_nugget * Eigen::VectorXd::Ones(K_uu.rows());

  const Eigen::MatrixXd K_fu = covariance(measurements, inducing_points);
  const Eigen::MatrixXd Q_ff = K_fu * (K_uu.ldlt().solve(K_fu.transpose()));

  Eigen::MatrixXd K = Q_ff;
  const auto indexers = group_by(dataset.features, grouper).indexers();
  for (const auto &idx_pair : indexers) {
    for (const Eigen::Index i : idx_pair.second) {
      for (const Eigen::Index j : idx_pair.second) {
        K(i, j) = covariance(measurements[i], measurements[j]);
      }
    }
  }
  K += dataset.targets.covariance;
  K.diagonal() += measurement_nugget * Eigen::VectorXd::Ones(K.rows());

  const double expected = -negative_log_likelihood(dataset.targets.mean, K);
  const double actual = sparse.log_likelihood(dataset);

  EXPECT_NEAR(expected, actual, 1e-2);
}

struct FixedInducingPoints {

  FixedInducingPoints(double min, double max, std::size_t num_points = 10)
      : min_(min), max_(max), num_points_(num_points) {}

  template <typename CovarianceFunction>
  std::vector<double> operator()(const CovarianceFunction &cov,
                                 const std::vector<double> &features) const {
    return linspace(min_, max_, num_points_);
  }

  double min_;
  double max_;
  std::size_t num_points_;
};

TEST(test_sparse_gp, test_update_exists) {

  using CovFunc = SquaredExponential<EuclideanDistance>;
  using MeanFunc = ZeroMean;
  using GrouperFunction = LeaveOneIntervalOut;
  using InducingPointStrategy = UniformlySpacedInducingPoints;

  using SparseGPR =
      SparseGaussianProcessRegression<CovFunc, MeanFunc, GrouperFunction,
                                      InducingPointStrategy>;

  using FitType = Fit<SparseGPFit<double>>;

  EXPECT_TRUE(bool(has_valid_update<SparseGPR, FitType, double>::value));
  EXPECT_TRUE(bool(can_update_in_place<SparseGPR, FitType, double>::value));
}

TYPED_TEST(SparseGaussianProcessTest, test_update) {
  auto grouper = this->grouper;
  auto covariance = make_simple_covariance_function();
  auto dataset = make_toy_linear_data();

  const double min =
      *std::min_element(dataset.features.begin(), dataset.features.end());
  const double max =
      *std::max_element(dataset.features.begin(), dataset.features.end());

  FixedInducingPoints strategy(min, max, 8);
  auto sparse =
      sparse_gp_from_covariance(covariance, grouper, strategy, "sparse");
  sparse.set_param(details::inducing_nugget_name(), 1e-3);
  sparse.set_param(details::measurement_nugget_name(), 1e-12);

  auto groups = dataset.group_by(grouper).groups();
  const std::size_t num_groups = groups.size();

  const auto held_out_pair = groups.first_group();
  groups.erase(held_out_pair.first);

  ASSERT_EQ(groups.size(), num_groups - 1);

  const auto partial_dataset = groups.combine();

  auto full_fit = sparse.fit(dataset);
  auto partial_fit = sparse.fit(partial_dataset);

  // Copy the fit so we can update it in place
  decltype(partial_fit) updated_fit(partial_fit);
  updated_fit.update_in_place(held_out_pair.second);

  auto test_features = linspace(0.01, 9.9, 11);

  auto full_pred =
      full_fit.predict_with_measurement_noise(test_features).joint();
  auto partial_pred =
      partial_fit.predict_with_measurement_noise(test_features).joint();
  auto updated_pred =
      updated_fit.predict_with_measurement_noise(test_features).joint();

  double partial_cov_diff =
      (partial_pred.covariance - full_pred.covariance).norm();
  double updated_cov_diff =
      (updated_pred.covariance - full_pred.covariance).norm();

  auto compute_sigma = [](const auto &fit_model) -> Eigen::MatrixXd {
    const Eigen::Index n = fit_model.get_fit().sigma_R.cols();
    Eigen::MatrixXd sigma = sqrt_solve(fit_model.get_fit().sigma_R,
                                       fit_model.get_fit().permutation_indices,
                                       Eigen::MatrixXd::Identity(n, n));
    return sigma.transpose() * sigma;
  };

  const Eigen::MatrixXd full_sigma = compute_sigma(full_fit);
  const Eigen::MatrixXd partial_sigma = compute_sigma(partial_fit);
  const Eigen::MatrixXd updated_sigma = compute_sigma(updated_fit);

  EXPECT_GT((partial_pred.mean - full_pred.mean).norm(), 1e-2);
  EXPECT_GT((partial_sigma - full_sigma).norm(), 1.);
  EXPECT_GT(partial_cov_diff, 1e-1);

  EXPECT_LT((updated_pred.mean - full_pred.mean).norm(), 1e-6);
  EXPECT_LT((updated_sigma - full_sigma).norm(), 1e-6);
  EXPECT_LT(updated_cov_diff, 1e-6);
}

TYPED_TEST(SparseGaussianProcessTest, test_rebase_inducing_points) {
  auto grouper = this->grouper;
  auto covariance = make_simple_covariance_function();
  auto dataset = make_toy_linear_data();

  const double min =
      *std::min_element(dataset.features.begin(), dataset.features.end());
  const double max =
      *std::max_element(dataset.features.begin(), dataset.features.end());

  FixedInducingPoints strategy(min, max, 8);
  auto sparse =
      sparse_gp_from_covariance(covariance, grouper, strategy, "sparse");
  sparse.set_param(details::inducing_nugget_name(), 1e-3);
  sparse.set_param(details::measurement_nugget_name(), 1e-12);

  auto full_fit = sparse.fit(dataset);
  auto test_features = linspace(0.01, 9.9, 11);
  auto full_pred =
      full_fit.predict_with_measurement_noise(test_features).joint();

  const std::vector<double> low_res_features = {5.};
  const auto low_res_fit = rebase_inducing_points(full_fit, low_res_features);
  auto low_res_pred =
      low_res_fit.predict_with_measurement_noise(test_features).joint();
  // Converting to a low res set of inducing points should lose information.
  EXPECT_GT((low_res_pred.mean - full_pred.mean).norm(), 10.);

  const auto high_res_features = linspace(0.01, 9.9, 51);
  const auto high_res_fit = rebase_inducing_points(full_fit, high_res_features);
  auto high_res_pred =
      high_res_fit.predict_with_measurement_noise(test_features).joint();
  // Increasing the inducing points shouldn't change much
  EXPECT_LT((high_res_pred.mean - full_pred.mean).norm(), 1e-6);

  const auto low_high_res_fit =
      rebase_inducing_points(low_res_fit, high_res_features);
  auto low_high_res_pred =
      low_high_res_fit.predict_with_measurement_noise(test_features).joint();
  // Decreasing then increasing the inducing points should lose info
  EXPECT_GT((low_high_res_pred.mean - full_pred.mean).norm(), 10.);
}

TYPED_TEST(SparseGaussianProcessTest, test_rebase_and_update) {

  auto grouper = this->grouper;
  auto covariance = make_simple_covariance_function();
  auto dataset = make_toy_linear_data();

  UniformlySpacedInducingPoints strategy(10);
  auto model =
      sparse_gp_from_covariance(covariance, grouper, strategy, "sparse");
  const auto inducing_points = strategy(covariance, dataset.features);
  auto test_features = linspace(0.1, 9.9, 5);

  const auto grouped = dataset.group_by(grouper).groups();
  auto iteratively_fit_model = model.fit(grouped.first_value());

  // The first fit is going to space the inducing points only over the
  // first group, here we set the inducing points to what they
  // would be if we'd fit to everything
  iteratively_fit_model =
      rebase_inducing_points(iteratively_fit_model, inducing_points);

  // Then iteratively update with the rest of the groups
  bool first = true;
  for (const auto &pair : grouped) {
    if (!first) {
      iteratively_fit_model.update_in_place(pair.second);
    } else {
      first = false;
    }
  }

  const auto direct_fit_model = model.fit(dataset);

  const auto iter_pred = iteratively_fit_model.predict(test_features).joint();
  const auto direct_pred = direct_fit_model.predict(test_features).joint();

  EXPECT_LT((direct_pred.mean - iter_pred.mean).norm(), 1e-5);
  EXPECT_LT((direct_pred.covariance - iter_pred.covariance).norm(), 1e-5);
}

// TYPED_TEST(SparseGaussianProcessTest, test_shift_inducing_points) {
//  auto grouper = this->grouper;
//  auto covariance = make_simple_covariance_function();
//  auto dataset = make_toy_linear_data();
//
//  const double min =
//      *std::min_element(dataset.features.begin(), dataset.features.end());
//  const double max =
//      *std::max_element(dataset.features.begin(), dataset.features.end());
//
//  FixedInducingPoints strategy(min, max, 8);
//  auto sparse =
//      sparse_gp_from_covariance(covariance, grouper, strategy, "sparse");
//  sparse.set_param(details::inducing_nugget_name(), 1e-3);
//  sparse.set_param(details::measurement_nugget_name(), 1e-12);
//
//  const auto full_fit = sparse.fit(dataset);
//  const auto test_features = linspace(0.01, 9.9, 11);
//  const auto full_pred =
//      full_fit.predict_with_measurement_noise(test_features).joint();
//
//  auto shifted_fit = full_fit;
//  Eigen::Index n = shifted_fit.get_fit().information.size();
//  shifted_fit.get_fit().shift_mean(Eigen::VectorXd::Ones(n));
//
//  const auto shifted_pred =
//      shifted_fit.predict_with_measurement_noise(test_features).joint();
//
//  const auto test_shift = Eigen::VectorXd::Ones(shifted_pred.mean.size());
//  EXPECT_LT((shifted_pred.mean - test_shift - full_pred.mean).norm(), 1e-4);
//  EXPECT_LT((shifted_pred.covariance - full_pred.covariance).norm(), 1e-8);
//}

} // namespace albatross
