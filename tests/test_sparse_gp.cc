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

std::string get_group(const double &f) {
  return std::to_string(static_cast<int>(f) / 10);
}

struct LeaveOneIntervalOut {

  std::string operator() (const double &f) const {
    return get_group(f);
  }

};

template <typename IndexerType>
class SparseGaussianProcessTest : public ::testing::Test {
public:
  IndexerType indexer;
};

typedef ::testing::Types<LeaveOneOut, LeaveOneIntervalOut>
    IndependenceAssumptions;
TYPED_TEST_CASE(SparseGaussianProcessTest, IndependenceAssumptions);

template <typename CovFunc, typename Indexer>
void expect_sparse_gp_performance(const CovFunc &covariance,
                                  const Indexer &indexer,
                                  double sparse_error_threshold,
                                  double really_sparse_error_threshold) {

  auto dataset = make_toy_linear_data();
  auto direct = gp_from_covariance(covariance, "direct");

  UniformlySpacedInducingPoints strategy(8);
  auto sparse =
      sparse_gp_from_covariance(covariance, strategy, indexer, "sparse");
  sparse.set_param(details::inducing_nugget_name(), 1e-3);
  sparse.set_param(details::measurement_nugget_name(), 1e-12);

  UniformlySpacedInducingPoints bad_strategy(3);
  auto really_sparse = sparse_gp_from_covariance(covariance, bad_strategy,
                                                 indexer, "really_sparse");
  really_sparse.set_param(details::inducing_nugget_name(), 1e-3);
  really_sparse.set_param(details::measurement_nugget_name(), 1e-12);

  auto direct_fit = direct.fit(dataset);
  auto sparse_fit = sparse.fit(dataset);
  auto really_sparse_fit = really_sparse.fit(dataset);

  auto test_features = linspace(0.01, 9.9, 11);

  auto direct_pred =
      direct_fit.predict_with_measurement_noise(test_features).joint();
  auto sparse_pred =
      sparse_fit.predict_with_measurement_noise(test_features).joint();
  auto really_sparse_pred =
      really_sparse_fit.predict_with_measurement_noise(test_features).joint();

  double sparse_error = (sparse_pred.mean - direct_pred.mean).norm();
  double really_sparse_error =
      (really_sparse_pred.mean - direct_pred.mean).norm();
  EXPECT_LT(sparse_error, sparse_error_threshold);
  EXPECT_LT(really_sparse_error, really_sparse_error_threshold);
  EXPECT_GT(really_sparse_error, sparse_error);

  double sparse_cov_diff =
      (sparse_pred.covariance - direct_pred.covariance).norm();
  double really_sparse_cov_diff =
      (really_sparse_pred.covariance - direct_pred.covariance).norm();

  EXPECT_LT(sparse_cov_diff, sparse_error_threshold);
  EXPECT_LT(really_sparse_cov_diff, really_sparse_error_threshold);
  EXPECT_GT(really_sparse_cov_diff, sparse_cov_diff);
}

TYPED_TEST(SparseGaussianProcessTest, test_sanity) {

  auto indexer = this->indexer;
  auto covariance = make_simple_covariance_function();

  // When the length scale is large the model with more inducing points
  // gets very nearly singular.  this checks to make sure that's dealt with
  // gracefully.
  covariance.set_param("squared_exponential_length_scale", 1000.);
  expect_sparse_gp_performance(covariance, indexer, 1e-2, 0.5);

  covariance.set_param("squared_exponential_length_scale", 100.);
  expect_sparse_gp_performance(covariance, indexer, 1e-2, 0.5);

  // Then when the length scale is shorter, the really sparse model
  // should become significantly worse than the sparse one.
  covariance.set_param("squared_exponential_length_scale", 10.);
  expect_sparse_gp_performance(covariance, indexer, 5e-2, 100.);
}

TYPED_TEST(SparseGaussianProcessTest, test_scales) {

  auto indexer = this->indexer;

  auto large_dataset = make_toy_sine_data(5., 10., 0.1, 1000);

  auto covariance = make_simple_covariance_function();

  auto direct = gp_from_covariance(covariance, "direct");

  using namespace std::chrono;
  high_resolution_clock::time_point start = high_resolution_clock::now();
  auto direct_fit = direct.fit(large_dataset);
  high_resolution_clock::time_point end = high_resolution_clock::now();
  auto direct_duration = duration_cast<microseconds>(end - start).count();

  UniformlySpacedInducingPoints strategy(100);
  auto sparse =
      sparse_gp_from_covariance(covariance, strategy, indexer, "sparse");

  start = high_resolution_clock::now();
  auto sparse_fit = sparse.fit(large_dataset);
  end = high_resolution_clock::now();
  auto sparse_duration = duration_cast<microseconds>(end - start).count();

  // Make sure the sparse version is a lot faster.
  EXPECT_LT(sparse_duration, 0.3 * direct_duration);
}

} // namespace albatross
