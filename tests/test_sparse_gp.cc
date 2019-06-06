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

struct LeaveOneIntervalOut : public LeaveOneGroupOut<double> {
  LeaveOneIntervalOut() : LeaveOneGroupOut<double>(get_group){};
};

template <typename IndexerType>
class SparseGaussianProcessTest : public ::testing::Test {
public:
  IndexerType indexer;
};

typedef ::testing::Types<LeaveOneOut, LeaveOneIntervalOut>
    IndependenceAssumptions;
TYPED_TEST_CASE(SparseGaussianProcessTest, IndependenceAssumptions);

TYPED_TEST(SparseGaussianProcessTest, test_sanity) {

  auto indexer = this->indexer;

  auto covariance = make_simple_covariance_function();
  auto dataset = make_toy_linear_data();

  auto direct = gp_from_covariance(covariance, "direct");

  UniformlySpacedInducingPoints strategy(8);
  auto sparse =
      sparse_gp_from_covariance(covariance, strategy, indexer, "sparse");

  UniformlySpacedInducingPoints bad_strategy(3);
  auto really_sparse = sparse_gp_from_covariance(covariance, bad_strategy,
                                                 indexer, "really_sparse");

  auto test_features = linspace(0.01, 9.9, 11);

  auto sparse_pred =
      sparse.fit(dataset).predict_with_measurement_noise(test_features).joint();
  auto really_sparse_pred = really_sparse.fit(dataset)
                                .predict_with_measurement_noise(test_features)
                                .joint();
  auto direct_pred =
      direct.fit(dataset).predict_with_measurement_noise(test_features).joint();

  double sparse_error = (sparse_pred.mean - direct_pred.mean).norm();
  double really_sparse_error =
      (really_sparse_pred.mean - direct_pred.mean).norm();
  EXPECT_LT(sparse_error, 1e-2);
  EXPECT_LT(really_sparse_error, 0.5);
  EXPECT_GT(really_sparse_error, sparse_error);

  double sparse_cov_diff =
      (sparse_pred.covariance - direct_pred.covariance).norm();
  double really_sparse_cov_diff =
      (really_sparse_pred.covariance - direct_pred.covariance).norm();

  std::cout << direct_pred.covariance << std::endl;
  std::cout << sparse_pred.covariance << std::endl;

  EXPECT_LT(sparse_cov_diff, 1e-2);
  EXPECT_LT(really_sparse_cov_diff, 0.5);
  EXPECT_GT(really_sparse_cov_diff, sparse_cov_diff);
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
