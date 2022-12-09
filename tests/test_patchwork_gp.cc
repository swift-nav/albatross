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
#include <fstream>

#include <albatross/PatchworkGP>
#include <albatross/utils/CsvUtils>

namespace albatross {

struct ExamplePatchworkFunctions {

  ExamplePatchworkFunctions() : width(5.) {}

  ExamplePatchworkFunctions(double width_) : width(width_){};

  long int grouper(const double &x) const { return lround(floor(x / width)); }

  std::vector<double> boundary(long int x, long int y) const {
    if (std::abs(x - y) == 1) {
      double center = width * 0.5 * cast::to_double(x + y);
      std::vector<double> boundary = {center - 1., center, center + 1.};
      return boundary;
    }
    return {};
  }

  long int nearest_group(const std::vector<long int> &groups,
                         const long int &query) const {
    long int nearest = query;
    long int nearest_distance = std::numeric_limits<long int>::max();
    for (const auto &group : groups) {
      const auto distance = std::abs(query - group);
      if (distance < nearest_distance) {
        nearest = group;
        nearest_distance = distance;
      }
    }
    return nearest;
  }

  double width;
};

RegressionDataset<double>
shuffle_dataset(const RegressionDataset<double> &dataset) {
  RegressionDataset<double> output(dataset);

  std::default_random_engine gen(2012);
  std::size_t n = dataset.size();
  std::uniform_int_distribution<std::size_t> dist(0, n - 1);

  for (std::size_t i = 0; i < n - 1; ++i) {
    std::size_t j = dist(gen);
    if (i != j) {
      std::swap(output.features[i], output.features[j]);
      std::swap(output.targets.mean[cast::to_index(i)],
                output.targets.mean[cast::to_index(j)]);
    }
  }
  return output;
}

template <typename CovFunc, typename PatchworkFunctions>
void expect_patchwork_gp_performance(
    const CovFunc &covariance, const PatchworkFunctions &patchwork_functions,
    double mean_threshold, double cov_threshold) {
  // There was a subtle bug which dealt with out of order groups
  // shuffling tests that edge case.
  const auto dataset = shuffle_dataset(make_toy_linear_data());

  const auto direct = gp_from_covariance(covariance, "direct");

  const auto patchwork =
      patchwork_gp_from_covariance(covariance, patchwork_functions);

  const auto direct_fit = direct.fit(dataset);
  const auto patchwork_fit = patchwork.fit(dataset);

  const auto test_features = linspace(0.01, 9.9, 11);

  const auto direct_pred = direct_fit.predict(test_features).joint();
  const auto patchwork_pred = patchwork_fit.predict(test_features).joint();

  const double patchwork_error =
      root_mean_square_error(patchwork_pred.mean, direct_pred.mean);

  EXPECT_LT(patchwork_error, mean_threshold);

  const double patchwork_cov_diff =
      (patchwork_pred.covariance - direct_pred.covariance).norm();

  EXPECT_LT(patchwork_cov_diff, cov_threshold);
}

TEST(test_patchwork_gp, test_traits) {

  struct PatchworkTestType {};

  EXPECT_TRUE(
      bool(details::patchwork_functions_are_valid<ExamplePatchworkFunctions,
                                                  double>::value));
  EXPECT_FALSE(
      bool(details::patchwork_functions_are_valid<ExamplePatchworkFunctions,
                                                  PatchworkTestType>::value));
}

TEST(test_patchwork_gp, test_sanity) {

  const ExamplePatchworkFunctions patchwork_functions;

  auto covariance = make_simple_covariance_function();

  covariance.set_param_value("squared_exponential_length_scale", 1000.);
  expect_patchwork_gp_performance(covariance, patchwork_functions, 0.1, 0.3);

  covariance.set_param_value("squared_exponential_length_scale", 100.);
  expect_patchwork_gp_performance(covariance, patchwork_functions, 1e-2, 0.3);

  covariance.set_param_value("squared_exponential_length_scale", 10.);
  expect_patchwork_gp_performance(covariance, patchwork_functions, 5e-2, 0.3);
}

TEST(test_patchwork_gp, test_one_group) {

  auto covariance = make_simple_covariance_function();

  // Using a large width makes this just use one group.
  const ExamplePatchworkFunctions patchwork_functions(1000.);

  expect_patchwork_gp_performance(covariance, patchwork_functions, 1e-6, 1e-6);
}

TEST(test_patchwork_gp, test_scales) {

  const auto covariance = make_simple_covariance_function();

  const auto large_dataset = make_toy_sine_data(5., 10., 0.1, 1000);
  const auto test_features = linspace(0.01, 9.9, 11);

  const auto direct = gp_from_covariance(covariance, "direct");

  using namespace std::chrono;
  high_resolution_clock::time_point start = high_resolution_clock::now();
  const auto direct_fit = direct.fit(large_dataset);
  const auto direct_pred = direct_fit.predict(test_features).joint();
  high_resolution_clock::time_point end = high_resolution_clock::now();
  const auto direct_duration = duration_cast<microseconds>(end - start).count();

  const ExamplePatchworkFunctions patchwork_functions(100.);

  auto patchwork =
      patchwork_gp_from_covariance(covariance, patchwork_functions);

  start = high_resolution_clock::now();
  const auto patchwork_fit = patchwork.fit(large_dataset);
  const auto patchwork_pred = patchwork_fit.predict(test_features).joint();
  end = high_resolution_clock::now();
  auto patchwork_duration = duration_cast<microseconds>(end - start).count();

  // Make sure the patchwork version is a lot faster.
  EXPECT_LT(static_cast<double>(patchwork_duration),
            0.3 * static_cast<double>(direct_duration));
}

} // namespace albatross
