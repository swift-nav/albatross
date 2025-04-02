/*
 * Copyright (C) 2025 Swift Navigation Inc.
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

#include <albatross/CGGP>

namespace albatross {

std::ostream &operator<<(std::ostream &os, Eigen::ComputationInfo info) {
  switch (info) {
  case Eigen::Success:
    os << "Success";
    break;
  case Eigen::NumericalIssue:
    os << "NumericalIssue";
    break;
  case Eigen::NoConvergence:
    os << "NoConvergence";
    break;
  case Eigen::InvalidInput:
    os << "InvalidInput";
    break;
  default:
    os << "<invalid enum value " << info << ">";
  };
  return os;
}

namespace {
template <typename FitType>
void describe_fit(const FitType &fit, std::ostream &os = std::cout) {
  const auto options = fit.get_options();
  const auto state = fit.get_solver_state();
  os << "        Status: " << state.info << std::endl;
  os << "     Tolerance: " << options.tolerance << std::endl;
  os << "         Error: " << state.error << std::endl;
  os << "Max iterations: " << options.max_iterations << std::endl;
  os << "    Iterations: " << state.iterations << std::endl;
}
} // namespace

TEST(TestConjugateGradientGP, TestMean) {
  using CovFunc = SquaredExponential<EuclideanDistance>;

  CovFunc covariance(1, 1);
  auto dataset = make_toy_linear_data(5, 1, 1.e-3, 5000);
  auto direct = gp_from_covariance(covariance, "direct");
  auto cg = cg_gp_from_covariance(covariance, "cg", IterativeSolverOptions{},
                                  Eigen::PartialCholesky<double>{20, 1.e-3});
  // auto cg = cg_gp_from_covariance(covariance, "cg");

  auto begin = std::chrono::steady_clock::now();
  auto direct_fit = direct.fit(dataset);
  std::cout << "direct fit: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now() - begin)
                       .count() /
                   1.e9
            << std::endl;
  begin = std::chrono::steady_clock::now();
  auto cg_fit = cg.fit(dataset);
  std::cout << "cg fit: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now() - begin)
                       .count() /
                   1.e9
            << std::endl;

  describe_fit(cg_fit.get_fit());
  auto test_features = linspace(0.01, 9.9, 100);

  begin = std::chrono::steady_clock::now();
  auto direct_pred =
      direct_fit.predict_with_measurement_noise(test_features).joint();
  std::cout << "direct predict: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now() - begin)
                       .count() /
                   1.e9
            << std::endl;

  begin = std::chrono::steady_clock::now();
  auto cg_pred = cg_fit.predict_with_measurement_noise(test_features).joint();
  std::cout << "cg predict: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now() - begin)
                       .count() /
                   1.e9
            << std::endl;
  describe_fit(cg_fit.get_fit());

  EXPECT_TRUE(cg_pred.mean.array().isFinite().all());
  EXPECT_TRUE(cg_pred.covariance.array().isFinite().all());

  double mean_error = (direct_pred.mean - cg_pred.mean).norm();
  EXPECT_LT(mean_error, 1.e-9);

  double cov_error = (direct_pred.covariance - cg_pred.covariance).norm();
  EXPECT_LT(cov_error, 1.e-9);

  begin = std::chrono::steady_clock::now();
  auto direct_marginal =
      direct_fit.predict_with_measurement_noise(test_features).marginal();
  std::cout << "direct marginal: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now() - begin)
                       .count() /
                   1.e9
            << std::endl;

  begin = std::chrono::steady_clock::now();
  auto cg_marginal =
      cg_fit.predict_with_measurement_noise(test_features).marginal();
  std::cout << "cg marginal: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now() - begin)
                       .count() /
                   1.e9
            << std::endl;
  describe_fit(cg_fit.get_fit());
  double marginal_error = (direct_marginal.covariance.diagonal() -
                           cg_marginal.covariance.diagonal())
                              .norm();
  EXPECT_LT(marginal_error, 1.e-9);

  begin = std::chrono::steady_clock::now();
  auto direct_mean =
      direct_fit.predict_with_measurement_noise(test_features).mean();
  std::cout << "direct mean: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now() - begin)
                       .count() /
                   1.e9
            << std::endl;

  begin = std::chrono::steady_clock::now();
  auto cg_mean = cg_fit.predict_with_measurement_noise(test_features).mean();
  std::cout << "cg mean: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now() - begin)
                       .count() /
                   1.e9
            << std::endl;
  describe_fit(cg_fit.get_fit());
  double mean_predict_error = (direct_mean - cg_mean).norm();
  EXPECT_LT(mean_predict_error, 1.e-9);
}

} // namespace albatross