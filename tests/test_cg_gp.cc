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

constexpr const Eigen::Index cProblemSize{6000};

} // namespace

TEST(TestConjugateGradientGP, TestMean) {
  using CovFunc =
      SumOfCovarianceFunctions<SquaredExponential<EuclideanDistance>,
                               IndependentNoise<double>>;

  CovFunc covariance{};
  covariance.set_param_value("squared_exponential_length_scale", 500.0);
  covariance.set_param_value("sigma_squared_exponential", 2.0);
  covariance.set_param_value("sigma_independent_noise", 1.e-2);
  std::cout << "Problem size: " << cProblemSize << '\n';

  auto params = covariance.get_params();
  for (const auto &[key, param] : params) {
    std::cout << "Param '" << key << "': " << param.value << '\n';
  }

  auto dataset = make_toy_linear_data(5, 0.01, 1.e-2, cProblemSize);
  auto direct = gp_from_covariance(covariance, "direct");
  IterativeSolverOptions options{};
  options.tolerance = 1.e-8;
  options.max_iterations = cProblemSize / 100;
  auto cg = cg_gp_from_covariance(covariance, "cg", options,
                                  Eigen::PartialCholesky<double>{100, 1.e-1});
  // auto cg = cg_gp_from_covariance(covariance, "cg");

  auto begin = std::chrono::steady_clock::now();

  const auto report_seconds_till_now = [&begin](const std::string &label) {
    std::cout << label << ": "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - begin)
                         .count() /
                     1.e9
              << '\n';
  };

  static constexpr const double cIterativeErrorTolerance{1.e-5};

  auto direct_fit = direct.fit(dataset);
  report_seconds_till_now("direct fit");
  begin = std::chrono::steady_clock::now();
  auto cg_fit = cg.fit(dataset);
  report_seconds_till_now("cg fit");

  describe_fit(cg_fit.get_fit());
  auto test_features = linspace(0.01, 9.9, 100);

  begin = std::chrono::steady_clock::now();
  auto direct_pred =
    direct_fit.predict_with_measurement_noise(test_features).joint();
  report_seconds_till_now("direct predict");

  begin = std::chrono::steady_clock::now();
  auto cg_pred = cg_fit.predict_with_measurement_noise(test_features).joint();
  report_seconds_till_now("cg predict");
  describe_fit(cg_fit.get_fit());

  EXPECT_TRUE(cg_pred.mean.array().isFinite().all());
  EXPECT_TRUE(cg_pred.covariance.array().isFinite().all());

  double mean_error = (direct_pred.mean - cg_pred.mean).norm();
  EXPECT_LT(mean_error, cIterativeErrorTolerance);

  double cov_error = (direct_pred.covariance - cg_pred.covariance).norm();
  EXPECT_LT(cov_error, cIterativeErrorTolerance);

  begin = std::chrono::steady_clock::now();
  auto direct_marginal =
    direct_fit.predict_with_measurement_noise(test_features).marginal();
  report_seconds_till_now("direct marginal");

  begin = std::chrono::steady_clock::now();
  auto cg_marginal =
    cg_fit.predict_with_measurement_noise(test_features).marginal();
  report_seconds_till_now("cg marginal");
  describe_fit(cg_fit.get_fit());
  double marginal_error = (direct_marginal.covariance.diagonal() -
                           cg_marginal.covariance.diagonal())
                              .norm();
  EXPECT_LT(marginal_error, cIterativeErrorTolerance);

  begin = std::chrono::steady_clock::now();
  auto direct_mean =
      direct_fit.predict_with_measurement_noise(test_features).mean();
  report_seconds_till_now("direct mean");

  begin = std::chrono::steady_clock::now();
  auto cg_mean = cg_fit.predict_with_measurement_noise(test_features).mean();
  report_seconds_till_now("cg mean");
  std::cout
      << "(Mean prediction requires no additional iterative computation)\n";
  double mean_predict_error = (direct_mean - cg_mean).norm();
  EXPECT_LT(mean_predict_error, cIterativeErrorTolerance);
}

} // namespace albatross