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

TEST(TestConjugateGradientGP, TestMean) {
  using CovFunc = SquaredExponential<EuclideanDistance>;

  CovFunc covariance(1, 1);
  auto dataset = make_toy_linear_data();
  auto direct = gp_from_covariance(covariance, "direct");
  auto cg = cg_gp_from_covariance(covariance, "cg");

  auto direct_fit = direct.fit(dataset);
  auto cg_fit = cg.fit(dataset);

  auto test_features = linspace(0.01, 9.9, 11);

  auto direct_pred = direct_fit.predict_with_measurement_noise(test_features).joint();
  auto cg_pred = cg_fit.predict_with_measurement_noise(test_features).joint();

  double mean_error = (direct_pred.mean - cg_pred.mean).norm();
  EXPECT_LT(mean_error, 1.e-9);

  double cov_distance =
      albatross::distance::wasserstein_2(direct_pred, cg_pred);

  EXPECT_LT(cov_distance, 1.e-9);
}

} // namespace albatross