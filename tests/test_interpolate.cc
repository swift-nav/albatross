/*
 * Copyright (C) 2020 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <albatross/GP>

#include "../include/albatross/src/models/interpolate.hpp"

#include <gtest/gtest.h>

namespace albatross {

std::vector<double> uniform_points_on_line(const std::size_t n,
                                           const double low,
                                           const double high) {
  std::vector<double> xs;
  for (std::size_t i = 0; i < n; i++) {
    double ratio = (double)i / (double)(n - 1);
    xs.push_back(low + ratio * (high - low));
  }
  return xs;
};

TEST(test_interpolator, test_interpolate) {

  const auto xs = uniform_points_on_line(21, 0., 2 * 3.14159);

  Eigen::VectorXd targets(xs.size());
  for (std::size_t i = 0; i < xs.size(); ++i) {
    targets[i] = std::sin(xs[i]);
  }

  const auto interp_xs = uniform_points_on_line(101, 0., 2 * 3.14159);

  Eigen::VectorXd mean_truth(interp_xs.size());
  Eigen::VectorXd derivative_truth(interp_xs.size());
  Eigen::VectorXd second_derivative_truth(interp_xs.size());

  for (std::size_t i = 0; i < interp_xs.size(); ++i) {
    mean_truth[i] = std::sin(interp_xs[i]);
    derivative_truth[i] = std::cos(interp_xs[i]);
    second_derivative_truth[i] = -std::sin(interp_xs[i]);
  }

  GaussianProcessInterpolator interpolator;
  interpolator.set_param("squared_exponential_length_scale", 10.);
  interpolator.set_param("sigma_squared_exponential", 10.);
  interpolator.set_param("sigma_independent_noise", 1e-6);


  const auto predictor = interpolator.fit(xs, targets).predict(interp_xs);

  EXPECT_LT((predictor.mean() - mean_truth).norm(), 1e-3);
  EXPECT_LT((predictor.derivative() - derivative_truth).norm(), 1e-2);
  EXPECT_LT((predictor.second_derivative() - second_derivative_truth).norm(), 1e-1);

}

}
