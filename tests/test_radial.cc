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

#include <albatross/CovarianceFunctions>
#include <gtest/gtest.h>

#include "test_utils.h"

namespace albatross {

inline auto random_spherical_dataset(std::vector<Eigen::VectorXd> points,
                                     int seed = 7) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  gen.seed(seed);
  std::normal_distribution<> d{0., 0.1};

  Eigen::VectorXd targets(static_cast<Eigen::Index>(points.size()));

  auto spherical_function = [](Eigen::VectorXd &x) {
    return x[0] * x[1] + x[1] * x[2] + x[3];
  };

  for (std::size_t i = 0; i < points.size(); i++) {
    targets[static_cast<Eigen::Index>(i)] = spherical_function(points[i]);
  }

  return RegressionDataset<Eigen::VectorXd>(points, targets);
}

TEST(test_radial, test_is_positive_definite) {
  const auto points = random_spherical_points(100);

  const Exponential<AngularDistance> term(2 * M_PI);

  const Eigen::MatrixXd cov = term(points);

  EXPECT_GE(cov.eigenvalues().real().array().minCoeff(), 0.);
}

TEST(test_radial, test_gridded_features) {

  EuclideanDistance cov;
  std::vector<double> features = linspace(0., 5., 6);
  const auto grid = cov.gridded_features(features, 0.5);
  EXPECT_GT(grid.size(), features.size());
}

TEST(test_radial, test_get_ssr_features) {

  SquaredExponential<EuclideanDistance> cov;

  std::vector<double> features = linspace(0., 5., 6);

  cov.squared_exponential_length_scale.value = 100.;
  EXPECT_EQ(cov.get_ssr_features(features).size(), 2);

  cov.squared_exponential_length_scale.value = 10.;
  EXPECT_GT(cov.get_ssr_features(features).size(), 3);
  EXPECT_LT(cov.get_ssr_features(features).size(), 10);

  cov.squared_exponential_length_scale.value = 1.;
  EXPECT_GT(cov.get_ssr_features(features).size(), 10);
}

} // namespace albatross
