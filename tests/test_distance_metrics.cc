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

#include "covariance_functions/distance_metrics.h"
#include "test_utils.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

namespace albatross {

TEST(test_distance_metrics, test_euclidean_distance) {

  Eigen::VectorXd x(3);
  Eigen::VectorXd y(3);
  EuclideanDistance dist;

  x << 1., 1., 1.;
  y << 1., 1., 2.;
  EXPECT_DOUBLE_EQ(dist(x, y), 1.);

  x << 1., 1., 1.;
  y << 2., 2., 2.;
  EXPECT_DOUBLE_EQ(dist(x, y), sqrt(3.));

  x << 2., 2., 2.;
  y << 2., 2., 2.;
  EXPECT_DOUBLE_EQ(dist(x, y), 0.);
}

TEST(test_distance_metrics, test_radial_distance) {

  Eigen::VectorXd x(3);
  Eigen::VectorXd y(3);
  RadialDistance dist;

  x << 0., 0., 1.;
  y << 0., 0., 1.;
  EXPECT_DOUBLE_EQ(dist(x, y), 0.);

  x << 0., 0., 1.;
  y << 0., 1., 0.;
  EXPECT_DOUBLE_EQ(dist(x, y), 0.);

  x << 0., 1., 1.;
  y << 1., 0., 0.;
  EXPECT_DOUBLE_EQ(dist(x, y), sqrt(2.) - 1.);
}

TEST(test_distance_metrics, test_angular_distance) {

  Eigen::VectorXd x(3);
  Eigen::VectorXd y(3);
  AngularDistance dist;

  x << 0., 0., 1.;
  y << 0., 0., 1.;
  EXPECT_DOUBLE_EQ(dist(x, y), 0.);

  x << 0., 0., 1.;
  y << 0., 0., -1.;
  EXPECT_DOUBLE_EQ(dist(x, y), M_PI);

  x << 0., 0., 1.;
  y << 0., 1., 0.;
  EXPECT_DOUBLE_EQ(dist(x, y), M_PI / 2.);
}

TEST(test_distance_metrics, test_distance_matrix) {

  const auto points = random_spherical_points(10);

  EuclideanDistance dist;

  const auto dist_matrix = distance_matrix(dist, points);
  EXPECT_EQ(dist_matrix.rows(), points.size());
  EXPECT_EQ(dist_matrix.cols(), points.size());
}

} // namespace albatross
