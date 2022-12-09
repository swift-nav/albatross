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
                                     std::size_t seed = 7) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  gen.seed(static_cast<std::mt19937::result_type>(seed));
  std::normal_distribution<> d{0., 0.1};

  Eigen::VectorXd targets(cast::to_index(points.size()));

  auto spherical_function = [](Eigen::VectorXd &x) {
    return x[0] * x[1] + x[1] * x[2] + x[3];
  };

  for (std::size_t i = 0; i < points.size(); i++) {
    targets[cast::to_index(i)] = spherical_function(points[i]);
  }

  return RegressionDataset<Eigen::VectorXd>(points, targets);
}

TEST(test_radial, test_is_positive_definite) {
  const auto points = random_spherical_points(100);

  const Exponential<AngularDistance> term(2 * M_PI);

  const Eigen::MatrixXd cov = term(points);

  EXPECT_GE(cov.eigenvalues().real().array().minCoeff(), 0.);
}

class SquaredExponentialSSRTest {
public:
  std::vector<double> features() const { return linspace(0., 10., 101); }

  auto covariance_function() const {
    SquaredExponential<EuclideanDistance> cov(5., 1.);
    return cov;
  }

  double get_tolerance() const { return 1e-12; }
};

class ExponentialSSRTest {
public:
  std::vector<double> features() const { return linspace(0., 10., 11); }

  auto covariance_function() const {
    Exponential<EuclideanDistance> cov(5., 1.);
    return cov;
  }

  double get_tolerance() const { return 1e-2; }
};

class ExponentialAngularSSRTest {
public:
  std::vector<double> features() const { return linspace(0., M_2_PI, 11); }

  auto covariance_function() const {
    Exponential<EuclideanDistance> cov(M_PI_4, 1.);
    return cov;
  }

  double get_tolerance() const { return 1e-2; }
};

template <typename T>
class CovarianceStateSpaceTester : public ::testing::Test {
public:
  T test_case;
};

using StateSpaceTestCases =
    ::testing::Types<SquaredExponentialSSRTest, ExponentialSSRTest,
                     ExponentialAngularSSRTest>;
TYPED_TEST_CASE(CovarianceStateSpaceTester, StateSpaceTestCases);

TYPED_TEST(CovarianceStateSpaceTester, test_state_space_representation) {

  const auto xs = this->test_case.features();

  const auto cov_func = this->test_case.covariance_function();

  expect_state_space_representation_quality(cov_func, xs,
                                            this->test_case.get_tolerance());
}

} // namespace albatross
