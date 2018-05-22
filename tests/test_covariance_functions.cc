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

#include "covariance_functions/covariance_functions.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

namespace albatross {

std::vector<Eigen::Vector3d> points_on_a_line(const int n) {
  std::vector<Eigen::Vector3d> xs;
  for (int i = 0; i < n; i++) {
    Eigen::Vector3d x;
    for (int j = 0; j < 3; j++)
      x[static_cast<std::size_t>(j)] = 1000 * i + j;
    xs.push_back(x);
  }
  return xs;
}

TEST(test_covariance_functions, test_build_covariance) {
  using Feature = Eigen::Vector3d;
  using Noise = IndependentNoise<Feature>;
  using SqExp = SquaredExponential<EuclideanDistance>;
  using RadialSqExp = SquaredExponential<RadialDistance>;

  CovarianceFunction<SqExp> sqexp = {SqExp()};
  CovarianceFunction<Constant> constant = {Constant()};
  CovarianceFunction<Noise> noise = {Noise()};
  CovarianceFunction<RadialSqExp> radial_sqexp = {RadialSqExp()};

  // Add and multiply covariance functions together and make sure they are
  // still capable of producing a covariance matrix.
  auto product = sqexp * radial_sqexp;
  auto covariance_function = constant + product + noise;

  auto xs = points_on_a_line(5);
  Eigen::MatrixXd C = symmetric_covariance(covariance_function, xs);
  assert(C.rows() == xs.size());
  assert(C.cols() == xs.size());
}

/*
 * In the following we test any covariance functions which should support
 * Eigen::Vector feature vectors.
 */
template <typename T>
class TestVectorCovarianceFunctions : public ::testing::Test {

public:
  typedef CovarianceFunction<T> CovFunc;
  T value_;
};

typedef ::testing::Types<SquaredExponential<EuclideanDistance>,
                         SquaredExponential<RadialDistance>>
    VectorCompatibleCovarianceFunctions;

TYPED_TEST_CASE(TestVectorCovarianceFunctions,
                VectorCompatibleCovarianceFunctions);

TYPED_TEST(TestVectorCovarianceFunctions, WorksWithEigen) {

  typename TestFixture::CovFunc covariance_function;

  auto xs = points_on_a_line(5);
  Eigen::MatrixXd C = symmetric_covariance(covariance_function, xs);
  assert(C.rows() == xs.size());
  assert(C.cols() == xs.size());
  // Make sure C is positive definite.
  auto inverse = C.inverse();
}
} // namespace albatross
