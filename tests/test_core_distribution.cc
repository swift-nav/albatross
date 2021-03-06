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

#include <albatross/Distribution>
#include <albatross/Indexing>
#include <gtest/gtest.h>

#include "test_core_distribution.h"

namespace albatross {

TYPED_TEST_P(DistributionTest, test_subset) {

  TypeParam test_case;
  const auto dist = test_case.create();

  std::vector<int> indices = {1, 3, 2};
  const auto ss = subset(dist, indices);

  expect_subset_equal(dist.mean, ss.mean, indices);

  expect_subset_equal(dist.covariance, ss.covariance, indices);
};

TYPED_TEST_P(DistributionTest, test_multiply_with_matrix) {

  TypeParam test_case;
  const auto dist = test_case.create();

  Eigen::MatrixXd mat = Eigen::MatrixXd::Random(dist.size() - 1, dist.size());

  const auto transformed_distribution = dist * mat;

  EXPECT_EQ(transformed_distribution.mean, mat * dist.mean);
  EXPECT_EQ(transformed_distribution.covariance,
            mat * dist.covariance * mat.transpose());
};

TYPED_TEST_P(DistributionTest, test_multiply_with_vector) {

  TypeParam test_case;
  const auto dist = test_case.create();

  Eigen::VectorXd vector = Eigen::VectorXd::Random(dist.size());

  const auto transformed_distribution = dist * vector;

  double expected_mean = vector.dot(dist.mean);
  double expected_variance = vector.dot(dist.covariance * vector);
  EXPECT_EQ(transformed_distribution.mean[0], expected_mean);
  EXPECT_EQ(transformed_distribution.covariance(0, 0), expected_variance);
};

TYPED_TEST_P(DistributionTest, test_multiply_by_scalar) {

  TypeParam test_case;
  const auto dist = test_case.create();

  double scalar = 3.;

  const auto transformed_distribution = dist * scalar;

  EXPECT_EQ(transformed_distribution.mean, scalar * dist.mean);
  Eigen::MatrixXd scaled = dist.covariance;
  scaled *= scalar * scalar;
  const Eigen::MatrixXd actual = transformed_distribution.covariance;
  EXPECT_EQ(actual, scaled);
};

TYPED_TEST_P(DistributionTest, test_add) {

  TypeParam test_case;
  const auto dist = test_case.create();

  const auto transformed_distribution = dist + dist;

  EXPECT_EQ(transformed_distribution.mean, dist.mean + dist.mean);
  const Eigen::MatrixXd expected = 2 * dist.covariance;
  const Eigen::MatrixXd actual = transformed_distribution.covariance;
  EXPECT_EQ(actual, expected);
};

TYPED_TEST_P(DistributionTest, test_subtract) {

  TypeParam test_case;
  const auto dist = test_case.create();

  const auto transformed_distribution = dist - dist;

  EXPECT_EQ(transformed_distribution.mean,
            Eigen::VectorXd::Zero(dist.mean.size()));
  const Eigen::MatrixXd expected = 2 * dist.covariance;
  const Eigen::MatrixXd actual = transformed_distribution.covariance;
  EXPECT_EQ(actual, expected);
};

REGISTER_TYPED_TEST_CASE_P(DistributionTest, test_subset,
                           test_multiply_with_matrix, test_multiply_with_vector,
                           test_multiply_by_scalar, test_add, test_subtract);

Eigen::VectorXd arange(int k = 5) {
  Eigen::VectorXd mean(k);
  for (Eigen::Index i = 0; i < k; i++) {
    mean[i] = i;
  }
  return mean;
}

struct MarginalWithCovariance
    : public DistributionTestCase<MarginalDistribution> {
  virtual MarginalDistribution create() const {
    Eigen::Index k = 5;
    Eigen::VectorXd mean = arange(k);
    Eigen::MatrixXd covariance = Eigen::MatrixXd::Random(k, k);
    covariance = covariance * covariance.transpose();
    return MarginalDistribution(mean, covariance.diagonal().asDiagonal());
  }
};

struct JointWithCovariance : public DistributionTestCase<JointDistribution> {
  virtual JointDistribution create() const {
    Eigen::Index k = 5;
    Eigen::VectorXd mean = arange(k);
    Eigen::MatrixXd covariance = Eigen::MatrixXd::Random(k, k);
    covariance = covariance * covariance.transpose();

    return JointDistribution(mean, covariance);
  }
};

typedef ::testing::Types<MarginalWithCovariance, JointWithCovariance> ToTest;

INSTANTIATE_TYPED_TEST_CASE_P(Albatross, DistributionTest, ToTest);

} // namespace albatross
