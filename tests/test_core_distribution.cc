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

#include <gtest/gtest.h>

#include "Distribution"
#include "test_core_distribution.h"

namespace albatross {

using TestDistribution = Distribution<Eigen::MatrixXd>;

/*
 * Simply makes sure that a BaseModel that should be able to
 * make perfect predictions compiles and runs as expected.
 */
TEST(test_core_distribution, test_constructors) {
  int k = 5;
  Eigen::VectorXd mean(k);
  Eigen::MatrixXd cov(k, k);

  for (Eigen::Index i = 0; i < k; i++) {
    mean[i] = i;
  }

  cov = mean * mean.transpose();

  TestDistribution no_cov(mean);
  ASSERT_FALSE(no_cov.has_covariance());

  TestDistribution with_cov(mean, cov);
  ASSERT_TRUE(with_cov.has_covariance());
}

Eigen::VectorXd arange(int k = 5) {
  Eigen::VectorXd mean(k);
  for (Eigen::Index i = 0; i < k; i++) {
    mean[i] = i;
  }
  return mean;
}

void expect_subset_equal(const Eigen::VectorXd &original,
                         const Eigen::VectorXd &actual,
                         const std::vector<int> indices) {
  for (std::size_t i = 0; i < indices.size(); i++) {
    EXPECT_EQ(actual[i], original[indices[i]]);
  }
};

void expect_subset_equal(const Eigen::MatrixXd &original,
                         const Eigen::MatrixXd &actual,
                         const std::vector<int> indices) {
  for (std::size_t i = 0; i < indices.size(); i++) {
    for (std::size_t j = 0; j < indices.size(); j++) {
      EXPECT_EQ(actual(i, j), original(indices[i], indices[j]));
    }
  }
}

template <typename Scalar, int Size>
void expect_subset_equal(const Eigen::DiagonalMatrix<Scalar, Size> &original,
                         const Eigen::DiagonalMatrix<Scalar, Size> &actual,
                         const std::vector<int> indices) {
  expect_subset_equal(original.diagonal(), actual.diagonal(), indices);
}

template <typename DistributionType>
class PolymorphicDistributionTest : public ::testing::Test {};

typedef ::testing::Types<MarginalDistribution, JointDistribution>
    DistributionsToTest;
TYPED_TEST_CASE(PolymorphicDistributionTest, DistributionsToTest);

TYPED_TEST(PolymorphicDistributionTest, can_compute_subset) {
  const TypeParam dist(arange(5));

  std::vector<int> indices = {1, 3, 2};
  const auto ss = subset(indices, dist);

  expect_subset_equal(dist.mean, ss.mean, indices);

  if (dist.has_covariance()) {
    expect_subset_equal(dist.covariance, ss.covariance, indices);
  }
}
} // namespace albatross
