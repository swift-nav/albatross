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

  if (dist.has_covariance()) {
    expect_subset_equal(dist.covariance, ss.covariance, indices);
  } else {
    EXPECT_FALSE(ss.has_covariance());
  }
};

REGISTER_TYPED_TEST_CASE_P(DistributionTest, test_subset);

Eigen::VectorXd arange(int k = 5) {
  Eigen::VectorXd mean(k);
  for (Eigen::Index i = 0; i < k; i++) {
    mean[i] = i;
  }
  return mean;
}

struct MarginalNoCovariance
    : public DistributionTestCase<MarginalDistribution> {
  virtual MarginalDistribution create() const {
    Eigen::VectorXd mean = arange(5);
    return MarginalDistribution(mean);
  }
};

struct MarginalWithCovariance
    : public DistributionTestCase<MarginalDistribution> {
  virtual MarginalDistribution create() const {
    Eigen::Index k = 5;
    Eigen::VectorXd mean = arange(k);
    Eigen::MatrixXd covariance = Eigen::MatrixXd::Random(k, k);
    covariance = covariance * covariance.transpose();

    return MarginalDistribution(mean, mean.asDiagonal());
  }
};

struct JointNoCovariance : public DistributionTestCase<JointDistribution> {
  virtual JointDistribution create() const {
    Eigen::Index k = 5;
    Eigen::VectorXd mean = arange(k);
    return JointDistribution(mean);
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

typedef ::testing::Types<MarginalNoCovariance, MarginalWithCovariance,
                         JointNoCovariance, JointWithCovariance>
    ToTest;

INSTANTIATE_TYPED_TEST_CASE_P(Albatross, DistributionTest, ToTest);

} // namespace albatross
