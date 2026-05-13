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

TEST(test_core_distribution, create_one_dim) {

  const double mean = M_PI;
  const double var = std::log(2);
  const MarginalDistribution one_dim(mean, var);
  const Eigen::VectorXd mean_vec = Eigen::VectorXd::Constant(1, mean);
  const Eigen::VectorXd var_vec = Eigen::VectorXd::Constant(1, var);

  const MarginalDistribution from_vectors(mean_vec, var_vec);
  EXPECT_EQ(one_dim, from_vectors);
}

TYPED_TEST_P(DistributionTest, test_subset) {

  TypeParam test_case;
  const auto dist = test_case.create();

  std::vector<Eigen::Index> indices = {1, 3, 2};
  const auto ss = subset(dist, indices);

  expect_subset_equal(dist.mean, ss.mean, indices);

  expect_subset_equal(dist.covariance, ss.covariance, indices);
};

TYPED_TEST_P(DistributionTest, test_multiply_with_matrix_marginal) {

  TypeParam test_case;
  const auto dist = test_case.create();

  const auto dist_size = cast::to_index(dist.size());
  Eigen::MatrixXd mat = Eigen::MatrixXd::Random(dist_size - 1, dist_size);

  const auto transformed_distribution = (mat * dist).marginal();
  const MarginalDistribution alternate = mat * dist;
  EXPECT_EQ(alternate, transformed_distribution);

  EXPECT_EQ(transformed_distribution.mean, mat * dist.mean);
  const Eigen::MatrixXd cov = mat * dist.covariance * mat.transpose();
  EXPECT_LT(
      (transformed_distribution.covariance.diagonal() - cov.diagonal()).norm(),
      1e-8);
};

TYPED_TEST_P(DistributionTest, test_multiply_with_matrix_joint) {

  TypeParam test_case;
  const auto dist = test_case.create();

  const auto dist_size = cast::to_index(dist.size());
  Eigen::MatrixXd mat = Eigen::MatrixXd::Random(dist_size - 1, dist_size);

  const auto transformed_distribution = (mat * dist).joint();
  const JointDistribution alternate = mat * dist;
  EXPECT_EQ(alternate, transformed_distribution);

  EXPECT_EQ(transformed_distribution.mean, mat * dist.mean);
  const Eigen::MatrixXd cov = mat * dist.covariance * mat.transpose();
  EXPECT_LT((transformed_distribution.covariance - cov).norm(), 1e-8);
};

TYPED_TEST_P(DistributionTest, test_multiply_with_sparse_matrix_marginal) {

  const Eigen::Index n = 3;
  const Eigen::VectorXd mean = Eigen::VectorXd::Random(n, 1);
  const Eigen::VectorXd var = Eigen::VectorXd::Random(n, 1).array().square();
  const MarginalDistribution dist(mean, var);
  const auto dist_size = cast::to_index(dist.size());
  const Eigen::SparseMatrix<double> mat =
      Eigen::MatrixXd::Random(dist_size - 1, dist_size).sparseView();

  const auto transformed_distribution = (mat * dist).marginal();
  const MarginalDistribution alternate = mat * dist;
  EXPECT_EQ(alternate, transformed_distribution);

  EXPECT_EQ(transformed_distribution.mean, mat * dist.mean);
  const Eigen::MatrixXd cov = mat * dist.covariance * mat.transpose();
  EXPECT_LT(
      (transformed_distribution.covariance.diagonal() - cov.diagonal()).norm(),
      1e-8);
};

TYPED_TEST_P(DistributionTest, test_multiply_with_sparse_matrix_joint) {

  const Eigen::Index n = 3;
  const Eigen::VectorXd mean = Eigen::VectorXd::Random(n, 1);
  const Eigen::VectorXd var = Eigen::VectorXd::Random(n, 1).array().square();
  const MarginalDistribution dist(mean, var);
  const auto dist_size = cast::to_index(dist.size());
  const Eigen::SparseMatrix<double> mat =
      Eigen::MatrixXd::Random(dist_size - 1, dist_size).sparseView();

  const auto transformed_distribution = (mat * dist).joint();
  const JointDistribution alternate = mat * dist;
  EXPECT_EQ(alternate, transformed_distribution);

  EXPECT_EQ(transformed_distribution.mean, mat * dist.mean);
  const Eigen::MatrixXd cov = mat * dist.covariance * mat.transpose();
  EXPECT_LT((transformed_distribution.covariance - cov).norm(), 1e-8);
};

TYPED_TEST_P(DistributionTest, test_multiply_with_vector) {

  TypeParam test_case;
  const auto dist = test_case.create();
  Eigen::VectorXd vector = Eigen::VectorXd::Random(cast::to_index(dist.size()));

  const auto transformed_distribution = (vector.transpose() * dist).joint();
  const JointDistribution alternate = vector.transpose() * dist;
  EXPECT_EQ(alternate, transformed_distribution);

  double expected_mean = vector.dot(dist.mean);
  double expected_variance = vector.dot(dist.covariance * vector);
  EXPECT_EQ(transformed_distribution.mean[0], expected_mean);
  EXPECT_NEAR(transformed_distribution.covariance(0, 0), expected_variance,
              1e-8);
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

TYPED_TEST_P(DistributionTest, test_equal) {
  TypeParam test_case;
  {
    const auto dist = test_case.create();
    const auto exact_copy = dist;
    EXPECT_TRUE(dist == exact_copy);
  }

  {
    const auto dist = test_case.create();
    auto perturbed = dist;
    perturbed.mean += Eigen::VectorXd::Constant(dist.mean.size(), 1e-12);
    EXPECT_FALSE(dist == perturbed);
  }
};

TYPED_TEST_P(DistributionTest, test_approximately_equal) {
  TypeParam test_case;
  {
    const auto dist = test_case.create();
    const auto exact_copy = dist;
    // Default epsilon is 1e-3
    EXPECT_TRUE(dist.approximately_equal(exact_copy));
  }

  {
    const auto dist = test_case.create();
    const long size = dist.mean.size();
    auto perturbed = dist;
    perturbed.mean += Eigen::VectorXd::Constant(size, 1e-4);
    // Default epsilon is 1e-3, so this should pass
    EXPECT_TRUE(dist.approximately_equal(perturbed));
  }

  {
    const auto dist = test_case.create();
    const auto exact_copy = dist;
    EXPECT_TRUE(dist.approximately_equal(exact_copy, 1e-12));
  }

  {
    const auto dist = test_case.create();
    const long size = dist.mean.size();
    auto perturbed = dist;
    perturbed.mean += Eigen::VectorXd::Constant(size, 1e-4);
    EXPECT_TRUE(dist.approximately_equal(perturbed, 1e-3));
  }

  {
    const auto dist = test_case.create();
    const long size = dist.mean.size();
    auto perturbed = dist;
    perturbed.mean += Eigen::VectorXd::Constant(size, 1e-3);
    EXPECT_FALSE(dist.approximately_equal(perturbed, 1e-4));
  }
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

TYPED_TEST_P(DistributionTest, test_operator_indexing) {

  TypeParam test_case;
  const auto dist = test_case.create();

  for (std::size_t idx = 0; idx < dist.size(); ++idx) {
    MarginalDistribution m = dist[idx];

    EXPECT_EQ(m.mean[0], dist.mean[cast::to_index(idx)]);
    EXPECT_EQ(m.get_diagonal(0), dist.get_diagonal(cast::to_index(idx)));
  }
};

REGISTER_TYPED_TEST_SUITE_P(DistributionTest, test_subset,
                            test_multiply_with_matrix_joint,
                            test_multiply_with_matrix_marginal,
                            test_multiply_with_sparse_matrix_joint,
                            test_multiply_with_sparse_matrix_marginal,
                            test_multiply_with_vector, test_multiply_by_scalar,
                            test_equal, test_approximately_equal, test_add,
                            test_subtract, test_operator_indexing);

Eigen::VectorXd arange(Eigen::Index k = 5) {
  Eigen::VectorXd mean(k);
  for (Eigen::Index i = 0; i < k; i++) {
    mean[i] = cast::to_double(i);
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

INSTANTIATE_TYPED_TEST_SUITE_P(Albatross, DistributionTest, ToTest);

} // namespace albatross
