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

#include <albatross/Evaluation>
#include <albatross/utils/RandomUtils>

namespace albatross {

TEST(test_random_utils, randint_without_replacement) {

  int iterations = 10;
  int k = 6;

  std::default_random_engine gen;

  for (int i = 0; i < iterations; i++) {
    for (int n = 0; n <= k; n++) {
      const auto inds = randint_without_replacement(n, i, i + k, gen);
      for (const auto &j : inds) {
        EXPECT_LE(j, i + k);
        EXPECT_GE(j, i);
      }
    }
  }
}

TEST(test_random_utils, randint_without_replacement_full_set) {
  std::default_random_engine gen;
  const auto inds = randint_without_replacement(10, 0, 9, gen);
  EXPECT_EQ(inds.size(), 10);
  std::set<std::size_t> unique_inds(inds.begin(), inds.end());
  EXPECT_EQ(unique_inds.size(), inds.size());
}

TEST(test_random_utils, test_random_multivariate_normal_1d) {
  std::default_random_engine gen(2012);

  double expected_mean = 5.;
  Eigen::VectorXd mean(1);
  mean << expected_mean;
  double expected_variance = 3.;
  Eigen::MatrixXd cov(1, 1);
  cov << expected_variance;

  Eigen::Index k = 10000;
  Eigen::MatrixXd samples(k, 1);
  for (Eigen::Index i = 0; i < k; ++i) {
    samples.row(i) = random_multivariate_normal(mean, cov, gen);
  }

  EXPECT_NEAR(samples.mean(), expected_mean, 0.1);
  EXPECT_NEAR(standard_deviation(samples), std::sqrt(expected_variance), 0.1);
}

TEST(test_random_utils, test_random_covariance_matrix) {
  std::default_random_engine gen(2012);

  std::uniform_int_distribution<Eigen::Index> size_distribution(1, 20);

  Eigen::Index k = 100;
  for (Eigen::Index i = 0; i < k; ++i) {
    Eigen::Index n = size_distribution(gen);
    const auto cov = random_covariance_matrix(n, gen);
    EXPECT_GE(cov.eigenvalues().real().maxCoeff(),
              std::numeric_limits<double>::epsilon());

    EXPECT_LE((cov - cov.transpose()).norm(), 1e-6);
  }
}

TEST(test_random_utils, test_random_multivariate_normal) {
  // Draw a bunch of samples from various sizes of
  // multivariate normal distributions and make sure the
  // results follow the expected chi squared distributions.
  std::default_random_engine gen(2012);

  std::uniform_int_distribution<Eigen::Index> size_distribution(1, 20);

  Eigen::Index k = 1000;
  std::vector<double> cdfs;
  for (Eigen::Index i = 0; i < k; ++i) {
    Eigen::Index n = size_distribution(gen);
    const auto cov = random_covariance_matrix(n, gen);
    Eigen::VectorXd mean(n);
    gaussian_fill(mean, gen);

    const auto sample = random_multivariate_normal(mean, cov, gen);

    cdfs.push_back(chi_squared_cdf(sample - mean, cov));
  }

  EXPECT_LT(uniform_ks_test(cdfs), 0.05);
}

} // namespace albatross
