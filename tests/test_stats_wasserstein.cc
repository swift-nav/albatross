/*
 * Copyright (C) 2019 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <albatross/Common>
#include <albatross/Distribution>
#include <albatross/Evaluation>
#include <albatross/Stats>
#include <albatross/utils/RandomUtils>

#include <albatross/src/utils/eigen_utils.hpp>
#include <gtest/gtest.h>

namespace albatross {

template <typename RandomNumberGenerator>
JointDistribution random_distribution(Eigen::Index dimension,
                                      RandomNumberGenerator &gen) {
  const auto covariance = random_covariance_matrix(dimension, gen);
  Eigen::VectorXd mean(dimension);
  gaussian_fill(mean, gen);

  return {mean, covariance};
}

static constexpr const double cPoorConditioning{1.e-8};

template <typename RandomNumberGenerator>
JointDistribution
ill_conditioned_random_distribution(Eigen::Index dimension,
                                    RandomNumberGenerator &gen) {
  bool gave_tiny_eigenvalue = false;
  auto dist = [gave_tiny = gave_tiny_eigenvalue](auto &rng) mutable {
    if (!gave_tiny) {
      gave_tiny = true;
      return cPoorConditioning;
    }
    return std::gamma_distribution<double>(2., 2.)(rng);
  };
  const auto covariance = random_covariance_matrix(dimension, dist, gen);
  Eigen::VectorXd mean(dimension);
  gaussian_fill(mean, gen);

  return {mean, covariance};
}

static constexpr Eigen::Index cDistributionDimension = 30;
static constexpr std::size_t cNumIterations = 10000;

// The Wasserstein distance between a distribution and itself should
// be zero to within numerical precision.
TEST(test_stats, test_wasserstein_zero) {
  std::default_random_engine gen(2222);

  for (std::size_t iter = 0; iter < cNumIterations; ++iter) {
    const Eigen::Index dimension = std::uniform_int_distribution<Eigen::Index>(
        1, cDistributionDimension)(gen);
    const auto dist = random_distribution(dimension, gen);

    EXPECT_LT(distance::wasserstein_2(dist, dist),
              1.e-12 * dist.covariance.trace() +
                  1.e-12 * dist.mean.squaredNorm());
  }
}

// The Wasserstein distance between a distribution and itself should
// be zero to within numerical precision.  For ill-conditioned
// covariances, we relax this definition of precision, but we still
// make sure we return a finite value.  (This case occurred a number
// of times in practice, but it was not covered by the vanilla MVN
// generator.)
TEST(test_stats, test_wasserstein_zero_ill_conditioned) {
  std::default_random_engine gen(2222);

  for (std::size_t iter = 0; iter < cNumIterations; ++iter) {
    const Eigen::Index dimension = std::uniform_int_distribution<Eigen::Index>(
        1, cDistributionDimension)(gen);
    const auto dist = ill_conditioned_random_distribution(dimension, gen);

    const auto w2 = distance::wasserstein_2(dist, dist);
    EXPECT_TRUE(std::isfinite(w2));

    EXPECT_LT(w2, 2 * cPoorConditioning * dist.covariance.trace() +
                      2 * cPoorConditioning * dist.mean.squaredNorm());
  }
}

// The Wasserstein distance between two distributions should aways be
// nonnegative.
TEST(test_stats, test_wasserstein_nonnegative) {
  std::default_random_engine gen(2222);

  for (std::size_t iter = 0; iter < cNumIterations; ++iter) {
    const Eigen::Index dimension = std::uniform_int_distribution<Eigen::Index>(
        1, cDistributionDimension)(gen);
    const auto dist_a = random_distribution(dimension, gen);
    const auto dist_b = random_distribution(dimension, gen);

    EXPECT_GE(distance::wasserstein_2(dist_a, dist_b), 0);
  }
}

// If two distributions differ only in their mean, then the
// Wasserstein 2-distance should differ according to the square of the
// distance between means (i.e. the Wasserstein distance has the same
// units as the mean).
TEST(test_stats, test_wasserstein_shift) {
  std::default_random_engine gen(2222);

  for (std::size_t iter = 0; iter < cNumIterations; ++iter) {
    const Eigen::Index dimension = std::uniform_int_distribution<Eigen::Index>(
        1, cDistributionDimension)(gen);
    auto dist_a = random_distribution(dimension, gen);
    auto dist_b = dist_a;
    gaussian_fill(dist_b.mean, gen);

    const double distance = distance::wasserstein_2(dist_a, dist_b);

    const double mean_distance = (dist_a.mean - dist_b.mean).squaredNorm();

    EXPECT_LT(distance - mean_distance, 1.e-10);
  }
}

// If we inflate the covariance of the distribution, the Wasserstein
// distance to the original distribution should increase.
TEST(test_stats, test_wasserstein_grows_with_covariance) {
  std::default_random_engine gen(2222);

  for (std::size_t iter = 0; iter < cNumIterations; ++iter) {
    const Eigen::Index dimension = std::uniform_int_distribution<Eigen::Index>(
        1, cDistributionDimension)(gen);
    auto dist_a = random_distribution(dimension, gen);
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> cov_eigs(
        dist_a.covariance);

    auto dist_b = dist_a;
    dist_b.covariance =
        cov_eigs.eigenvectors() *
        (cov_eigs.eigenvalues().array() * 2).matrix().asDiagonal() *
        cov_eigs.eigenvectors().transpose();

    auto dist_c = dist_a;
    dist_c.covariance =
        cov_eigs.eigenvectors() *
        (cov_eigs.eigenvalues().array() * 4).matrix().asDiagonal() *
        cov_eigs.eigenvectors().transpose();

    const double distance_ab = distance::wasserstein_2(dist_a, dist_b);
    const double distance_ac = distance::wasserstein_2(dist_a, dist_c);

    EXPECT_GT(distance_ac, distance_ab);
  }
}

} // namespace albatross
