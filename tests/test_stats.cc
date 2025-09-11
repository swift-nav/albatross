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

TEST(test_stats, test_gaussian_pdf) {
  // These examples were generated in python using scipy.stats.norm.pdf
  std::vector<double> test_xs = {
      -1.49529605, -0.35674996, -1.19464126, 0.7431096,   0.94945083,
      -0.06465424, -0.36805315, -1.38905131, -1.56751365, 1.8271551};
  std::vector<double> test_variances = {
      3.39311978, 0.55516885, 0.72540077, 0.05034394, 0.16184329,
      2.31795834, 0.00988035, 0.11177149, 0.77043322, 2.84884525};
  std::vector<double> expected = {
      1.55783121e-01, 4.77438315e-01, 1.75146437e-01, 7.38065599e-03,
      6.12161951e-02, 2.61797595e-01, 4.23016986e-03, 2.12923882e-04,
      9.22586650e-02, 1.31554532e-01};

  for (std::size_t i = 0; i < test_xs.size(); ++i) {
    EXPECT_NEAR(gaussian::pdf(test_xs[i], test_variances[i]), expected[i],
                1e-6);
    EXPECT_NEAR(gaussian::log_pdf(test_xs[i], test_variances[i]),
                std::log(expected[i]), 1e-6);
  }

  EXPECT_LT(gaussian::pdf(-100., 1.), 1e-12);
  EXPECT_LT(gaussian::pdf(100., 1.), 1e-12);
  EXPECT_LT(gaussian::pdf(1., 1e-6), 1e-12);
  EXPECT_LT(gaussian::pdf(1e12, 1e8), 1e-12);
}

TEST(test_stats, test_chi_squared) {
  EXPECT_LT(fabs(chi_squared_cdf(3.84, 1) - 0.95), 1e-4);
  EXPECT_LT(fabs(chi_squared_cdf(10.83, 1) - 0.999), 1e-4);

  EXPECT_LT(fabs(chi_squared_cdf(5.99, 2) - 0.95), 1e-4);
  EXPECT_LT(fabs(chi_squared_cdf(13.82, 2) - 0.999), 1e-4);

  EXPECT_LT(fabs(chi_squared_cdf(11.07, 5) - 0.95), 1e-4);
  EXPECT_LT(fabs(chi_squared_cdf(15.09, 5) - 0.99), 1e-4);

  EXPECT_LT(fabs(chi_squared_cdf(9.260, 23) - 0.005), 1e-5);
  EXPECT_LT(fabs(chi_squared_cdf(38.932, 21) - 0.99), 1e-5);
  EXPECT_LT(fabs(chi_squared_cdf(96.578, 80) - 0.9), 1e-5);
  EXPECT_LT(fabs(chi_squared_cdf(70.065, 100) - 0.01), 1e-5);

  EXPECT_EQ(chi_squared_cdf(0., 0.), 1.);
  EXPECT_LT(chi_squared_cdf(0., 1), 1e-6);
  EXPECT_LT(chi_squared_cdf(0., 2), 1e-6);
  EXPECT_LT(chi_squared_cdf(0., 10), 1e-6);
  EXPECT_LT(chi_squared_cdf(0., 100.), 1e-6);

  EXPECT_LT(fabs(chi_squared_cdf(1.e-4, 0.) - 1.), 1e-4);
  EXPECT_LT(fabs(chi_squared_cdf(1., 0.) - 1.), 1e-4);
  EXPECT_LT(fabs(chi_squared_cdf(1000, 100) - 1.), 1e-4);
  EXPECT_LT(fabs(chi_squared_cdf(10000, 100) - 1.), 1e-4);
  EXPECT_LT(fabs(chi_squared_cdf(100000, 100) - 1.), 1e-4);
  EXPECT_LT(fabs(chi_squared_cdf(INFINITY, 1) - 1.), 1e-4);

  EXPECT_TRUE(std::isnan(chi_squared_cdf(-1e-6, 0)));
  EXPECT_TRUE(std::isnan(chi_squared_cdf(-1e-6, 1)));
  EXPECT_TRUE(std::isnan(chi_squared_cdf(-1e-6, 100)));

  EXPECT_TRUE(std::isnan(chi_squared_cdf(NAN, 0)));
  EXPECT_TRUE(std::isnan(chi_squared_cdf(NAN, 1)));
}

TEST(test_stats, test_uniform_ks) {
  std::default_random_engine gen(2012);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);

  std::vector<double> samples;
  std::size_t n = 1000;
  for (std::size_t i = 0; i < n; ++i) {
    samples.emplace_back(uniform(gen));
  }

  EXPECT_LT(uniform_ks_test(samples), 0.05);
}

TEST(test_stats, test_chi_squared_cdf) {
  Eigen::Index k = 5;
  std::default_random_engine gen(2012);
  std::size_t iterations = 1000;
  std::vector<double> cdfs(iterations);
  for (std::size_t i = 0; i < iterations; ++i) {
    const auto covariance = random_covariance_matrix(k, gen);
    const auto sample = random_multivariate_normal(covariance, gen);
    // Collect all the cdfs
    cdfs[i] = chi_squared_cdf(sample, covariance);
  }

  EXPECT_LT(*std::min_element(cdfs.begin(), cdfs.end()), 0.1);
  EXPECT_GT(*std::max_element(cdfs.begin(), cdfs.end()), 0.9);
  double ks = uniform_ks_test(cdfs);
  EXPECT_LT(ks, 0.05);
}

TEST(test_stats, test_chi_squared_cdf_bounds) {
  for (std::size_t dof = 1; dof < 50; ++dof) {
    const double a = 0.5 * cast::to_double(dof);
    const auto bounds =
        details::incomplete_gamma_quadrature_bounds(a, 100 * sqrt(a));

    // low/high for the chi squared will be double low/high for the incomplete
    // gamma
    const double lowest = 2. * bounds.first;
    const double highest = 2. * bounds.second;

    // The bounds provided should correspond to the z argument which corresponds
    // to near zero and near one evaluations of the incomplete gamma
    EXPECT_NEAR(chi_squared_cdf(lowest, dof), 0., 1e-8);
    EXPECT_NEAR(chi_squared_cdf(highest, dof), 1., 1e-8);
  }
}

TEST(test_stats, test_chi_squared_cdf_monotonic) {
  Eigen::Index k = 5;

  std::default_random_engine gen(2012);

  const auto covariance = random_covariance_matrix(k, gen);
  const auto sample = random_multivariate_normal(covariance, gen);

  std::size_t iterations = 50;
  ASSERT_LT(chi_squared_cdf(sample, covariance), 1.);
  double previous = -std::numeric_limits<double>::epsilon();
  // Evaluate the cdf while scaling the sampled vector by increasingly
  // large amounts, the cdf should also continue increasing.
  for (std::size_t i = 0; i < iterations; ++i) {
    double scale = cast::to_double(i) / 5.;
    double cdf = chi_squared_cdf(scale * sample, covariance);
    EXPECT_LE(previous, cdf);
    previous = cdf;
  }
}

TEST(test_stats, test_chi_squared_cdf_monotonic_1d) {
  std::size_t iterations = 500;
  double previous = -std::numeric_limits<double>::epsilon();
  // Evaluate the cdf with one dimension iteratively increasing to a
  // value equivalent to 50 standard deviations from the mean. We
  // explicitly test that high because of known instabilities in the
  // tails of the incomplete gamma function.
  for (std::size_t i = 0; i < iterations; ++i) {
    double x = cast::to_double(i) / 50.;
    double cdf = chi_squared_cdf(x * x, 1);
    EXPECT_LE(previous, cdf);
    previous = cdf;
  }
}

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
