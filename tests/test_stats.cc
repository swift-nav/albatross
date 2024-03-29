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

} // namespace albatross
