/*
 * Copyright (C) 2025 Swift Navigation Inc.
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
#include <random>
#include <gtest/gtest.h>

namespace albatross {

static constexpr const std::size_t cRandomIterations{1000};

// This test is kind of annoying because it's hard to get a good
// analytic bound on the cosine series, which converges very slowly
// and non-monotonically.  This is the reason there is so much fiddly
// calculation of how far to go in the series and what the error bound
// should be.
TEST(SphericalCovariance, Matern12LongitudinalMatchesSeries) {
  static constexpr const double cDegreesOfFreedom{0.5};
  std::default_random_engine rng{22};
  std::uniform_int_distribution<std::size_t> terms_dist(
      constant::cDefaultLongitudinalMaternSeriesLength, 1000);
  std::uniform_real_distribution<double> log_length_scale_dist(-3, 0);
  std::uniform_real_distribution<double> distance_dist(0., M_PI);
  for (std::size_t iteration = 0; iteration < cRandomIterations; ++iteration) {
    double length_scale = std::pow(10, log_length_scale_dist(rng));
    double distance = distance_dist(rng);

    // As kappa / distance grows, we boost the number of terms in the
    // series because of how slowly it converges.
    const double sensitivity = std::log10(distance * length_scale);
    std::size_t pad = 0;
    if (sensitivity < 0) {
      pad = static_cast<std::size_t>(std::pow(10, -sensitivity + 1));
    }
    std::size_t terms =
        terms_dist(rng) +
        // Have to make sure we at least go bigger than \kappa by a
        // bit because of the denominator of the series
        static_cast<std::size_t>(
            detail::length_scale_spde_to_gp(length_scale, cDegreesOfFreedom));
    terms += pad;

    const auto [series, err] = matern_longitudinal_covariance_series(
        distance, length_scale, cDegreesOfFreedom, terms);
    EXPECT_TRUE(std::isfinite(series));
    const double closed =
        matern_12_longitudinal_covariance(distance, length_scale);
    EXPECT_TRUE(std::isfinite(closed));
    // This is not a great definition for error because the series is
    // rotational (cosine), but once m gets well above kappa it's a
    // convenient guess at a bound.
    const double threshold = std::fabs(err) * 10;
    EXPECT_NEAR(series, closed, threshold)
        << "err: " << err << "\nlength_scale = " << length_scale
        << "\nterms = " << terms << "\ndistance = " << distance;
  }
}

TEST(SphericalCovariance, Matern32LongitudinalMatchesSeries) {
  static constexpr const double cDegreesOfFreedom{1.5};
  std::default_random_engine rng{22};
  std::uniform_int_distribution<std::size_t> terms_dist(
      constant::cDefaultLongitudinalMaternSeriesLength, 1000);
  std::uniform_real_distribution<double> log_length_scale_dist(-3, 0);
  std::uniform_real_distribution<double> distance_dist(0., M_PI);
  for (std::size_t iteration = 0; iteration < cRandomIterations; ++iteration) {
    double length_scale = std::pow(10, log_length_scale_dist(rng));
    double distance = distance_dist(rng);

    // As kappa / distance grows, we boost the number of terms in the
    // series because of how slowly it converges.
    const double sensitivity = std::log10(distance * length_scale);
    std::size_t pad = 0;
    if (sensitivity < 0) {
      pad = static_cast<std::size_t>(std::pow(10, -sensitivity + 1));
    }
    std::size_t terms =
        terms_dist(rng) +
        // Have to make sure we at least go bigger than \kappa by a
        // bit because of the denominator of the series
        static_cast<std::size_t>(
            detail::length_scale_spde_to_gp(length_scale, cDegreesOfFreedom));
    terms += pad;

    const auto [series, err] = matern_longitudinal_covariance_series(
        distance, length_scale, cDegreesOfFreedom, terms);
    EXPECT_TRUE(std::isfinite(series));
    const double closed =
        matern_32_longitudinal_covariance(distance, length_scale);
    EXPECT_TRUE(std::isfinite(closed));
    // This is not a great definition for error because the series is
    // rotational (cosine), but once m gets well above kappa it's a
    // convenient guess at a bound.
    const double threshold = std::fabs(err) * 50;
    EXPECT_NEAR(series, closed, threshold)
        << "err: " << err << "\nlength_scale = " << length_scale
        << "\nterms = " << terms << "\ndistance = " << distance;
  }
}

TEST(SphericalCovariance, Matern52LongitudinalMatchesSeries) {
  static constexpr const double cDegreesOfFreedom{2.5};
  std::default_random_engine rng{22};
  std::uniform_int_distribution<std::size_t> terms_dist(
      constant::cDefaultLongitudinalMaternSeriesLength, 1000);
  std::uniform_real_distribution<double> log_length_scale_dist(-3, 0);
  std::uniform_real_distribution<double> distance_dist(0., M_PI);
  for (std::size_t iteration = 0; iteration < cRandomIterations; ++iteration) {
    double length_scale = std::pow(10, log_length_scale_dist(rng));
    double distance = distance_dist(rng);

    // As kappa / distance grows, we boost the number of terms in the
    // series because of how slowly it converges.
    const double sensitivity = std::log10(distance * length_scale);
    std::size_t pad = 0;
    if (sensitivity < 0) {
      pad = static_cast<std::size_t>(std::pow(10, -sensitivity + 2));
    }
    std::size_t terms =
        terms_dist(rng) +
        // Have to make sure we at least go bigger than \kappa by a
        // bit because of the denominator of the series
        static_cast<std::size_t>(
            detail::length_scale_spde_to_gp(length_scale, cDegreesOfFreedom));
    terms += pad;

    const auto [series, _err] = matern_longitudinal_covariance_series(
        distance, length_scale, cDegreesOfFreedom, terms);
    const auto [series2, _err2] = matern_longitudinal_covariance_series(
        distance, length_scale, cDegreesOfFreedom, terms * 2);
    EXPECT_TRUE(std::isfinite(series));
    const double closed =
        matern_52_longitudinal_covariance(distance, length_scale);
    EXPECT_TRUE(std::isfinite(closed));
    const double err = std::fabs(series - closed);
    const double err2 = std::fabs(series2 - closed);
    EXPECT_LE(err2, err + 1e-10)
        << "err_violation = " << std::fabs(err - err2)
        << "\nlength_scale = " << length_scale << "\nterms = " << terms
        << "\ndistance = " << distance;
  }
}

} // namespace albatross
