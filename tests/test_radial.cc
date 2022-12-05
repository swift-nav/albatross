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
#include <albatross/CovarianceFunctions>
#include <chrono>

#include "test_utils.h"

#include "uninlineable.h"

namespace albatross {

inline auto random_spherical_dataset(std::vector<Eigen::VectorXd> points,
                                     int seed = 7) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  gen.seed(seed);
  std::normal_distribution<> d{0., 0.1};

  Eigen::VectorXd targets(static_cast<Eigen::Index>(points.size()));

  auto spherical_function = [](Eigen::VectorXd &x) {
    return x[0] * x[1] + x[1] * x[2] + x[3];
  };

  for (std::size_t i = 0; i < points.size(); i++) {
    targets[static_cast<Eigen::Index>(i)] = spherical_function(points[i]);
  }

  return RegressionDataset<Eigen::VectorXd>(points, targets);
}

TEST(test_radial, test_is_positive_definite) {
  const auto points = random_spherical_points(100);

  const Exponential<AngularDistance> term(2 * M_PI);

  const Eigen::MatrixXd cov = term(points);

  EXPECT_GE(cov.eigenvalues().real().array().minCoeff(), 0.);
}

static inline auto random_generator(std::size_t seed) {
  std::random_device random_device{};
  std::mt19937 generator{random_device()};
  generator.seed(seed);
  std::uniform_real_distribution<> dist{-1., 1.};
  return [dist, generator]() mutable { return dist(generator); };
}

static inline auto random_index_generator(std::size_t seed, Eigen::Index size) {
  std::random_device random_device{};
  std::mt19937 generator{random_device()};
  generator.seed(seed);
  std::uniform_int_distribution<Eigen::Index> dist{0, size - 1};
  return [dist, generator]() mutable { return dist(generator); };
}

TEST(test_radial, test_block_full_matches_single) {
  constexpr double length_scale = 0.5;
  constexpr double sigma = 0.1;
  constexpr double tolerance = 1e-10;
  constexpr std::size_t iterations = 100;

  const albatross::SquaredExponential<albatross::EuclideanDistance> cov{
      length_scale, sigma};

  const auto gen = random_generator(22);
  std::vector<double> x(1024);

  for (std::size_t i = 0; i < iterations; ++i) {
    std::generate(x.begin(), x.end(), gen);

    const Eigen::MatrixXd cov_loop = compute_covariance_matrix(cov, x);
    const Eigen::MatrixXd cov_block_full =
        albatross::block_squared_exponential_full(x, x, length_scale, sigma);
    const Eigen::MatrixXd cov_block_full_rows =
        albatross::block_squared_exponential_full_rows(x, x, length_scale,
                                                       sigma);
    // const auto begin = std::chrono::steady_clock::now();
    const Eigen::MatrixXd cov_block_full_vecs =
        albatross::block_squared_exponential_full_vecs(x, x, length_scale,
                                                       sigma);
    const Eigen::MatrixXd cov_block_full_vecs_uninlineable =
        albatross::block_squared_exponential_full_vecs_uninlineable(
            x, x, length_scale, sigma);
    // const auto end = std::chrono::steady_clock::now();

    // std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end -
    //                                                                   begin)
    //                  .count()
    //           << "ns" << std::endl;

    const Eigen::MatrixXd cov_block_columns =
        albatross::block_squared_exponential_columns(x, x, length_scale, sigma);
    const Eigen::MatrixXd cov_block_column_major =
        albatross::squared_exponential_column_major(x, x, length_scale, sigma);
    const Eigen::MatrixXd cov_block_row_major =
        albatross::squared_exponential_row_major(x, x, length_scale, sigma);

    EXPECT_LE((cov_loop - cov_block_full).array().abs().maxCoeff(), tolerance);
    EXPECT_LE((cov_loop - cov_block_full_rows).array().abs().maxCoeff(),
              tolerance);
    EXPECT_LE((cov_loop - cov_block_full_vecs).array().abs().maxCoeff(),
              tolerance);
    EXPECT_LE(
        (cov_loop - cov_block_full_vecs_uninlineable).array().abs().maxCoeff(),
        tolerance);
    EXPECT_LE((cov_loop - cov_block_columns).array().abs().maxCoeff(),
              tolerance);
    EXPECT_LE((cov_loop - cov_block_column_major).array().abs().maxCoeff(),
              tolerance);
    EXPECT_LE((cov_loop - cov_block_row_major).array().abs().maxCoeff(),
              tolerance);
  }

  constexpr std::size_t ysize = 8192;
  std::vector<double> y(ysize);
  std::generate(y.begin(), y.end(), gen);
  auto get_random_idx =
      random_index_generator(static_cast<std::size_t>(nearbyint(y[0])), ysize);
  const auto begin = std::chrono::steady_clock::now();
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(iterations); ++i) {
    const Eigen::MatrixXd cov_block_full_vecs =
        albatross::block_squared_exponential_full_vecs_uninlineable(
            y, y, length_scale, sigma);
    const auto random_idx0 = get_random_idx();
    const auto random_idx1 = get_random_idx();
    y[random_idx0] += cov_block_full_vecs(
        random_idx1, static_cast<Eigen::Index>(ysize) - random_idx0 - 1);
    y[random_idx1] -= cov_block_full_vecs(
        random_idx0, static_cast<Eigen::Index>(ysize) - random_idx1 - 1);
  }
  const auto end = std::chrono::steady_clock::now();

  std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                       .count() /
                   iterations
            << "ns" << std::endl;
}

class SquaredExponentialSSRTest {
 public:
  std::vector<double> features() const { return linspace(0., 10., 101); }

  auto covariance_function() const {
    SquaredExponential<EuclideanDistance> cov(5., 1.);
    return cov;
  }

  double get_tolerance() const { return 1e-12; }
};

class ExponentialSSRTest {
 public:
  std::vector<double> features() const { return linspace(0., 10., 11); }

  auto covariance_function() const {
    Exponential<EuclideanDistance> cov(5., 1.);
    return cov;
  }

  double get_tolerance() const { return 1e-2; }
};

class ExponentialAngularSSRTest {
 public:
  std::vector<double> features() const { return linspace(0., M_2_PI, 11); }

  auto covariance_function() const {
    Exponential<EuclideanDistance> cov(M_PI_4, 1.);
    return cov;
  }

  double get_tolerance() const { return 1e-2; }
};

template <typename T>
class CovarianceStateSpaceTester : public ::testing::Test {
 public:
  T test_case;
};

using StateSpaceTestCases =
    ::testing::Types<SquaredExponentialSSRTest, ExponentialSSRTest,
                     ExponentialAngularSSRTest>;
TYPED_TEST_CASE(CovarianceStateSpaceTester, StateSpaceTestCases);

TYPED_TEST(CovarianceStateSpaceTester, test_state_space_representation) {
  const auto xs = this->test_case.features();

  const auto cov_func = this->test_case.covariance_function();

  expect_state_space_representation_quality(cov_func, xs,
                                            this->test_case.get_tolerance());
}

}  // namespace albatross
