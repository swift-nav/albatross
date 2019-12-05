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

#ifndef ALBATROSS_TESTS_TEST_UTILS_H
#define ALBATROSS_TESTS_TEST_UTILS_H

#include <albatross/Core>
#include <albatross/GP>

#include "mock_model.h"

namespace albatross {

static inline auto make_toy_sine_data(const double a = 5., const double b = 10.,
                                      const double sigma = 0.1,
                                      const std::size_t n = 10) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  gen.seed(3);
  std::normal_distribution<> d{0., sigma};
  std::vector<double> features;
  Eigen::VectorXd targets(n);

  for (std::size_t i = 0; i < n; i++) {
    double x = static_cast<double>(i);
    features.push_back(x);
    targets[i] = a * sin(x * b) + d(gen);
  }

  return RegressionDataset<double>(features, targets);
}

static inline auto make_toy_linear_data(const double a = 5.,
                                        const double b = 1.,
                                        const double sigma = 0.1,
                                        const std::size_t n = 10) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  gen.seed(3);
  std::normal_distribution<> d{0., sigma};
  std::vector<double> features;
  Eigen::VectorXd targets(n);

  for (std::size_t i = 0; i < n; i++) {
    double x = static_cast<double>(i);
    features.push_back(x);
    targets[i] = a + x * b + d(gen);
  }

  return RegressionDataset<double>(features, targets);
}

class MockParameterHandler : public ParameterHandlingMixin {
public:
  MockParameterHandler(const ParameterStore &params)
      : ParameterHandlingMixin(params){};
};

class TestParameterHandler : public ParameterHandlingMixin {
public:
  TestParameterHandler() : ParameterHandlingMixin() {
    params_ = {{"A", 1.}, {"B", 2.}};
  };
};

static inline void
expect_parameter_vector_equal(const std::vector<ParameterValue> &x,
                              const std::vector<ParameterValue> &y) {
  for (std::size_t i = 0; i < x.size(); i++) {
    EXPECT_DOUBLE_EQ(x[i], y[i]);
  }
  EXPECT_EQ(x.size(), y.size());
}

static inline auto
make_heteroscedastic_toy_linear_data(const double a = 5., const double b = 1.,
                                     const double sigma = 0.1) {

  std::random_device rd{};
  std::mt19937 gen{rd()};
  gen.seed(7);
  std::normal_distribution<> d{0., 1.};

  RegressionDataset<double> dataset = make_toy_linear_data(a, b, sigma);

  auto targets = dataset.targets.mean;
  auto variance = Eigen::VectorXd(targets.size());

  for (int i = 0; i < targets.size(); i++) {
    double std = 0.1 * fabs(dataset.features[i]);
    targets[i] += std * d(gen);
    variance[i] = sigma * sigma + std * std;
  }

  auto diag_matrix = variance.asDiagonal();

  MarginalDistribution target_dist(targets, diag_matrix);

  return RegressionDataset<double>(dataset.features, target_dist);
}

inline auto toy_covariance_function() {
  using Noise = IndependentNoise<double>;
  SquaredExponential<EuclideanDistance> squared_exponential(100., 100.);
  IndependentNoise<double> noise = Noise(0.1);
  auto covariance = squared_exponential + noise;
  return covariance;
}

/*
 * Here we create data and a model that will make it easier to test
 * that models using the model_adapter.h interface work.
 */
struct AdaptedFeature {
  double value;
};

static inline auto make_adapted_toy_linear_data(const double a = 5.,
                                                const double b = 1.,
                                                const double sigma = 0.1,
                                                const std::size_t n = 10) {
  const auto dataset = make_toy_linear_data(a, b, sigma, n);

  std::vector<AdaptedFeature> adapted_features;
  for (const auto &f : dataset.features) {
    adapted_features.push_back({f});
  }

  RegressionDataset<AdaptedFeature> adapted_dataset(adapted_features,
                                                    dataset.targets);
  return adapted_dataset;
}

inline auto random_spherical_points(std::size_t n = 10, double radius = 1.,
                                    int seed = 5) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  gen.seed(seed);

  std::uniform_real_distribution<double> rand_lon(0., 2 * M_PI);
  std::uniform_real_distribution<double> rand_lat(-M_PI / 2., M_PI / 2.);

  std::vector<Eigen::VectorXd> points;

  for (std::size_t i = 0; i < n; i++) {
    const double lon = rand_lon(gen);
    const double lat = rand_lat(gen);
    Eigen::VectorXd x(3);
    // Convert the spherical coordinates to X,Y,Z
    x << cos(lat) * cos(lon) * radius, cos(lat) * sin(lon) * radius,
        sin(lat) * radius;
    points.push_back(x);
  }
  return points;
}

template <typename CovarianceFunction, typename FeatureType>
void expect_state_space_representation_quality(
    const CovarianceFunction &cov_func,
    const std::vector<FeatureType> &features, double threshold) {

  const auto ssr_features = cov_func.state_space_representation(features);

  const Eigen::MatrixXd ssr_cov = cov_func(ssr_features);
  const Eigen::MatrixXd cross_cov = cov_func(features, ssr_features);
  const Eigen::MatrixXd full_cov = cov_func(features);

  // This is "the covariance explained by the state space representation".
  // In other words, it's the posterior covariance of the observations
  // if you had fit using the state space representation.
  // If the state space representation fully explains the observations
  // then we know we've formed a (relatively) loss less representation.
  const Eigen::MatrixXd explained =
      cross_cov * ssr_cov.ldlt().solve(cross_cov.transpose());
  const Eigen::MatrixXd posterior = full_cov - explained;
  EXPECT_LT(posterior.norm() / full_cov.norm(), threshold);
}

} // namespace albatross

#endif
