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

#include "Core"

//#include "GP"

#include <random>

#include "mock_model.h"

namespace albatross {

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


//
// static inline auto
// make_heteroscedastic_toy_linear_data(const double a = 5., const double b =
// 1.,
//                                     const double sigma = 0.1) {
//
//  std::random_device rd{};
//  std::mt19937 gen{rd()};
//  gen.seed(7);
//  std::normal_distribution<> d{0., 1.};
//
//  RegressionDataset<double> dataset = make_toy_linear_data(a, b, sigma);
//
//  auto targets = dataset.targets.mean;
//  auto variance = Eigen::VectorXd(targets.size());
//
//  for (int i = 0; i < targets.size(); i++) {
//    double std = 0.1 * fabs(dataset.features[i]);
//    targets[i] += std * d(gen);
//    variance[i] = sigma * sigma + std * std;
//  }
//
//  auto diag_matrix = variance.asDiagonal();
//
//  MarginalDistribution target_dist(targets, diag_matrix);
//
//  return RegressionDataset<double>(dataset.features, target_dist);
//}
//

//inline auto toy_covariance_function() {
//  using Noise = IndependentNoise<double>;
//  SquaredExponential<EuclideanDistance> squared_exponential(100., 100.);
//  IndependentNoise<double> noise = Noise(0.1);
//  auto covariance = squared_exponential + noise;
//  return covariance;
//}

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

// class LinearRegressionTest : public ::testing::Test {
// public:
//  LinearRegressionTest() : model_ptr_(), dataset_() {
//    model_ptr_ = std::make_unique<LinearRegression>();
//    dataset_ = make_toy_linear_data();
//  };
//
//  std::unique_ptr<LinearRegression> model_ptr_;
//  RegressionDataset<double> dataset_;
//};
//
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
//
//// Group values by interval, but return keys that once sorted won't be
//// in order
// inline std::string group_by_interval(const double &x) {
//  if (x <= 3) {
//    return "2";
//  } else if (x <= 6) {
//    return "3";
//  } else {
//    return "1";
//  }
//}
//
// inline bool is_monotonic_increasing(const Eigen::VectorXd &x) {
//  for (Eigen::Index i = 0; i < x.size() - 1; i++) {
//    if (x[i + 1] - x[i] <= 0.) {
//      return false;
//    }
//  }
//  return true;
//}

} // namespace albatross

#endif
