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

#include "core/serialize.h"
#include "covariance_functions/covariance_functions.h"
#include "evaluate.h"
#include "models/gp.h"
#include "models/least_squares.h"
#include <Eigen/Dense>
#include <cereal/types/map.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <random>

namespace albatross {

// A simple predictor which is effectively just an integer.
struct MockPredictor {
  int value;
  MockPredictor(int v) : value(v){};
};

struct MockFit {
  std::map<int, double> train_data;

public:
  template <class Archive> void serialize(Archive &ar) {
    ar(cereal::make_nvp("train_data", train_data));
  };

  bool operator==(const MockFit &other) const {
    return train_data == other.train_data;
  };
};

/*
 * A simple model which builds a map from MockPredict (aka, int)
 * to a double value.
 */
class MockModel : public SerializableRegressionModel<MockPredictor, MockFit> {
public:
  MockModel(double parameter = 3.14159) {
    this->params_["parameter"] = parameter;
  };

  std::string get_name() const override { return "mock_model"; };

  template <typename Archive> void save(Archive &archive) const {
    archive(
        cereal::base_class<SerializableRegressionModel<MockPredictor, MockFit>>(
            this));
  }

  template <typename Archive> void load(Archive &archive) {
    archive(
        cereal::base_class<SerializableRegressionModel<MockPredictor, MockFit>>(
            this));
  }

protected:
  // builds the map from int to value
  MockFit serializable_fit_(const std::vector<MockPredictor> &features,
                            const TargetDistribution &targets) const override {
    int n = static_cast<int>(features.size());
    Eigen::VectorXd predictions(n);

    MockFit model_fit;
    for (int i = 0; i < n; i++) {
      model_fit.train_data[features[static_cast<std::size_t>(i)].value] =
          targets.mean[i];
    }
    return model_fit;
  }

  // looks up the prediction in the map
  PredictDistribution
  predict_(const std::vector<MockPredictor> &features) const override {
    int n = static_cast<int>(features.size());
    Eigen::VectorXd predictions(n);

    for (int i = 0; i < n; i++) {
      int index = features[static_cast<std::size_t>(i)].value;
      predictions[i] = this->model_fit_.train_data.find(index)->second;
    }

    return PredictDistribution(predictions);
  }
};

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

static inline RegressionDataset<MockPredictor>
mock_training_data(const int n = 10) {
  std::vector<MockPredictor> features;
  Eigen::VectorXd targets(n);
  for (int i = 0; i < n; i++) {
    features.push_back(MockPredictor(i));
    targets[i] = static_cast<double>(i + n);
  }
  return RegressionDataset<MockPredictor>(features, targets);
}

static inline void expect_params_equal(const ParameterStore &x,
                                       const ParameterStore &y) {
  // Make sure all pairs in x are in y.
  for (const auto &x_pair : x) {
    const auto y_value = y.at(x_pair.first);
    EXPECT_DOUBLE_EQ(x_pair.second, y_value);
  }
  // And all pairs in y are in x.
  for (const auto &y_pair : y) {
    const auto x_value = x.at(y_pair.first);
    EXPECT_DOUBLE_EQ(y_pair.second, x_value);
  }
}

static inline void
expect_parameter_vector_equal(const std::vector<ParameterValue> &x,
                              const std::vector<ParameterValue> &y) {
  for (std::size_t i = 0; i < x.size(); i++) {
    EXPECT_DOUBLE_EQ(x[i], y[i]);
  }
  EXPECT_EQ(x.size(), y.size());
}

static inline auto make_toy_linear_data(const double a = 5.,
                                        const double b = 1.,
                                        const double sigma = 0.1,
                                        const s32 n = 10) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  gen.seed(3);
  std::normal_distribution<> d{0., sigma};
  std::vector<double> features;
  Eigen::VectorXd targets(n);

  for (s32 i = 0; i < n; i++) {
    double x = static_cast<double>(i);
    features.push_back(x);
    targets[i] = a + x * b + d(gen);
  }

  return RegressionDataset<double>(features, targets);
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

  TargetDistribution target_dist(targets, diag_matrix);

  return RegressionDataset<double>(dataset.features, target_dist);
}

static inline auto toy_covariance_function() {
  using SqrExp = SquaredExponential<EuclideanDistance>;
  using Noise = IndependentNoise<double>;
  CovarianceFunction<SqrExp> squared_exponential = {SqrExp(100., 100.)};
  CovarianceFunction<Noise> noise = {Noise(0.1)};
  auto covariance = squared_exponential + noise;
  return covariance;
}

static inline std::unique_ptr<RegressionModel<double>> toy_gaussian_process() {
  return gp_pointer_from_covariance<double>(toy_covariance_function());
}

/*
 * Here we create data and a model that will make it easier to test
 * that models using the model_adapter.h interface work.
 */
struct AdaptedFeature {
  double value;
};

static inline auto make_adapted_toy_linear_data() {
  const auto dataset = make_toy_linear_data();

  std::vector<AdaptedFeature> adapted_features;
  for (const auto &f : dataset.features) {
    adapted_features.push_back({f});
  }

  RegressionDataset<AdaptedFeature> adapted_dataset(adapted_features,
                                                    dataset.targets);
  return adapted_dataset;
}

template <typename SubModelType>
class AdaptedExample
    : public AdaptedRegressionModel<AdaptedFeature, SubModelType> {
public:
  AdaptedExample(const SubModelType &model)
      : AdaptedRegressionModel<AdaptedFeature, SubModelType>(model){};
  virtual ~AdaptedExample(){};

  virtual const double
  convert_feature(const AdaptedFeature &parent_feature) const {
    return parent_feature.value;
  }
};

static inline std::unique_ptr<RegressionModel<AdaptedFeature>>
adapted_toy_gaussian_process() {
  auto covariance = toy_covariance_function();
  auto gp = gp_from_covariance<double>(covariance);
  return std::make_unique<AdaptedExample<decltype(gp)>>(gp);
}

class LinearRegressionTest : public ::testing::Test {
public:
  LinearRegressionTest() : model_ptr_(), dataset_() {
    model_ptr_ = std::make_unique<LinearRegression>();
    dataset_ = make_toy_linear_data();
  };

  std::unique_ptr<LinearRegression> model_ptr_;
  RegressionDataset<double> dataset_;
};

inline auto random_spherical_points(s32 n = 10, double radius = 1.,
                                    int seed = 5) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  gen.seed(seed);

  std::uniform_real_distribution<double> rand_lon(0., 2 * M_PI);
  std::uniform_real_distribution<double> rand_lat(-M_PI / 2., M_PI / 2.);

  std::vector<Eigen::VectorXd> points;

  for (s32 i = 0; i < n; i++) {
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

} // namespace albatross

#endif
