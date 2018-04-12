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

#include <gtest/gtest.h>
#include <cmath>
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include "evaluate.h"
#include "core/serialize.h"
#include "models/gp.h"
#include "covariance_functions/covariance_functions.h"
#include "models/linear_regression.h"
#include <cereal/types/map.hpp>


namespace albatross {

// A simple predictor which is effectively just an integer.
struct MockPredictor {
  int value;
  MockPredictor(int v) : value(v){};
};

struct MockFit {
  std::map<int, double> train_data;

 public:
  template <class Archive>
  void serialize(Archive & ar) {
    ar(cereal::make_nvp("train_data", train_data));
  };

  bool operator == (const MockFit &other) const {
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

  std::string get_name() const override{ return "mock_model"; };

  template <typename Archive>
  void save(Archive & archive) const {
    archive(cereal::base_class<SerializableRegressionModel<MockPredictor, MockFit>>(this));
  }

  template <typename Archive>
  void load(Archive & archive) {
    archive(cereal::base_class<SerializableRegressionModel<MockPredictor, MockFit>>(this));
  }

 protected:

  // builds the map from int to value
  MockFit serializable_fit_(const std::vector<MockPredictor> &features,
            const Eigen::VectorXd &targets) const override {
    int n = static_cast<int>(features.size());
    Eigen::VectorXd predictions(n);

    MockFit model_fit;
    for (int i = 0; i < n; i++) {
      model_fit.train_data[features[static_cast<std::size_t>(i)].value] = targets[i];
    }
    return model_fit;
  }

  // looks up the prediction in the map
  PredictionDistribution predict_(
      const std::vector<MockPredictor> &features) const {
    int n = static_cast<int>(features.size());
    Eigen::VectorXd predictions(n);

    for (int i = 0; i < n; i++) {
      int index = features[static_cast<std::size_t>(i)].value;
      predictions[i] = this->model_fit_.train_data.find(index)->second;
    }

    return PredictionDistribution(predictions);
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

static inline RegressionDataset<MockPredictor> mock_training_data(const int n = 10) {
  std::vector<MockPredictor> features;
  Eigen::VectorXd targets(n);
  for (int i = 0; i < n; i++) {
    features.push_back(MockPredictor(i));
    targets[i] = static_cast<double>(i + n);
  }
  return RegressionDataset<MockPredictor>(features, targets);
}

static inline void expect_params_equal(const ParameterStore &x, const ParameterStore &y) {
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

static inline void expect_parameter_vector_equal(const std::vector<ParameterValue> &x,
                                                 const std::vector<ParameterValue> &y) {
  for (std::size_t i = 0; i < x.size(); i++) {
    EXPECT_DOUBLE_EQ(x[i], y[i]);
  }
  EXPECT_EQ(x.size(), y.size());
}


static inline RegressionDataset<Eigen::VectorXd> make_toy_linear_regression_data(const double a = 5.,
                                                                                 const double b = 1.,
                                                                                 const double sigma = 0.1) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  gen.seed(3);
  std::normal_distribution<> d{0., sigma};

  s32 n = 10;
  std::vector<Eigen::VectorXd> features;
  Eigen::VectorXd targets(n);
  Eigen::VectorXd coefs(2);
  coefs << a, b;

  for (s32 i = 0; i < n; i++) {
    double x = static_cast<double>(i);
    auto feature = Eigen::VectorXd(2);
    feature << 1., x;
    features.push_back(feature);
    targets[i] = feature.dot(coefs) + d(gen);
  }

  return RegressionDataset<Eigen::VectorXd>(features, targets);
}

class LinearRegressionTest : public ::testing::Test {
 public:
  LinearRegressionTest() : model_ptr_(), dataset_({}, {}) {

    model_ptr_ = std::make_unique<LinearRegression>();
    dataset_ = make_toy_linear_regression_data();
  };

  std::unique_ptr<LinearRegression> model_ptr_;
  RegressionDataset<Eigen::VectorXd> dataset_;
};

using SqrExp = SquaredExponential<ScalarDistance>;
class SquaredExpoentialGaussianProcess : public GaussianProcessRegression<double, CovarianceFunction<SqrExp>> {
};

}

#endif
