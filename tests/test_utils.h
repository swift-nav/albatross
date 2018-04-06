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
#include "models/linear_regression.h"

namespace albatross {

// A simple predictor which is effectively just an integer.
struct MockPredictor {
  int value;
  MockPredictor(int v) : value(v){};
};

struct MockFit {
  std::map<int, double> train_data;

  bool operator == (const MockFit &other) const {
    return train_data == other.train_data;
  };
};

/*
 * A simple model which builds a map from MockPredict (aka, int)
 * to a double value.
 */
class MockModel : public RegressionModel<MockPredictor, MockFit> {
 public:
  MockModel() {};

  std::string get_name() const override{ return "mock_model"; };

 private:
  // builds the map from int to value
  MockFit fit_(const std::vector<MockPredictor> &features,
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
      predictions[i] = fit_storage_->train_data.find(index)->second;
    }

    return PredictionDistribution(predictions);
  }
};

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

}

#endif
