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
#include "core/model.h"

namespace albatross {

// A simple predictor which is effectively just an integer.
struct MockPredictor {
  int value;
  MockPredictor(int v) : value(v){};
};

/*
 * A simple model which builds a map from MockPredict (aka, int)
 * to a double value.
 */
class MockModel : public RegressionModel<MockPredictor> {
 public:
  MockModel() : train_data_(){};

  std::string get_name() const { return "mock_model"; };

 private:
  // builds the map from int to value
  void fit_(const std::vector<MockPredictor> &features,
            const Eigen::VectorXd &targets) {
    int n = static_cast<int>(features.size());
    Eigen::VectorXd predictions(n);

    for (int i = 0; i < n; i++) {
      train_data_[features[static_cast<std::size_t>(i)].value] = targets[i];
    }
  }

  // looks up the prediction in the map
  PredictionDistribution predict_(
      const std::vector<MockPredictor> &features) const {
    int n = static_cast<int>(features.size());
    Eigen::VectorXd predictions(n);

    for (int i = 0; i < n; i++) {
      int index = features[static_cast<std::size_t>(i)].value;
      predictions[i] = train_data_.find(index)->second;
    }

    return PredictionDistribution(predictions);
  }

  std::map<int, double> train_data_;
};

/*
 * Simply makes sure that a BaseModel that should be able to
 * make perfect predictions compiles and runs as expected.
 */
TEST(test_base_model, test_base_model) {
  int n = 10;
  std::vector<MockPredictor> features;
  Eigen::VectorXd targets(n);
  for (int i = 0; i < n; i++) {
    features.push_back(MockPredictor(i));
    targets[i] = static_cast<double>(i + n);
  }

  MockModel m;
  m.fit(features, targets);
  PredictionDistribution predictions = m.predict(features);

  EXPECT_LT((predictions.mean - targets).norm(), 1e-10);
}
}
