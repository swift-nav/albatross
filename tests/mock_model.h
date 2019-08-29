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

#ifndef ALBATROSS_TESTS_MOCK_MODEL_H
#define ALBATROSS_TESTS_MOCK_MODEL_H

namespace albatross {

class MockModel;

// A simple predictor which is effectively just an integer.
struct MockFeature {
  int value;

  MockFeature() : value(){};
  MockFeature(int v) : value(v){};

  bool operator==(const MockFeature &other) const {
    return value == other.value;
  };

  template <class Archive> void serialize(Archive &archive) {
    archive(cereal::make_nvp("value", value));
  }
};

struct ContainsMockFeature {
  MockFeature mock;
};

template <> struct Fit<MockModel> {
  std::map<int, double> train_data;

  template <class Archive> void serialize(Archive &ar) {
    ar(cereal::make_nvp("train_data", train_data));
  };

  bool operator==(const Fit &other) const {
    return train_data == other.train_data;
  };
};

/*
 * A simple model which builds a map from MockPredict (aka, int)
 * to a double value.
 */
class MockModel : public ModelBase<MockModel> {
public:
  ALBATROSS_DECLARE_PARAMS(foo, bar);

  MockModel(double foo_ = 3.14159, double bar_ = sqrt(2.)) {
    this->foo = {foo_, GaussianPrior(3., 2.)};
    this->bar = {bar_, PositivePrior()};
  };

  std::string get_name() const { return "mock_model"; }

  bool operator==(const MockModel &other) const {
    return other.get_params() == this->get_params();
  }

  Fit<MockModel> _fit_impl(const std::vector<MockFeature> &features,
                           const MarginalDistribution &targets) const {
    int n = static_cast<int>(features.size());
    Eigen::VectorXd predictions(n);
    Fit<MockModel> model_fit;
    for (int i = 0; i < n; i++) {
      model_fit.train_data[features[static_cast<std::size_t>(i)].value] =
          targets.mean[i];
    }
    return model_fit;
  }

  // looks up the prediction in the map
  Eigen::VectorXd _predict_impl(const std::vector<MockFeature> &features,
                                const Fit<MockModel> &fit_,
                                PredictTypeIdentity<Eigen::VectorXd> &&) const {
    int n = static_cast<int>(features.size());
    Eigen::VectorXd predictions(n);

    for (int i = 0; i < n; i++) {
      int index = features[static_cast<std::size_t>(i)].value;
      predictions[i] = fit_.train_data.find(index)->second;
    }

    return predictions;
  }

  // convert before predicting
  Eigen::VectorXd
  _predict_impl(const std::vector<ContainsMockFeature> &features,
                const Fit<MockModel> &fit_,
                PredictTypeIdentity<Eigen::VectorXd> &&) const {
    std::vector<MockFeature> mock_features;
    for (const auto &f : features) {
      mock_features.push_back(f.mock);
    }
    return _predict_impl(mock_features, fit_,
                         PredictTypeIdentity<Eigen::VectorXd>());
  }
};

static inline RegressionDataset<MockFeature>
mock_training_data(const int n = 10) {
  std::vector<MockFeature> features;
  Eigen::VectorXd targets(n);
  for (int i = 0; i < n; i++) {
    features.push_back(MockFeature(i));
    targets[i] = static_cast<double>(i + n);
  }
  return RegressionDataset<MockFeature>(features, targets);
}
} // namespace albatross

#endif
