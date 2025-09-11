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

#include "test_utils.h"

namespace albatross {

TEST(test_core_model, test_get_name) {
  auto dataset = mock_training_data();

  MockModel m;
  EXPECT_EQ(m.get_name(), "mock_model");
}

/*
 * Simply makes sure that a BaseModel that should be able to
 * make perfect predictions compiles and runs as expected.
 */
TEST(test_core_model, test_fit_predict) {
  auto dataset = mock_training_data();

  MockModel m;
  const auto fit_model = m.fit(dataset.features, dataset.targets);
  Eigen::VectorXd predictions = fit_model.predict(dataset.features).mean();

  EXPECT_LT((predictions - dataset.targets.mean).norm(), 1e-10);
}

TEST(test_core_model, test_fit_predict_different_types) {
  auto dataset = mock_training_data();
  MockModel m;

  const auto fit_model = m.fit(dataset.features, dataset.targets);

  std::vector<ContainsMockFeature> derived_features;
  for (const auto &f : dataset.features) {
    derived_features.push_back({f});
  }

  Eigen::VectorXd predictions = fit_model.predict(derived_features).mean();

  EXPECT_LT((predictions - dataset.targets.mean).norm(), 1e-10);
}

template <typename ModelType>
void test_get_set(ModelBase<ModelType> &model, const std::string &key) {
  // Make sure a key exists, then modify it and make sure it
  // takes on the new value.
  const auto orig = model.get_param_value(key);
  model.set_param(key, orig + 1.);
  EXPECT_EQ(model.get_params().at(key), orig + 1.);
}

TEST(test_core_model, test_get_set_params) {
  auto model = MockModel();
  auto params = model.get_params();
  std::size_t count = 0;
  for (const auto &pair : params) {
    test_get_set(model, pair.first);
    ++count;
  }
  EXPECT_GT(count, 0);
};

} // namespace albatross
