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
#include "test_utils.h"

namespace albatross {
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
  RegressionDataset<MockPredictor> dataset(features, targets);

  auto model_fit_direct = m.fit(features, targets);
  auto model_fit_dataset = m.fit(dataset);

  // It shouldn't matter how we call fit.
  EXPECT_EQ(model_fit_direct, model_fit_direct);

  // We shoudl be able to perfectly predict in this case.
  PredictionDistribution predictions = m.predict(features);
  EXPECT_LT((predictions.mean - targets).norm(), 1e-10);
}



}
