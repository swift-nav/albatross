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
TEST(test_base_model, test_base_model_fit_predict) {
  auto dataset = mock_training_data();
  MockModel m;
  m.fit(dataset);
  // We should be able to perfectly predict in this case.
  PredictionDistribution predictions = m.predict(dataset.features);
  EXPECT_LT((predictions.mean - dataset.targets).norm(), 1e-10);
}

TEST(test_base_model, test_regression_model_abstraction) {
  // This just tests to make sure that an implementation of a RegressionModel
  // can be passed around as a pointer to the abstract class.
  std::unique_ptr<RegressionModel<MockPredictor>> m_ptr = std::make_unique<MockModel>();
}

}
