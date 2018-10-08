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

#include "core/functional_model.h"
#include "test_utils.h"
#include <gtest/gtest.h>

namespace albatross {

/*
 * Simply makes sure that a BaseModel that should be able to
 * make perfect predictions compiles and runs as expected.
 */
TEST(test_functional_model, test_fit_predict) {
  auto dataset = mock_training_data();

  GenericModelFunctions<MockFeature, MockModel> funcs;

  funcs.fitter = [](const std::vector<MockFeature> &features,
                    const MarginalDistribution &targets) {
    auto model = MockModel();
    model.fit(features, targets);
    return model;
  };

  funcs.predictor = [](const std::vector<MockFeature> &features,
                       const MockModel &model) {
    return model.predict(features);
  };

  FunctionalRegressionModel<MockFeature, MockModel> functional_model(
      funcs.fitter, funcs.predictor);

  functional_model.fit(dataset);
  // We should be able to perfectly predict in this case.
  JointDistribution predictions = functional_model.predict(dataset.features);
  EXPECT_LT((predictions.mean - dataset.targets.mean).norm(), 1e-10);
}

/*
 * Simply makes sure that a BaseModel that should be able to
 * make perfect predictions compiles and runs as expected.
 */
TEST(test_functional_model, test_get_generic_functions) {
  auto dataset = mock_training_data();

  MockModel m;

  GenericModelFunctions<MockFeature, MockModel> funcs =
      get_generic_functions(m);

  FunctionalRegressionModel<MockFeature, MockModel> functional_model(
      funcs.fitter, funcs.predictor);

  functional_model.fit(dataset);
  // We should be able to perfectly predict in this case.
  JointDistribution predictions = functional_model.predict(dataset.features);
  EXPECT_LT((predictions.mean - dataset.targets.mean).norm(), 1e-10);
}

} // namespace albatross
