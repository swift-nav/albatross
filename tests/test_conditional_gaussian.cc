/*
 * Copyright (C) 2021 Swift Navigation Inc.
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

#include "test_models.h"

namespace albatross {

TEST(test_conditional, test_equivalent_to_gp) {
  MakeGaussianProcessWithMean gp_with_mean_case;

  const auto dataset = gp_with_mean_case.get_dataset();
  const auto gp = gp_with_mean_case.get_model();

  const auto prior = gp.prior(dataset.features);

  const ConditionalGaussian model(prior, dataset.targets);

  ASSERT_GT(dataset.size(), 5);

  const std::vector<std::size_t> train_inds = {0, 2, 4};
  const std::vector<std::size_t> test_inds = {1, 3};

  const auto gp_fit = gp.fit(dataset.subset(train_inds));

  const auto meas_features =
      as_measurements(dataset.subset(test_inds).features);
  const auto gp_pred = gp_fit.predict(meas_features).joint();

  const auto conditional_fit = model.fit(train_inds);
  const auto conditional_pred = conditional_fit.predict(test_inds).joint();

  EXPECT_LT((conditional_pred.mean - gp_pred.mean).norm(), 1e-6);
  EXPECT_LT((conditional_pred.covariance - gp_pred.covariance).norm(), 1e-6);
}

}  // namespace albatross
