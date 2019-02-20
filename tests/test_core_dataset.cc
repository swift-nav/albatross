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

#include "Dataset"
#include <gtest/gtest.h>

namespace albatross {

TEST(test_dataset, test_construct) {

  std::vector<int> features = {3, 7, 1};
  Eigen::VectorXd targets = Eigen::VectorXd::Random(3);

  RegressionDataset<int> dataset(features, targets);

  EXPECT_EQ(dataset.size(), features.size());
  EXPECT_EQ(dataset.size(), static_cast<std::size_t>(targets.size()));

  std::vector<std::size_t> indices = {0, 2};
  const auto subset_dataset = subset(indices, dataset);
  EXPECT_EQ(subset_dataset.size(), indices.size());
}

} // namespace albatross
