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

#include "Dataset"

#include "test_utils.h"

namespace albatross {

TEST(test_concatenate, test_concatenate_datasets) {

  auto dataset = make_toy_linear_data();
  dataset.metadata["test"] = "metadata";

  std::vector<std::size_t> first_inds;
  for (std::size_t i = 0; i < dataset.features.size() - 3; ++i) {
    first_inds.push_back(i);
  }
  const auto last_inds = indices_complement(first_inds, dataset.features.size());

  const auto first = albatross::subset(first_inds, dataset);
  const auto last = albatross::subset(last_inds, dataset);

  std::vector<decltype(dataset)> splits = {first, last};
  const auto reassembled = concatenate_datasets(splits);

  EXPECT_EQ(reassembled.features, dataset.features);
  EXPECT_EQ(reassembled.targets.mean, dataset.targets.mean);
  EXPECT_EQ(reassembled.targets.covariance.diagonal(),
            dataset.targets.covariance.diagonal());
}

TEST(test_concatenate, test_concatenate_distributions) {

  Eigen::VectorXd a(3);
  a << 1., 2., 3.;
  Eigen::VectorXd b(2);
  b << 4., 5.;

  MarginalDistribution A(a, a.asDiagonal());
  MarginalDistribution B(b, b.asDiagonal());

  std::vector<MarginalDistribution> to_concatenate = {A, B};
  const auto C = concatenate_distributions(to_concatenate);

  Eigen::VectorXd c(5);
  c << 1., 2., 3., 4., 5.;
  EXPECT_EQ(C.mean, c);
  EXPECT_EQ(C.covariance.diagonal(), c);
}

} // namespace albatross
