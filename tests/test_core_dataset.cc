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

#include <albatross/Dataset>
#include <gtest/gtest.h>

namespace albatross {

TEST(test_dataset, test_construct_and_subset) {
  std::vector<int> features = {3, 7, 1};
  Eigen::VectorXd targets = Eigen::VectorXd::Random(3);

  RegressionDataset<int> dataset(features, targets);

  EXPECT_EQ(dataset.size(), features.size());
  EXPECT_EQ(dataset.size(), static_cast<std::size_t>(targets.size()));

  std::vector<std::size_t> indices = {0, 2};
  const auto subset_dataset = subset(dataset, indices);
  EXPECT_EQ(subset_dataset.size(), indices.size());
}

void expect_split_recombine(const RegressionDataset<int> &dataset) {
  std::vector<std::size_t> first_indices = {0, 1};
  const auto first = subset(dataset, first_indices);
  EXPECT_EQ(first.size(), first_indices.size());

  std::vector<std::size_t> second_indices = {2};
  const auto second = subset(dataset, second_indices);
  EXPECT_EQ(second.size(), second_indices.size());

  const auto reconstructed = concatenate_datasets(first, second);

  EXPECT_EQ(dataset, reconstructed);
}

TEST(test_dataset, test_concatenate_same_type) {
  std::vector<int> features = {3, 7, 1};
  Eigen::VectorXd mean_only_targets = Eigen::VectorXd::Random(3);

  RegressionDataset<int> mean_only_dataset(features, mean_only_targets);

  expect_split_recombine(mean_only_dataset);

  Eigen::VectorXd variance = Eigen::VectorXd::Ones(mean_only_targets.size());
  MarginalDistribution targets(mean_only_targets, variance.asDiagonal());
  RegressionDataset<int> dataset(features, targets);

  expect_split_recombine(dataset);
}

TEST(test_dataset, test_concatenate_different_type) {
  std::vector<int> int_features = {3, 7, 1};
  Eigen::VectorXd int_targets = Eigen::VectorXd::Random(3);
  RegressionDataset<int> int_dataset(int_features, int_targets);

  std::vector<double> double_features = {3., 7., 1.};
  Eigen::VectorXd double_targets = Eigen::VectorXd::Random(3);
  RegressionDataset<double> double_dataset(double_features, double_targets);

  const auto reconstructed = concatenate_datasets(int_dataset, double_dataset);

  EXPECT_TRUE(bool(std::is_same<typename decltype(reconstructed.features)::value_type,
                                variant<int, double>>::value));

  for (std::size_t i = 0; i < reconstructed.features.size(); ++i) {
    if (i < int_features.size()) {
      EXPECT_TRUE(reconstructed.features[i].is<int>());
      EXPECT_FALSE(reconstructed.features[i].is<double>());
      int actual = reconstructed.features[i].get<int>();
      EXPECT_EQ(actual, int_features[i]);
    } else {
      EXPECT_TRUE(reconstructed.features[i].is<double>());
      EXPECT_FALSE(reconstructed.features[i].is<int>());
      double actual = reconstructed.features[i].get<double>();
      EXPECT_EQ(actual, double_features[i - int_features.size()]);
    }
  }
}

} // namespace albatross
