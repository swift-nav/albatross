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
#include <albatross/Indexing>
#include <gtest/gtest.h>

namespace albatross {

TEST(test_dataset, test_construct_and_subset) {
  std::vector<int> features = {3, 7, 1};
  const auto targets = Eigen::VectorXd::Random(3);

  RegressionDataset<int> dataset(features, targets);

  EXPECT_EQ(dataset.size(), features.size());
  EXPECT_EQ(dataset.size(), cast::to_size(targets.size()));

  std::vector<std::size_t> indices = {0, 2};
  const auto subset_dataset = dataset.subset(indices);
  EXPECT_EQ(subset_dataset.size(), indices.size());

  auto is_3_or_1 = [](const int &x) { return x == 3 || x == 1; };

  const auto filtered_dataset = filter(dataset, is_3_or_1);
  EXPECT_EQ(subset_dataset, filtered_dataset);
}

template <typename T>
Eigen::VectorXd random_targets_for(const std::vector<T> &features) {
  return Eigen::VectorXd::Random(cast::to_index(features.size()));
}

template <typename T>
RegressionDataset<T> random_dataset_for(const std::vector<T> &features) {
  return {features, random_targets_for(features)};
}

TEST(test_dataset, test_deduplicate) {
  const auto dataset = random_dataset_for(std::vector<int>{0, 1, 1, 2});
  const auto dedupped = deduplicate(dataset);

  const std::vector<std::size_t> expected_inds = {0, 2, 3};

  EXPECT_EQ(dedupped, dataset.subset(expected_inds));
  EXPECT_EQ(dedupped, deduplicate(dedupped));
}

TEST(test_dataset, test_align_datasets_a_in_b) {
  auto dataset_a = random_dataset_for(std::vector<int>{0, 1, 2});
  auto dataset_b = random_dataset_for(std::vector<int>{2, 3, 0, 1});

  EXPECT_NE(dataset_a.features, dataset_b.features);
  align_datasets(&dataset_a, &dataset_b);
  EXPECT_EQ(dataset_a.size(), 3);
  EXPECT_EQ(dataset_a.features, dataset_b.features);
}

TEST(test_dataset, test_align_datasets_a_in_b_custom_compare) {
  auto dataset_a = random_dataset_for(std::vector<int>{0, 1, 2});
  auto dataset_b = random_dataset_for(std::vector<int>{2, 3, 0, 1});

  EXPECT_NE(dataset_a.features, dataset_b.features);

// GCC 6 gets confused by this line, I think because `align_datasets`
// is marked `inline`
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
  auto custom_compare = [](const auto &x, const auto &y) { return x == y; };
#pragma GCC diagnostic pop

  align_datasets(&dataset_a, &dataset_b, custom_compare);
  EXPECT_EQ(dataset_a.size(), 3);
  EXPECT_EQ(dataset_a.features, dataset_b.features);
}

TEST(test_dataset, test_align_datasets_a_in_b_unordered) {
  auto dataset_a = random_dataset_for(std::vector<int>{0, 2, 1});
  auto dataset_b = random_dataset_for(std::vector<int>{2, 3, 0, 1});

  EXPECT_NE(dataset_a.features, dataset_b.features);
  align_datasets(&dataset_a, &dataset_b);
  EXPECT_EQ(dataset_a.size(), 3);
  EXPECT_EQ(dataset_a.features, dataset_b.features);
}

TEST(test_dataset, test_align_datasets_a_not_in_b) {
  auto dataset_a = random_dataset_for(std::vector<int>{0, 1, 2, 3});
  auto dataset_b = random_dataset_for(std::vector<int>{2, 4, 0});

  EXPECT_NE(dataset_a.features, dataset_b.features);
  align_datasets(&dataset_a, &dataset_b);
  EXPECT_EQ(dataset_a.size(), 2);
  EXPECT_EQ(dataset_a.features, dataset_b.features);
}

TEST(test_dataset, test_align_datasets_no_intersect) {
  auto dataset_a = random_dataset_for(std::vector<int>{0, 1, 2});
  auto dataset_b = random_dataset_for(std::vector<int>{3, 4, 5});

  align_datasets(&dataset_a, &dataset_b);
  EXPECT_EQ(dataset_a.size(), 0);
  EXPECT_EQ(dataset_b.size(), 0);
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
  const auto mean_only_targets = random_targets_for(features);
  RegressionDataset<int> mean_only_dataset(features, mean_only_targets);

  expect_split_recombine(mean_only_dataset);

  Eigen::VectorXd variance = Eigen::VectorXd::Ones(mean_only_targets.size());
  MarginalDistribution targets(mean_only_targets, variance.asDiagonal());
  RegressionDataset<int> dataset(features, targets);

  expect_split_recombine(dataset);
}

TEST(test_dataset, test_concatenate_different_type) {
  std::vector<int> int_features = {3, 7, 1};
  RegressionDataset<int> int_dataset(int_features,
                                     random_targets_for(int_features));

  std::vector<double> double_features = {3., 7., 1.};
  RegressionDataset<double> double_dataset(double_features,
                                           random_targets_for(double_features));

  const auto reconstructed = concatenate_datasets(int_dataset, double_dataset);

  EXPECT_TRUE(
      bool(std::is_same<typename decltype(reconstructed.features)::value_type,
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

TEST(test_dataset, test_streamable_features) {
  auto dataset = random_dataset_for(std::vector<int>{3, 7, 1});

  std::ostringstream oss;
  oss << dataset << std::endl;
}

struct NotStreamable {};

TEST(test_dataset, test_not_streamable_features) {
  auto dataset = random_dataset_for(std::vector<NotStreamable>{
      NotStreamable(), NotStreamable(), NotStreamable()});

  std::ostringstream oss;
  oss << dataset << std::endl;
}

} // namespace albatross
