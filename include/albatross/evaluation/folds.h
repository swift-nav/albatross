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

#ifndef ALBATROSS_EVALUATION_FOLDS_H
#define ALBATROSS_EVALUATION_FOLDS_H

namespace albatross {
/*
 * A combination of training and testing datasets, typically used in cross
 * validation.
 */
template <typename FeatureType> struct RegressionFold {
  RegressionDataset<FeatureType> train_dataset;
  RegressionDataset<FeatureType> test_dataset;
  FoldName name;
  FoldIndices test_indices;

  RegressionFold(const RegressionDataset<FeatureType> &train_dataset_,
                 const RegressionDataset<FeatureType> &test_dataset_,
                 const FoldName &name_, const FoldIndices &test_indices_)
      : train_dataset(train_dataset_), test_dataset(test_dataset_), name(name_),
        test_indices(test_indices_){};
};

/*
 * Leave One Out
 */

template <typename FeatureType>
static inline FoldIndexer
leave_one_out_indexer(const std::vector<FeatureType> &features) {
  FoldIndexer groups;
  for (std::size_t i = 0; i < features.size(); i++) {
    FoldName group_name = std::to_string(i);
    groups[group_name] = {i};
  }
  return groups;
}

template <typename FeatureType>
static inline FoldIndexer
leave_one_out_indexer(const RegressionDataset<FeatureType> &dataset) {
  return leave_one_out_indexer(dataset.features);
}

struct LeaveOneOut {

  template <typename FeatureType>
  FoldIndexer operator()(const RegressionDataset<FeatureType> &dataset) const {
    return leave_one_out_indexer(dataset);
  }
};

/*
 * Leave One Group Out
 */

template <typename FeatureType>
static inline FoldIndexer
leave_one_group_out_indexer(const std::vector<FeatureType> &features,
                            GroupFunction<FeatureType> get_group_name) {
  FoldIndexer groups;
  for (std::size_t i = 0; i < features.size(); i++) {
    const std::string k = get_group_name(features[i]);
    // Get the existing indices if we've already encountered this group_name
    // otherwise initialize a new one.
    FoldIndices indices;
    if (groups.find(k) == groups.end()) {
      indices = FoldIndices();
    } else {
      indices = groups[k];
    }
    // Add the current index.
    indices.push_back(i);
    groups[k] = indices;
  }
  return groups;
}

template <typename FeatureType> struct LeaveOneGroupOut {

  LeaveOneGroupOut(GroupFunction<FeatureType> grouper_) : grouper(grouper_){};

  FoldIndexer operator()(const RegressionDataset<FeatureType> &dataset) const {
    return leave_one_group_out_indexer(dataset.features, grouper);
  }

  GroupFunction<FeatureType> grouper;
};

/*
 * Each flavor of cross validation can be described by a set of
 * FoldIndices, which store which indices should be used for the
 * test cases.  This function takes a map from FoldName to
 * FoldIndices and a dataset and creates the resulting folds.
 */
template <typename FeatureType>
static inline std::vector<RegressionFold<FeatureType>>
folds_from_fold_indexer(const RegressionDataset<FeatureType> &dataset,
                        const FoldIndexer &groups) {
  const std::size_t n = dataset.features.size();
  std::vector<RegressionFold<FeatureType>> folds;
  // For each fold, partition into train and test sets.
  for (const auto &pair : groups) {
    // These get exposed inside the returned RegressionFold and because
    // we'd like to prevent modification of the output from this function
    // from changing the input FoldIndexer we perform a copy here.
    const FoldName group_name(pair.first);
    const FoldIndices test_indices(pair.second);
    const auto train_indices = indices_complement(test_indices, n);

    std::vector<FeatureType> train_features =
        subset(dataset.features, train_indices);
    MarginalDistribution train_targets = subset(dataset.targets, train_indices);

    std::vector<FeatureType> test_features =
        subset(dataset.features, test_indices);
    MarginalDistribution test_targets = subset(dataset.targets, test_indices);

    assert(train_features.size() == train_targets.size());
    assert(test_features.size() == test_targets.size());
    assert(test_targets.size() + train_targets.size() == n);

    const RegressionDataset<FeatureType> train_split(train_features,
                                                     train_targets);
    const RegressionDataset<FeatureType> test_split(test_features,
                                                    test_targets);
    folds.push_back(RegressionFold<FeatureType>(train_split, test_split,
                                                group_name, test_indices));
  }
  return folds;
}

/*
 * Inspects a bunch of folds and creates a set of all the indicies
 * in an original dataset that comprise the test_datasets and the folds.
 */
template <typename FeatureType>
inline std::set<std::size_t>
unique_test_indices(const std::vector<RegressionFold<FeatureType>> &folds) {
  std::set<std::size_t> indices;
  for (const auto &fold : folds) {
    indices.insert(fold.test_indices.begin(), fold.test_indices.end());
  }
  return indices;
}

template <typename FeatureType>
inline std::size_t
dataset_size_from_folds(const std::vector<RegressionFold<FeatureType>> &folds) {
  const auto unique_indices = unique_test_indices(folds);

  // Make sure there were no duplicate test indices.
  std::size_t count = 0;
  for (const auto &fold : folds) {
    count += fold.test_indices.size();
  };
  assert(count == unique_indices.size());

  // Make sure the minimum was zero
  std::size_t zero =
      *std::min_element(unique_indices.begin(), unique_indices.end());
  if (zero != 0) {
    assert(false);
  }

  // And the maximum agrees with the size;
  std::size_t n =
      *std::max_element(unique_indices.begin(), unique_indices.end());
  assert(unique_indices.size() == n + 1);

  return n + 1;
}
} // namespace albatross

#endif /* ALBATROSS_EVALUATION_FOLDS_H */
