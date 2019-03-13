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

/*
 * Splits a dataset into cross validation folds where each fold contains all
 but
 * one predictor/target pair.
 */
template <typename FeatureType>
static inline FoldIndexer leave_one_group_out_indexer(
    const std::vector<FeatureType> &features,
    const GrouperFunction<FeatureType> &get_group_name) {
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
  // For a dataset with n features, we'll have n folds.
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
        subset(train_indices, dataset.features);
    MarginalDistribution train_targets = subset(train_indices, dataset.targets);

    std::vector<FeatureType> test_features =
        subset(test_indices, dataset.features);
    MarginalDistribution test_targets = subset(test_indices, dataset.targets);

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
}

#endif /* ALBATROSS_EVALUATION_FOLDS_H */
