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
template <typename FeatureType>
struct RegressionFold {
  RegressionDataset<FeatureType> train_dataset;
  RegressionDataset<FeatureType> test_dataset;
  GroupIndices test_indices;

  RegressionFold(const RegressionDataset<FeatureType> &train_dataset_,
                 const RegressionDataset<FeatureType> &test_dataset_,
                 const GroupIndices &test_indices_)
      : train_dataset(train_dataset_), test_dataset(test_dataset_),
        test_indices(test_indices_){};
};

///*
// * Leave One Out
// */
//template <typename FeatureType>
//inline GroupIndexer<std::size_t>
//leave_one_out_indexer(const std::vector<FeatureType> &features) {
//  GroupIndexer<std::size_t> groups;
//  for (std::size_t i = 0; i < features.size(); i++) {
//    groups[i] = {i};
//  }
//  return groups;
//}
//
//template <typename FeatureType>
//inline GroupIndexer<std::size_t>
//leave_one_out_indexer(const RegressionDataset<FeatureType> &dataset) {
//  return leave_one_out_indexer(dataset.features);
//}
//
//struct LeaveOneOut {
//
//  template <typename FeatureType>
//  GroupIndexer<std::size_t> operator()(const RegressionDataset<FeatureType> &dataset) const {
//    return leave_one_out_indexer(dataset);
//  }
//
//  template <typename FeatureType>
//  GroupIndexer<std::size_t> operator()(const std::vector<FeatureType> &features) const {
//    return leave_one_out_indexer(features);
//  }
//};

template <typename FeatureType>
inline RegressionFold<FeatureType>
create_fold (const GroupIndices &test_indices,
             const RegressionDataset<FeatureType> &dataset) {

  const auto train_indices = indices_complement(test_indices, dataset.size());

  std::vector<FeatureType> train_features =
      subset(dataset.features, train_indices);
  MarginalDistribution train_targets = subset(dataset.targets, train_indices);

  std::vector<FeatureType> test_features =
      subset(dataset.features, test_indices);
  MarginalDistribution test_targets = subset(dataset.targets, test_indices);

  assert(train_features.size() == train_targets.size());
  assert(test_features.size() == test_targets.size());
  assert(test_targets.size() + train_targets.size() == dataset.size());

  const RegressionDataset<FeatureType> train_split(train_features,
                                                   train_targets);
  const RegressionDataset<FeatureType> test_split(test_features,
                                                  test_targets);
  return RegressionFold<FeatureType>(train_split, test_split, test_indices);
};


/*
 * Each flavor of cross validation can be described by a set of
 * GroupIndices, which store which indices should be used for the
 * test cases.  This function takes a map from FoldName to
 * GroupIndices and a dataset and creates the resulting folds.
 */
template <typename FeatureType, typename GroupKey>
inline RegressionFolds<GroupKey, FeatureType>
folds_from_group_indexer(const RegressionDataset<FeatureType> &dataset,
                        const GroupIndexer<GroupKey> &groups) {

  const auto create_one_fold = [&dataset](const GroupKey &, const GroupIndices &test_indices) {
    return create_fold(test_indices, dataset);
  };

  return groups.apply(create_one_fold);
}

/*
 * Extracts the fold indexer that would have created a set of folds
 */
template <typename GroupKey, typename FeatureType>
inline
GroupIndexer<GroupKey>
group_indexer_from_folds(const std::map<GroupKey, FeatureType> &folds) {
  GroupIndexer<GroupKey> output;
  for (const auto &fold : folds) {
    assert(!map_contains(output, fold.first));
    output[fold.first] = fold.second.test_indices;
  }
  return output;
}

template <typename FeatureType, typename GrouperFunction>
inline auto
folds_from_grouper(const RegressionDataset<FeatureType> &dataset,
                   const GrouperFunction &grouper) {

  const auto create_one_fold = [&dataset](const auto &, const GroupIndices &test_indices) {
    return create_fold(test_indices, dataset);
  };

  return dataset.group_by(grouper).index_apply(create_one_fold);
}

//template <typename FeatureType, typename GroupKey>
//inline auto
//folds_from_grouper(const RegressionDataset<FeatureType> &dataset,
//                   GroupKey(*grouper)(FeatureType)) {
//
//  const auto create_one_fold = [&dataset](const auto &, const GroupIndices &test_indices) {
//    create_fold(test_indices, dataset);
//  };
//
//  const auto groupit = [&](const FeatureType &f) {
//    return grouper(f);
//  };
//
//  return dataset.group_by(groupit).index_apply(create_one_fold);
//}

/*
 * Inspects a bunch of folds and creates a set of all the indicies
 * in an original dataset that comprise the test_datasets and the folds.
 */
template <typename GroupKey>
inline std::set<std::size_t> unique_indices(const GroupIndexer<GroupKey> &indexer) {
  std::set<std::size_t> indices;
  for (const auto &pair : indexer) {
    indices.insert(pair.second.begin(), pair.second.end());
  }
  return indices;
}

template <typename GroupKey>
inline std::size_t dataset_size_from_indexer(const GroupIndexer<GroupKey> &indexer) {
  const auto unique_inds = unique_indices(indexer);

  // Make sure there were no duplicate test indices.
  std::size_t count = 0;
  for (const auto &pair : indexer) {
    count += pair.second.size();
  };
  assert(count == unique_inds.size());

  // Make sure the minimum was zero
  std::size_t zero = *std::min_element(unique_inds.begin(), unique_inds.end());
  if (zero != 0) {
    assert(false);
  }

  // And the maximum agrees with the size;
  std::size_t n = *std::max_element(unique_inds.begin(), unique_inds.end());
  assert(unique_inds.size() == n + 1);

  return n + 1;
}

template < typename GroupKey, typename FeatureType>
inline std::size_t
dataset_size_from_folds(const RegressionFolds<GroupKey, FeatureType> &folds) {
  return dataset_size_from_indexer(group_indexer_from_folds(folds));
}

} // namespace albatross

#endif /* ALBATROSS_EVALUATION_FOLDS_H */
