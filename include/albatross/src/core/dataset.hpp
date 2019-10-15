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

#ifndef ALBATROSS_CORE_DATASET_H
#define ALBATROSS_CORE_DATASET_H

namespace albatross {

/*
 * A RegressionDataset holds two vectors of data, the features
 * where a single feature can be any class that contains the information used
 * to make predictions of the target.  This is called a RegressionDataset since
 * it is assumed that each feature is regressed to a single double typed
 * target.
 */
template <typename FeatureType> struct RegressionDataset {
  std::vector<FeatureType> features;
  MarginalDistribution targets;
  std::map<std::string, std::string> metadata;

  using Feature = FeatureType;

  RegressionDataset(){};

  RegressionDataset(const std::vector<FeatureType> &features_,
                    const MarginalDistribution &targets_)
      : features(features_), targets(targets_) {
    // If the two inputs aren't the same size they clearly aren't
    // consistent.
    assert(static_cast<int>(features.size()) ==
           static_cast<int>(targets.size()));
  }

  RegressionDataset(const std::vector<FeatureType> &features_,
                    const Eigen::VectorXd &targets_)
      : RegressionDataset(features_, MarginalDistribution(targets_)) {}

  bool operator==(const RegressionDataset &other) const {
    return (features == other.features && targets == other.targets &&
            metadata == other.metadata);
  }

  std::size_t size() const { return features.size(); }

  template <typename GrouperFunc>
  GroupBy<RegressionDataset<FeatureType>, GrouperFunc>
  group_by(GrouperFunc grouper) const;
};

/*
 * Convenience method which subsets the features and targets of a dataset.
 */
template <typename SizeType, typename FeatureType>
inline RegressionDataset<FeatureType>
subset(const RegressionDataset<FeatureType> &dataset,
       const std::vector<SizeType> &indices) {
  return RegressionDataset<FeatureType>(subset(dataset.features, indices),
                                        subset(dataset.targets, indices));
}

template <typename X>
inline auto concatenate_datasets(const RegressionDataset<X> &x,
                                 const RegressionDataset<X> &y) {
  const auto targets = concatenate_marginals(x.targets, y.targets);
  std::vector<X> features = concatenate(x.features, y.features);
  return RegressionDataset<X>(features, targets);
}

template <typename X>
inline auto
concatenate_datasets(const std::vector<RegressionDataset<X>> &datasets) {
  std::vector<std::vector<X>> features;
  std::vector<MarginalDistribution> targets;

  for (const auto &dataset : datasets) {
    features.emplace_back(dataset.features);
    targets.emplace_back(dataset.targets);
  }

  return RegressionDataset<X>(concatenate(features),
                              concatenate_marginals(targets));
}

template <typename X, typename Y>
inline auto concatenate_datasets(const RegressionDataset<X> &x,
                                 const RegressionDataset<Y> &y) {
  const auto features = concatenate(x.features, y.features);
  const auto targets = concatenate_marginals(x.targets, y.targets);
  return RegressionDataset<typename decltype(features)::value_type>(features,
                                                                    targets);
}

} // namespace albatross

#endif
