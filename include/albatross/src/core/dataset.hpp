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

  template <class Archive>
  typename std::enable_if<valid_in_out_serializer<FeatureType, Archive>::value,
                          void>::type
  serialize(Archive &archive, const std::uint32_t) {
    archive(cereal::make_nvp("features", features));
    archive(cereal::make_nvp("targets", targets));
    archive(cereal::make_nvp("metadata", metadata));
  }

  template <class Archive>
  typename std::enable_if<!valid_in_out_serializer<FeatureType, Archive>::value,
                          void>::type
  serialize(Archive &, const std::uint32_t) {
    static_assert(delay_static_assert<Archive>::value,
                  "In order to serialize a RegressionDataset the corresponding "
                  "FeatureType must be serializable.");
  }
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
  std::vector<X> features(x.features);
  features.insert(features.end(), y.features.begin(), y.features.end());
  return RegressionDataset<X>(features, targets);
}

template <typename X, typename Y>
inline std::vector<variant<X, Y>> concatenate(const std::vector<X> &xs,
                                              const std::vector<Y> &ys) {
  std::vector<variant<X, Y>> features(xs.begin(), xs.end());
  for (const auto &y : ys) {
    features.emplace_back(y);
  }
  return features;
}

template <typename X>
inline std::vector<X> concatenate(const std::vector<X> &xs,
                                  const std::vector<X> &ys) {
  std::vector<X> features(xs.begin(), xs.end());
  for (const auto &y : ys) {
    features.emplace_back(y);
  }
  return features;
}

template <typename X, typename Y>
inline auto concatenate_datasets(const RegressionDataset<X> &x,
                                 const RegressionDataset<Y> &y) {
  const auto features = concatenate(x.features, y.features);
  const auto targets = concatenate_marginals(x.targets, y.targets);
  return RegressionDataset<variant<X, Y>>(features, targets);
}

} // namespace albatross

#endif
