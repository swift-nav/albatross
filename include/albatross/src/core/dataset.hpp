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
template <typename FeatureType>
struct RegressionDataset {
  std::vector<FeatureType> features;
  MarginalDistribution targets;
  std::map<std::string, std::string> metadata;

  using Feature = FeatureType;

  RegressionDataset() {}

  RegressionDataset(const std::vector<FeatureType> &features_,
                    const MarginalDistribution &targets_)
      : features(features_), targets(targets_) {
    // If the two inputs aren't the same size they clearly aren't
    // consistent.
    ALBATROSS_ASSERT(static_cast<int>(features.size()) ==
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

  template <typename SizeType>
  RegressionDataset subset(const std::vector<SizeType> &indices) const;

  template <typename GrouperFunc>
  GroupBy<RegressionDataset<FeatureType>, GrouperFunc> group_by(
      GrouperFunc grouper) const;
};

template <typename FeatureType>
inline auto create_dataset(const std::vector<FeatureType> &features,
                           const MarginalDistribution &targets) {
  return RegressionDataset<FeatureType>(features, targets);
}

/*
 * Convenience method which subsets the features and targets of a dataset.
 */
template <typename SizeType, typename FeatureType>
inline RegressionDataset<FeatureType> subset(
    const RegressionDataset<FeatureType> &dataset,
    const std::vector<SizeType> &indices) {
  return RegressionDataset<FeatureType>(subset(dataset.features, indices),
                                        subset(dataset.targets, indices));
}

template <typename FeatureType>
template <typename SizeType>
RegressionDataset<FeatureType> RegressionDataset<FeatureType>::subset(
    const std::vector<SizeType> &indices) const {
  return albatross::subset(*this, indices);
}

template <typename FeatureType>
RegressionDataset<FeatureType> deduplicate(
    const RegressionDataset<FeatureType> &dataset) {
  auto appears_later = [&](std::size_t index) -> bool {
    for (std::size_t j = index + 1; j < dataset.features.size(); ++j) {
      if (dataset.features[index] == dataset.features[j]) {
        return true;
      }
    }
    return false;
  };

  std::vector<std::size_t> unique_inds;
  for (std::size_t i = 0; i < dataset.size(); ++i) {
    if (!appears_later(i)) {
      unique_inds.push_back(i);
    }
  }

  return albatross::subset(dataset, unique_inds);
}

template <typename X, typename EqualTo>
inline auto align_datasets(RegressionDataset<X> *x, RegressionDataset<X> *y,
                           EqualTo equal_to) {
  std::vector<std::size_t> x_inds;
  std::vector<std::size_t> y_inds;
  for (std::size_t i = 0; i < x->size(); ++i) {
    for (std::size_t j = 0; j < y->size(); ++j) {
      if (equal_to(x->features[i], y->features[j])) {
        x_inds.push_back(i);
        y_inds.push_back(j);
        continue;
      }
    }
  }

  ALBATROSS_ASSERT(x_inds.size() == y_inds.size());

  if (x_inds.size() == 0) {
    *x = RegressionDataset<X>();
    *y = RegressionDataset<X>();
  } else {
    *x = subset(*x, x_inds);
    *y = subset(*y, y_inds);
  }
}

template <typename X>
inline auto align_datasets(RegressionDataset<X> *x, RegressionDataset<X> *y) {
  return align_datasets(x, y, std::equal_to<X>());
}

template <typename X>
inline auto concatenate_datasets(const RegressionDataset<X> &x,
                                 const RegressionDataset<X> &y) {
  const auto targets = concatenate_marginals(x.targets, y.targets);
  std::vector<X> features = concatenate(x.features, y.features);
  return RegressionDataset<X>(features, targets);
}

template <typename X>
inline auto concatenate_datasets(
    const std::vector<RegressionDataset<X>> &datasets) {
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

template <typename Derived, typename X>
inline auto operator*(const Eigen::SparseMatrixBase<Derived> &matrix,
                      const albatross::RegressionDataset<X> &dataset) {
  const auto transformed_features = matrix.derived() * dataset.features;
  using TransformedType = typename decltype(transformed_features)::value_type;

  return albatross::RegressionDataset<TransformedType>(
      transformed_features, matrix.derived() * dataset.targets);
}

template <typename Derived, typename X>
inline auto operator*(const Eigen::MatrixBase<Derived> &matrix,
                      const albatross::RegressionDataset<X> &dataset) {
  const auto transformed_features = matrix.derived() * dataset.features;
  using TransformedType = typename decltype(transformed_features)::value_type;

  return albatross::RegressionDataset<TransformedType>(
      transformed_features, matrix.derived() * dataset.targets);
}

template <typename X,
          typename std::enable_if_t<is_streamable<X>::value, int> = 0>
inline std::ostream &operator<<(std::ostream &os,
                                const RegressionDataset<X> &dataset) {
  for (std::size_t i = 0; i < dataset.size(); ++i) {
    os << dataset.features[i] << "    "
       << dataset.targets.mean[cast::to_index(i)] << "   +/- "
       << std::sqrt(dataset.targets.get_diagonal(cast::to_index(i)))
       << std::endl;
  }
  return os;
}

template <typename X,
          typename std::enable_if_t<!is_streamable<X>::value, int> = 0>
inline std::ostream &operator<<(std::ostream &os,
                                const RegressionDataset<X> &dataset) {
  for (std::size_t i = 0; i < dataset.size(); ++i) {
    os << i << "    " << dataset.targets.mean[cast::to_index(i)] << "   +/- "
       << std::sqrt(dataset.targets.get_diagonal(cast::to_index(i)))
       << std::endl;
  }
  return os;
}
}  // namespace albatross

#endif
