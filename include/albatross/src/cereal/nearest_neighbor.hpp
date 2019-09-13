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

#ifndef ALBATROSS_CEREAL_NEAREST_NEIGHBOR_HPP_
#define ALBATROSS_CEREAL_NEAREST_NEIGHBOR_HPP_

namespace albatross {

template <typename DistanceMetric> class NearestNeighborModel;

template <typename FeatureType> struct NearestNeighborFit;

} // namespace albatross

namespace cereal {

template <typename Archive, typename FeatureType>
inline void
save(Archive &archive,
     const albatross::Fit<albatross::NearestNeighborFit<FeatureType>> &fit,
     const std::uint32_t) {
  archive(cereal::make_nvp("training_features", fit.training_data.features));
  archive(cereal::make_nvp("training_targets", fit.training_data.targets));
}

template <typename Archive, typename FeatureType>
inline void
load(Archive &archive,
     albatross::Fit<albatross::NearestNeighborFit<FeatureType>> &fit,
     const std::uint32_t) {
  std::vector<FeatureType> features;
  archive(cereal::make_nvp("training_features", features));
  albatross::MarginalDistribution targets;
  archive(cereal::make_nvp("training_targets", targets));
  fit.training_data = RegressionDataset<FeatureType>(features, targets);
}

} // namespace cereal

#endif /* ALBATROSS_CEREAL_NEAREST_NEIGHBOR_HPP_ */
