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

#ifndef ALBATROSS_CEREAL_DATASET_HPP_
#define ALBATROSS_CEREAL_DATASET_HPP_

using albatross::delay_static_assert;
using albatross::RegressionDataset;
using albatross::valid_in_out_serializer;

namespace cereal {

template <class Archive, class FeatureType>
typename std::enable_if<valid_in_out_serializer<FeatureType, Archive>::value,
                        void>::type
serialize(Archive &archive, RegressionDataset<FeatureType> &dataset,
          const std::uint32_t) {
  archive(cereal::make_nvp("features", dataset.features));
  archive(cereal::make_nvp("targets", dataset.targets));
  archive(cereal::make_nvp("metadata", dataset.metadata));
}

template <class Archive, class FeatureType>
typename std::enable_if<!valid_in_out_serializer<FeatureType, Archive>::value,
                        void>::type
serialize(Archive &archive, RegressionDataset<FeatureType> &dataset,
          const std::uint32_t) {
  static_assert(delay_static_assert<Archive>::value,
                "In order to serialize a RegressionDataset the corresponding "
                "FeatureType must be serializable.");
}

} // namespace cereal

#endif /* ALBATROSS_CEREAL_DATASET_HPP_ */
