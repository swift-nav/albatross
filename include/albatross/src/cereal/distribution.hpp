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

#ifndef ALBATROSS_CEREAL_DISTRIBUTION_HPP_
#define ALBATROSS_CEREAL_DISTRIBUTION_HPP_

namespace cereal {

template <class Archive>
inline void serialize(Archive &archive, albatross::MarginalDistribution &dist,
                      const std::uint32_t) {
  archive(cereal::make_nvp("mean", dist.mean));
  archive(cereal::make_nvp("covariance", dist.covariance));
  archive(cereal::make_nvp("metadata", dist.metadata));
}

template <class Archive>
inline void serialize(Archive &archive, albatross::JointDistribution &dist,
                      const std::uint32_t) {
  archive(cereal::make_nvp("mean", dist.mean));
  archive(cereal::make_nvp("covariance", dist.covariance));
  archive(cereal::make_nvp("metadata", dist.metadata));
}

} // namespace cereal

#endif /* ALBATROSS_CEREAL_DISTRIBUTION_HPP_ */
