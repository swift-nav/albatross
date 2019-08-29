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

#ifndef ALBATROSS_SRC_CEREAL_GP_HPP_
#define ALBATROSS_SRC_CEREAL_GP_HPP_

using albatross::Fit;
using albatross::GPFit;

namespace cereal {

template <typename Archive, typename CovarianceRepresentation,
          typename FeatureType>
inline void serialize(Archive &archive,
                      Fit<GPFit<CovarianceRepresentation, FeatureType>> &fit,
                      const std::uint32_t) {
  archive(cereal::make_nvp("information", fit.information));
  archive(cereal::make_nvp("train_ldlt", fit.train_covariance));
  archive(cereal::make_nvp("train_features", fit.train_features));
}

} // namespace cereal

#endif /* ALBATROSS_SRC_CEREAL_GP_HPP_ */
