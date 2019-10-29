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

#ifndef THIRD_PARTY_ALBATROSS_INCLUDE_ALBATROSS_SRC_CEREAL_RANSAC_HPP_
#define THIRD_PARTY_ALBATROSS_INCLUDE_ALBATROSS_SRC_CEREAL_RANSAC_HPP_

using albatross::Fit;
using albatross::GaussianProcessRansacStrategy;
using albatross::GenericRansacStrategy;
using albatross::Ransac;
using albatross::RansacFit;
using albatross::RansacOutput;

namespace cereal {

template <typename Archive, typename GroupKey>
inline void serialize(Archive &archive, RansacOutput<GroupKey> &ransac_output,
                      const std::uint32_t) {
  archive(cereal::make_nvp("return_code", ransac_output.return_code));
  archive(cereal::make_nvp("inliers", ransac_output.inliers));
  archive(cereal::make_nvp("outliers", ransac_output.outliers));
  archive(cereal::make_nvp("consensus_metric", ransac_output.consensus_metric));
}

template <typename Archive, typename ModelType, typename StrategyType,
          typename FeatureType, typename GroupKey>
inline void
serialize(Archive &archive,
          Fit<RansacFit<ModelType, StrategyType, FeatureType, GroupKey>> &fit,
          const std::uint32_t) {
  archive(cereal::make_nvp("maybe_empty_fit_model", fit.maybe_empty_fit_model));
  archive(cereal::make_nvp("ransac_output", fit.ransac_output));
}

template <typename Archive, typename InlierMetric, typename ConsensusMetric,
          typename GrouperFunction>
inline void serialize(Archive &archive,
                      GenericRansacStrategy<InlierMetric, ConsensusMetric,
                                            GrouperFunction> &strategy,
                      const std::uint32_t) {
  archive(cereal::make_nvp("inlier_metric", strategy.inlier_metric_));
  archive(cereal::make_nvp("consensus_metric", strategy.consensus_metric_));
  archive(cereal::make_nvp("grouper_function", strategy.grouper_function_));
}

template <typename Archive, typename InlierMetric, typename ConsensusMetric,
          typename GrouperFunction>
inline void
serialize(Archive &archive,
          GaussianProcessRansacStrategy<InlierMetric, ConsensusMetric,
                                        GrouperFunction> &strategy,
          const std::uint32_t) {
  archive(cereal::make_nvp("inlier_metric", strategy.inlier_metric_));
  archive(cereal::make_nvp("consensus_metric", strategy.consensus_metric_));
  archive(cereal::make_nvp("grouper_function", strategy.grouper_function_));
}

} // namespace cereal

#endif /* THIRD_PARTY_ALBATROSS_INCLUDE_ALBATROSS_SRC_CEREAL_RANSAC_HPP_ */
