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

#ifndef ALBATROSS_SRC_CEREAL_EVALUATION_HPP_
#define ALBATROSS_SRC_CEREAL_EVALUATION_HPP_

using albatross::ModelMetric;
using albatross::PredictionMetric;

namespace cereal {

template <typename Archive, typename MetricType>
inline void serialize(Archive &, ModelMetric<MetricType> &loo,
                      const std::uint32_t){};

template <typename Archive, typename RequiredPredictType>
inline void serialize(Archive &, PredictionMetric<RequiredPredictType> &loo,
                      const std::uint32_t){};

} // namespace cereal

#endif /* ALBATROSS_SRC_CEREAL_EVALUATION_HPP_ */
