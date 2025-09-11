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

#ifndef ALBATROSS_SRC_CEREAL_PRIORS_HPP_
#define ALBATROSS_SRC_CEREAL_PRIORS_HPP_

namespace cereal {

template <typename Archive>
inline void serialize(Archive &archive ALBATROSS_UNUSED,
                      albatross::Prior &prior ALBATROSS_UNUSED,
                      const std::uint32_t) {}

template <typename Archive>
inline void serialize(Archive &archive, albatross::UniformPrior &prior,
                      const std::uint32_t) {
  archive(cereal::make_nvp("lower", prior.lower_),
          cereal::make_nvp("upper", prior.upper_));
}

template <typename Archive>
inline void serialize(Archive &archive, albatross::GaussianPrior &prior,
                      const std::uint32_t) {
  archive(cereal::make_nvp("mu", prior.mu_),
          cereal::make_nvp("sigma", prior.sigma_));
}

template <typename Archive>
inline void serialize(Archive &archive, albatross::PositiveGaussianPrior &prior,
                      const std::uint32_t) {
  archive(cereal::make_nvp("mu", prior.mu_),
          cereal::make_nvp("sigma", prior.sigma_));
}

template <typename Archive>
inline void serialize(Archive &archive, albatross::LogNormalPrior &prior,
                      const std::uint32_t) {
  archive(cereal::make_nvp("mu", prior.mu_),
          cereal::make_nvp("sigma", prior.sigma_));
}

template <typename Archive>
inline void serialize(Archive &archive, albatross::PriorContainer &priors,
                      const std::uint32_t) {
  archive(cereal::make_nvp("container", priors.priors_));
}

} // namespace cereal

#endif /* THIRD_PARTY_ALBATROSS_INCLUDE_ALBATROSS_SRC_CEREAL_PRIORS_HPP_ */
