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

#ifndef ALBATROSS_SRC_CEREAL_REPRESENTATIONS_HPP_
#define ALBATROSS_SRC_CEREAL_REPRESENTATIONS_HPP_

namespace cereal {

template <typename Archive>
inline void serialize(Archive &archive,
                      albatross::ExplainedCovariance &explained,
                      const std::uint32_t) {
  archive(cereal::make_nvp("outer_ldlt", explained.outer_ldlt),
          cereal::make_nvp("inner", explained.inner));
}

template <typename Archive>
inline void serialize(Archive &archive, albatross::DirectInverse &direct,
                      const std::uint32_t) {
  archive(cereal::make_nvp("inverse", direct.inverse_));
}

}  // namespace cereal
#endif /* THIRD_PARTY_ALBATROSS_INCLUDE_ALBATROSS_SRC_CEREAL_REPRESENTATIONS_HPP_ \
        */
