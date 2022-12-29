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

#ifndef ALBATROSS_SRC_CEREAL_BLOCK_UTILS_HPP
#define ALBATROSS_SRC_CEREAL_BLOCK_UTILS_HPP

namespace cereal {

template <typename Archive, typename Solver>
inline void serialize(Archive &archive,
                      albatross::BlockSymmetric<Solver> &block_sym,
                      const std::uint32_t) {
  archive(cereal::make_nvp("A", block_sym.A),
          cereal::make_nvp("Ai_B", block_sym.Ai_B),
          cereal::make_nvp("S", block_sym.S));
}

template <typename Archive>
inline void serialize(Archive &archive,
                      albatross::BlockDiagonalLDLT &block_ldlt,
                      const std::uint32_t) {
  archive(cereal::make_nvp("blocks", block_ldlt.blocks));
}

} // namespace cereal

#endif // ALBATROSS_SRC_CEREAL_BLOCK_UTILS_HPP
