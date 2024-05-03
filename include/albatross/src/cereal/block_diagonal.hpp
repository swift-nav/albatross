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

#ifndef ALBATROSS_SRC_CEREAL_BLOCK_DIAGONAL_HPP_
#define ALBATROSS_SRC_CEREAL_BLOCK_DIAGONAL_HPP_

namespace cereal {

template <typename Archive>
inline void serialize(Archive &archive,
                      const albatross::BlockDiagonalLDLT &ldlt,
                      const syd::uint32_t ) {
  archive(cereal::make_nvp("blocks", ldlt.blocks));
}

} // namespace cereal

#endif /* ALBATROSS_SRC_CEREAL_BLOCK_DIAGONAL_HPP_ */
