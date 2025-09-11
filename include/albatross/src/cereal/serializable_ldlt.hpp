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

#ifndef ALBATROSS_SRC_CEREAL_SERIALIZABLE_LDLT_HPP_
#define ALBATROSS_SRC_CEREAL_SERIALIZABLE_LDLT_HPP_

namespace cereal {

template <typename Archive>
inline void save(Archive &archive, const Eigen::SerializableLDLT &ldlt,
                 const std::uint32_t) {
  save_lower_triangle(archive, ldlt.matrix());
  archive(cereal::make_nvp("transpositions", ldlt.transpositionsP()),
          cereal::make_nvp("is_initialized", ldlt.is_initialized()));
}

template <typename Archive>
inline void load(Archive &archive, Eigen::SerializableLDLT &ldlt,
                 const std::uint32_t) {
  load_lower_triangle(archive, ldlt.mutable_matrix());
  archive(cereal::make_nvp("transpositions", ldlt.mutable_transpositions()),
          cereal::make_nvp("is_initialized", ldlt.mutable_is_initialized()));
}

} // namespace cereal

#endif /* ALBATROSS_SRC_CEREAL_SERIALIZABLE_LDLT_HPP_ */
