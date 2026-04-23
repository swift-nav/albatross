/*
 * Copyright (C) 2026 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_CEREAL_CHOLMOD_REPRESENTATION_HPP_
#define ALBATROSS_CEREAL_CHOLMOD_REPRESENTATION_HPP_

#include <cereal/types/memory.hpp>

namespace cereal {

template <typename Archive, typename Solver>
inline void serialize(Archive &archive,
                      albatross::CholmodCovariance<Solver> &rep,
                      const std::uint32_t) {
  archive(cereal::make_nvp("sparse_A", rep.sparse_A_),
          cereal::make_nvp("solver", rep.solver_));
}

} // namespace cereal

#endif /* ALBATROSS_CEREAL_CHOLMOD_REPRESENTATION_HPP_ */
