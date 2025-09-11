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

#ifndef INCLUDE_ALBATROSS_SRC_CEREAL_LEAST_SQUARES_HPP_
#define INCLUDE_ALBATROSS_SRC_CEREAL_LEAST_SQUARES_HPP_

using albatross::Fit;
using albatross::LeastSquares;

namespace cereal {

template <typename Archive, typename ImplType>
void serialize(Archive &archive, Fit<LeastSquares<ImplType>> &fit,
               const std::uint32_t) {
  archive(fit.coefs);
}

}  // namespace cereal

#endif /* INCLUDE_ALBATROSS_SRC_CEREAL_LEAST_SQUARES_HPP_ */
