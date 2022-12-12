/*
 * Copyright (C) 2022 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef INCLUDE_ALBATROSS_SRC_DETAILS_TYPECAST_HPP_
#define INCLUDE_ALBATROSS_SRC_DETAILS_TYPECAST_HPP_

namespace albatross {

namespace cast {

constexpr std::size_t to_size(Eigen::Index input) {
  ALBATROSS_ASSERT(input >= 0);
  return static_cast<std::size_t>(input);
}

constexpr Eigen::Index to_index(std::size_t input) {
  ALBATROSS_ASSERT(input <= to_size(std::numeric_limits<Eigen::Index>::max()));
  return static_cast<Eigen::Index>(input);
}

// 2^53 is the largest value of type `double` with integer precision;
// therefore, values of type `double` above this limit are not
// guaranteed to be the same after being rounded from `std::size_t` or
// `int64_t` to `double` and back.  Using this constant, we disallow
// index-rounding in such situations.
constexpr std::size_t kMaxIntegerPrecisionDoubleSize = 1ULL << 53;

constexpr double to_double(std::size_t input) {
  ALBATROSS_ASSERT(input <= kMaxIntegerPrecisionDoubleSize);
  return static_cast<double>(input);
}

// See `kMaxIntegerPrecisionDoubleSize` above for more.
//
// We have to be sure to handle the case of 32-bit `Eigen::Index`
constexpr Eigen::Index kMaxIntegerPrecisionDoubleIndex =
    static_cast<Eigen::Index>(std::min(
        static_cast<long long>(std::numeric_limits<Eigen::Index>::max()),
        1LL << 53));

constexpr double to_double(Eigen::Index input) {
  ALBATROSS_ASSERT(input <= kMaxIntegerPrecisionDoubleIndex &&
                   input >= -kMaxIntegerPrecisionDoubleIndex);
  return static_cast<double>(input);
}

} // namespace cast

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_SRC_DETAILS_TYPECAST_HPP_ */
