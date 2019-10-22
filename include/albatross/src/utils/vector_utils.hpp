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

#ifndef ALBATROSS_UTILS_VECTOR_UTILS_HPP_
#define ALBATROSS_UTILS_VECTOR_UTILS_HPP_

namespace albatross {

inline std::size_t safe_cast_to_size_t(double x) {
  assert(x < std::numeric_limits<std::size_t>::max());
  return static_cast<std::size_t>(x);
}

inline std::vector<double> linspace(double a, double b, std::size_t n) {
  double step = (b - a) / static_cast<double>(n - 1);
  std::vector<double> xs(n);
  double val = a;
  for (auto &x : xs) {
    assert(val <= b);
    x = val;
    val += step;
  }
  return xs;
}

} // namespace albatross

#endif /* ALBATROSS_UTILS_VECTOR_UTILS_HPP_ */
