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
  ALBATROSS_ASSERT(
      x < static_cast<double>(std::numeric_limits<std::size_t>::max()));
  return static_cast<std::size_t>(x);
}

inline std::vector<double> linspace(double a, double b, std::size_t n) {
  double step = (b - a) / cast::to_double(n - 1);
  std::vector<double> xs(n);
  double val = a;
  for (auto &x : xs) {
    x = val;
    val += step;
  }
  return xs;
}

inline bool all(const std::vector<bool> &xs) {
  // The thinking on having all({}) return true comes from interpreting "all" as
  // "is there anything that isn't true".
  if (xs.size() == 0) {
    return true;
  }
  // Due to optimizations of vector<bool> you can't get actual references so we
  // copy here to avoid future issues.
  for (const auto x : xs) {
    if (!x) {
      return false;
    }
  }
  return true;
}

inline bool any(const std::vector<bool> &xs) {
  // The thinking on having any({}) return false comes from interpreting "any"
  // as "is there at least one true"
  if (xs.size() == 0) {
    return false;
  }
  // Due to optimizations of vector<bool> you can't get actual references so we
  // copy here to avoid future issues.
  for (const auto x : xs) {
    if (x) {
      return true;
    }
  }
  return false;
}

template <typename X>
inline bool vector_contains(const std::vector<X> &vector, const X &x) {
  return std::find(vector.begin(), vector.end(), x) != vector.end();
}

} // namespace albatross

#endif /* ALBATROSS_UTILS_VECTOR_UTILS_HPP_ */
