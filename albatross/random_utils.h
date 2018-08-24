/*
 * Copyright (C) 2018 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_RANDOM_UTILS_H
#define ALBATROSS_RANDOM_UTILS_H

#include "core/indexing.h"
#include <random>
#include <set>

/*
 * Samples integers between low and high (inclusive) with replacement.
 */
inline std::vector<std::size_t>
randint_without_replacement(std::size_t n, std::size_t low, std::size_t high,
                            std::default_random_engine &gen) {
  assert(n >= 0);
  assert(n <= (high - low));

  std::uniform_int_distribution<std::size_t> dist(low, high);
  std::set<int> samples;
  while (samples.size() < n) {
    samples.insert(dist(gen));
  }
  return std::vector<std::size_t>(samples.begin(), samples.end());
}

#endif
