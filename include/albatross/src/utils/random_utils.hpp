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

namespace albatross {
/*
 * Samples integers between low and high (inclusive) with replacement.
 * The inclusive part may seem a bit confusing, but it aligns with
 * uniform_int_distribution behavior.
 */
inline std::vector<std::size_t>
randint_without_replacement(std::size_t n, std::size_t low, std::size_t high,
                            std::default_random_engine &gen) {
  std::size_t n_choices = high - low + 1;
  if (n > n_choices) {
    std::cout << "ERROR: n (" << n << ") is larger than n_choices ("
              << n_choices << ")" << std::endl;
    assert(false);
  }

  if (n == n_choices) {
    std::vector<std::size_t> all_inds(n_choices);
    for (std::size_t i = 0; i < n; ++i) {
      all_inds[i] = i + low;
    }
    return all_inds;
  }

  if (n > n_choices / 2.) {
    // Since we're trying to randomly sample more than half of the
    // points it'll be faster to randomly sample which points we
    // should throw out than which ones we should keep.
    const auto to_throw_out =
        randint_without_replacement(n_choices - n, 0, n_choices - 1, gen);
    auto to_keep = indices_complement(to_throw_out, n_choices);

    if (low != 0) {
      for (auto &el : to_keep) {
        el += low;
      }
    }

    return to_keep;
  }

  std::uniform_int_distribution<std::size_t> dist(low, high);
  std::set<int> samples;
  while (samples.size() < n) {
    samples.insert(dist(gen));
  }
  return std::vector<std::size_t>(samples.begin(), samples.end());
}

template <typename X>
inline std::vector<X>
random_without_replacement(const std::vector<X> &xs, std::size_t n,
                           std::default_random_engine &gen) {
  std::vector<X> random_sample;
  for (const auto &i : randint_without_replacement(n, 0, xs.size() - 1, gen)) {
    random_sample.emplace_back(xs[i]);
  }
  return random_sample;
}

} // namespace albatross

#endif
