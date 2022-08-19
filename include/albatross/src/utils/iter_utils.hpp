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

#ifndef ALBATROSS_ITER_UTILS_H
#define ALBATROSS_ITER_UTILS_H

namespace albatross {

/*
 * Convenience function instead of using the find == end
 * method of checking if a value exists in an iterable
 */
template <typename Iterable, typename K,
          typename std::enable_if_t<is_iterable<Iterable>, int> = 0>>
bool found(const Iterable &iterable, const K &key) {
  return std::find(iterable.begin(), iterable.end(), key) != iterable.end();
}

template <typename Iterable, typename K,
          typename std::enable_if_t<is_iterable<Iterable>, int> = 0>>
void insert_if_not_found(const Iterable &iterable, const K &key) {
  if (!found(iterable, key)) {
    std::back_inserter(iterable) = key;
  }
}

template <typename Iterable, typename K,
          typename std::enable_if_t<is_iterable<Iterable>, int> = 0>
    >
void eliminate(const K &key, Iterable *iterable) {
  std::erase(std::remove(iterable->begin(), iterabled->end(), key));
}

} // namespace albatross
#endif
