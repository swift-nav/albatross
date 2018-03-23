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

#ifndef ALBATROSS_MAP_UTILS_H
#define ALBATROSS_MAP_UTILS_H

#include <functional>
#include <map>
#include <unordered_map>
#include <vector>

namespace albatross {

/*
 * Convenience function instead of using the find == end
 * method of determining if a key exists in a map.
 */
template <typename K, typename V>
bool map_contains(const std::map<K, V> &m, const K &k) {
  return m.find(k) != m.end();
}

template <typename K, typename V>
bool map_contains(const std::unordered_map<K, V> &m, const K &k) {
  return m.find(k) != m.end();
}

/*
 * A function which makes a standard library map function
 * like a default map.  In particular this looks for a key
 * in the map and returns the value if that key exists.
 * If the key doesn't exist a new object of the value type
 * is inserted into the map, then returned.
 */
template <typename K, typename V>
V map_get_or_construct(const std::map<K, V> &m, const K &k) {
  if (!map_contains(m, k)) {
    V default_value = V();
    return default_value;
  }
  return m.at(k);
}

/*
 * Returns a vector consisting of all the keys in a map.
 */
template <typename K, typename V>
std::vector<K> map_keys(const std::map<K, V> m) {
  std::vector<K> keys;
  for (const auto &pair : m) {
    keys.push_back(pair.first);
  }
  return keys;
}

template <typename K, typename V>
std::map<K, V> map_join(const std::map<K, V> m, const std::map<K, V> other) {
  std::map<K, V> join;
  // Note the order here is reversed since insert will not insert if a key
  // already exists, in this case we want the result to contain all elements of
  // m overwritten by any elements in other.
  join.insert(other.begin(), other.end());
  join.insert(m.begin(), m.end());
  return join;
}
}
#endif
