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

namespace albatross {

/*
 * Convenience function instead of using the find == end
 * method of determining if a key exists in a map.
 */
template <template <typename...> class Map, typename K, typename V>
bool map_contains(const Map<K, V> &m, const K &k) {
  return m.find(k) != m.end();
}

/*
 * A function which makes a standard library map function
 * like a default map.  In particular this looks for a key
 * in the map and returns the value if that key exists.
 * If the key doesn't exist a new object of the value type
 * is inserted into the map, then returned.
 */
template <template <typename...> class Map, typename K, typename V>
V map_get_or_construct(const Map<K, V> &m, const K &k) {
  if (!map_contains(m, k)) {
    V default_value = V();
    return default_value;
  }
  return m.at(k);
}

/*
 * Returns a vector consisting of all the keys in a map.
 */
template <template <typename...> class Map, typename K, typename V>
std::vector<K> map_keys(const Map<K, V> &m) {
  std::vector<K> keys;
  for (const auto &pair : m) {
    keys.push_back(pair.first);
  }
  return keys;
}

/*
 * Returns a vector consisting of all the values in a map.
 */
template <template <typename...> class Map, typename K, typename V>
std::vector<V> map_values(const Map<K, V> &m) {
  std::vector<V> values;
  for (const auto &pair : m) {
    values.push_back(pair.second);
  }
  return values;
}

template <template <typename...> class Map, typename K, typename V>
Map<K, V> map_join(const Map<K, V> &m, const Map<K, V> &other) {
  Map<K, V> join(other);
  // Note the order here is reversed since insert will not insert if a key
  // already exists, in this case we want the result to contain all elements of
  // m overwritten by any elements in other.
  join.insert(m.begin(), m.end());
  return join;
}

template <template <typename...> class Map, typename K, typename V>
Map<K, V> map_join_strict(const Map<K, V> &m, const Map<K, V> &other) {
  Map<K, V> join(other);
  // Note the order here is reversed since insert will not insert if a key
  // already exists, in this case we want the result to contain all elements of
  // m overwritten by any elements in other.
  for (const auto &pair : m) {
    if (join.find(pair.first) != join.end()) {
      // duplicate key found in map_join.
      assert(false && "duplicate key found");
    }
    join[pair.first] = pair.second;
  }
  return join;
}

} // namespace albatross
#endif
