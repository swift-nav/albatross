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
template <typename Map, typename K,
          std::enable_if_t<has_find_key_v<Map, K>, int> = 0>
inline bool map_contains(const Map &m, const K &k) {
  return m.find(k) != m.end();
}

/*
 * Convenience function which returns m.at(k) unless k is not in
 * the map in which case it uses either the default constructor
 * or the provided alternate value.
 */
template <template <typename...> class Map, typename K, typename SearchKey,
          typename V, typename... More,
          typename = std::enable_if_t<
              has_find_key_v<Map<K, V, More...>, SearchKey>, void>>
inline V map_at_or(const Map<K, V, More...> &m, const SearchKey &k,
                   const V &value_if_missing = V()) {
  const auto iter = m.find(k);
  if (iter == m.end()) {
    return value_if_missing;
  }
  return iter->second;
}

/*
 * Returns a vector consisting of all the keys in a map.
 */
template <template <typename...> class Map, typename K, typename V,
          typename... More>
inline std::vector<K> map_keys(const Map<K, V, More...> &m) {
  std::vector<K> keys;
  keys.reserve(m.size());
  for (const auto &[key, _] : m) {
    keys.push_back(key);
  }
  return keys;
}

/*
 * Returns a vector consisting of all the values in a map.
 */
template <template <typename...> class Map, typename K, typename V,
          typename... More>
inline std::vector<V> map_values(const Map<K, V, More...> &m) {
  std::vector<V> values;
  values.reserve(m.size());
  for (const auto &[_, value] : m) {
    values.push_back(value);
  }
  return values;
}

template <template <typename...> class Map, typename K, typename V,
          typename... More>
inline Map<K, V, More...> map_join(const Map<K, V, More...> &m,
                                   const Map<K, V, More...> &other) {
  Map<K, V, More...> join(other);
  // Note the order here is reversed since insert will not insert if a key
  // already exists, in this case we want the result to contain all elements of
  // m overwritten by any elements in other.
  join.insert(m.begin(), m.end());
  return join;
}

template <template <typename...> class Map, typename K, typename V,
          typename... More>
inline Map<K, V, More...> map_join_strict(const Map<K, V, More...> &m,
                                          const Map<K, V, More...> &other) {
  Map<K, V, More...> join(other);
  // Note the order here is reversed since insert will not insert if a key
  // already exists, in this case we want the result to contain all elements of
  // m overwritten by any elements in other.
  for (const auto &[key, value] : m) {
    if (map_contains(join, key)) {
      // duplicate key found in map_join.
      ALBATROSS_ASSERT(false && "duplicate key found");
    }
    join[key] = value;
  }
  return join;
}

} // namespace albatross
#endif
