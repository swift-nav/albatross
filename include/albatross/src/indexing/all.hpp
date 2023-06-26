/*
 * Copyright (C) 2023 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_INDEXING_ALL_HPP_
#define ALBATROSS_INDEXING_ALL_HPP_

namespace albatross {

// Vector

template <typename ValueType, typename BoolFunction,
          typename std::enable_if<details::is_valid_value_only_filter_function<
                                      BoolFunction, ValueType>::value,
                                  int>::type = 0>
inline bool all(const std::vector<ValueType> &xs, BoolFunction &&f) {
  return std::all_of(xs.begin(), xs.end(), std::forward<BoolFunction>(f));
}

// Maps

namespace detail {
template <
    template <typename...> class Map, typename KeyType, typename ValueType,
    typename BoolFunction,
    typename std::enable_if<details::is_valid_key_value_filter_function<
                                BoolFunction, KeyType, ValueType>::value,
                            int>::type = 0>
inline bool all_generic_map(const Map<KeyType, ValueType> &map, BoolFunction &&f) {
  auto split_pair = [&f](const auto &pair) {
    return f(pair.first, pair.second);
  };
  return std::all_of(map.begin(), map.end(), split_pair);
}
}

template <typename KeyType, typename ValueType, typename BoolFunction>
inline bool all(const std::map<KeyType, ValueType> &map, BoolFunction &&f) {
  return all_generic_map(map, std::forward<BoolFunction>(f));
}

template <typename KeyType, typename ValueType, typename BoolFunction>
inline bool all(const Grouped<KeyType, ValueType> &map, BoolFunction &&f) {
  return all_generic_map(map, std::forward<BoolFunction>(f));
}

} // namespace albatross

#endif /* ALBATROSS_INDEXING_APPLY_HPP_ */
