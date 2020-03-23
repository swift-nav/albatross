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

#ifndef ALBATROSS_INDEXING_FILTER_HPP_
#define ALBATROSS_INDEXING_FILTER_HPP_

namespace albatross {

// Vector

template <typename ToKeepFunction, typename ValueType,
          typename std::enable_if<details::is_valid_value_only_filter_function<
                                      ToKeepFunction, ValueType>::value,
                                  int>::type = 0>
inline auto filter(const std::vector<ValueType> &values,
                   ToKeepFunction to_keep) {
  std::vector<ValueType> output;
  for (const auto &v : values) {
    if (to_keep(v)) {
      output.emplace_back(v);
    }
  }
  return output;
}

// Map

template <template <typename...> class Map, typename KeyType,
          typename ValueType, typename ToKeepFunction,
          typename std::enable_if<details::is_valid_value_only_filter_function<
                                      ToKeepFunction, ValueType>::value,
                                  int>::type = 0>
auto filter(const Map<KeyType, ValueType> &map, ToKeepFunction to_keep) {
  Grouped<KeyType, ValueType> output;
  for (const auto &pair : map) {
    if (to_keep(pair.second)) {
      output.emplace(pair.first, pair.second);
    }
  }
  return output;
}

template <
    template <typename...> class Map, typename KeyType, typename ValueType,
    typename ToKeepFunction,
    typename std::enable_if<details::is_valid_key_value_filter_function<
                                ToKeepFunction, KeyType, ValueType>::value,
                            int>::type = 0>
auto filter(const Map<KeyType, ValueType> &map, ToKeepFunction to_keep) {
  Grouped<KeyType, ValueType> output;
  for (const auto &pair : map) {
    if (to_keep(pair.first, pair.second)) {
      output.emplace(pair.first, pair.second);
    }
  }
  return output;
}

} // namespace albatross

#endif /* ALBATROSS_INDEXING_FILTER_HPP_ */
