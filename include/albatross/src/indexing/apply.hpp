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

#ifndef ALBATROSS_INDEXING_APPLY_HPP_
#define ALBATROSS_INDEXING_APPLY_HPP_

namespace albatross {

// Vector

template <typename ValueType, typename ApplyFunction,
          typename ApplyType = typename details::value_only_apply_result<
              ApplyFunction, ValueType>::type,
          typename std::enable_if<details::is_valid_value_only_apply_function<
                                      ApplyFunction, ValueType>::value &&
                                      std::is_same<void, ApplyType>::value,
                                  int>::type = 0>
inline void apply(const std::vector<ValueType> &xs, ApplyFunction &&f) {
  std::for_each(xs.begin(), xs.end(), std::forward<ApplyFunction>(f));
}

template <typename ValueType, typename ApplyFunction,
          typename ApplyType = typename details::value_only_apply_result<
              ApplyFunction, ValueType>::type,
          typename std::enable_if<details::is_valid_value_only_apply_function<
                                      ApplyFunction, ValueType>::value &&
                                      !std::is_same<void, ApplyType>::value,
                                  int>::type = 0>
inline std::vector<ApplyType> apply(const std::vector<ValueType> &xs,
                                    ApplyFunction &&f) {
  std::vector<ApplyType> output(xs.size());
  std::transform(xs.begin(), xs.end(), output.begin(),
                 std::forward<ApplyFunction>(f));
  return output;
}

// Generic Map

template <
    template <typename...> class Map, typename KeyType, typename ValueType,
    typename ApplyFunction,
    typename ApplyType = typename details::key_value_apply_result<
        ApplyFunction, KeyType, ValueType>::type,
    typename std::enable_if<details::is_valid_key_value_apply_function<
                                ApplyFunction, KeyType, ValueType>::value &&
                                std::is_same<void, ApplyType>::value,
                            int>::type = 0>
inline void apply_map(const Map<KeyType, ValueType> &map, ApplyFunction &&f) {
  for (const auto &pair : map) {
    f(pair.first, pair.second);
  }
}

template <
    template <typename...> class Map, typename KeyType, typename ValueType,
    typename ApplyFunction,
    typename ApplyType = typename details::key_value_apply_result<
        ApplyFunction, KeyType, ValueType>::type,
    typename std::enable_if<details::is_valid_key_value_apply_function<
                                ApplyFunction, KeyType, ValueType>::value &&
                                !std::is_same<void, ApplyType>::value,
                            int>::type = 0>
inline Grouped<KeyType, ApplyType> apply_map(const Map<KeyType, ValueType> &map,
                                             ApplyFunction &&f) {
  Grouped<KeyType, ApplyType> output;
  for (const auto &pair : map) {
    output.emplace(pair.first, f(pair.first, pair.second));
  }
  return output;
}

template <template <typename...> class Map, typename KeyType,
          typename ValueType, typename ApplyFunction,
          typename ApplyType = typename details::value_only_apply_result<
              ApplyFunction, ValueType>::type,
          typename std::enable_if<details::is_valid_value_only_apply_function<
                                      ApplyFunction, ValueType>::value &&
                                      !std::is_same<void, ApplyType>::value,
                                  int>::type = 0>
inline Grouped<KeyType, ApplyType> apply_map(const Map<KeyType, ValueType> &map,
                                             ApplyFunction &&f) {
  Grouped<KeyType, ApplyType> output;
  for (const auto &pair : map) {
    output.emplace(pair.first, f(pair.second));
  }
  return output;
}

template <template <typename...> class Map, typename KeyType,
          typename ValueType, typename ApplyFunction,
          typename ApplyType = typename details::value_only_apply_result<
              ApplyFunction, ValueType>::type,
          typename std::enable_if<details::is_valid_value_only_apply_function<
                                      ApplyFunction, ValueType>::value &&
                                      std::is_same<void, ApplyType>::value,
                                  int>::type = 0>
inline void apply_map(const Map<KeyType, ValueType> &map, ApplyFunction &&f) {
  for (const auto &pair : map) {
    f(pair.second);
  }
}

template <typename KeyType, typename ValueType, typename ApplyFunction>
inline auto apply(const std::map<KeyType, ValueType> &map, ApplyFunction &&f) {
  return apply_map(map, std::forward<ApplyFunction>(f));
}

template <typename KeyType, typename ValueType, typename ApplyFunction>
inline auto apply(const Grouped<KeyType, ValueType> &map, ApplyFunction &&f) {
  return apply_map(map, std::forward<ApplyFunction>(f));
}

}  // namespace albatross

#endif /* ALBATROSS_INDEXING_APPLY_HPP_ */
