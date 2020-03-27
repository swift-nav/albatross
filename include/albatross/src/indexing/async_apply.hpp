/*
 * Copyright (C) 2020 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef INCLUDE_ALBATROSS_SRC_UTILS_ASYNC_UTILS_HPP_
#define INCLUDE_ALBATROSS_SRC_UTILS_ASYNC_UTILS_HPP_

namespace albatross {

// This method makes sure we don't accidentally call async with the
// default mode which has some flaws:
//
// https://eli.thegreenplace.net/2016/the-promises-and-challenges-of-stdasync-task-based-parallelism-in-c11/
template <typename F, typename... Ts>
inline auto async_safe(F &&f, Ts &&... params) {
  return std::async(std::launch::async, std::forward<F>(f),
                    std::forward<Ts>(params)...);
}

template <typename ValueType, typename ApplyFunction,
          typename ApplyType = typename details::value_only_apply_result<
              ApplyFunction, ValueType>::type,
          typename std::enable_if<details::is_valid_value_only_apply_function<
                                      ApplyFunction, ValueType>::value &&
                                      std::is_same<void, ApplyType>::value,
                                  int>::type = 0>
inline void async_apply(const std::vector<ValueType> &xs, ApplyFunction &&f) {
  std::vector<std::future<void>> futures;
  for (const auto &x : xs) {
    futures.emplace_back(async_safe(f, x));
  }
  for (auto &f : futures) {
    f.get();
  }
}

template <typename ValueType, typename ApplyFunction,
          typename ApplyType = typename details::value_only_apply_result<
              ApplyFunction, ValueType>::type,
          typename std::enable_if<details::is_valid_value_only_apply_function<
                                      ApplyFunction, ValueType>::value &&
                                      !std::is_same<void, ApplyType>::value,
                                  int>::type = 0>
inline auto async_apply(const std::vector<ValueType> &xs, ApplyFunction &&f) {
  std::vector<std::future<ApplyType>> futures;
  for (const auto &x : xs) {
    futures.emplace_back(async_safe(f, x));
  }

  std::vector<ApplyType> output;
  for (auto &f : futures) {
    output.emplace_back(f.get());
  }
  return output;
}

// Map

template <
    template <typename...> class Map, typename KeyType, typename ValueType,
    typename ApplyFunction,
    typename ApplyType = typename details::key_value_apply_result<
        ApplyFunction, KeyType, ValueType>::type,
    typename std::enable_if<details::is_valid_key_value_apply_function<
                                ApplyFunction, KeyType, ValueType>::value &&
                                std::is_same<void, ApplyType>::value,
                            int>::type = 0>
inline void async_apply(const Map<KeyType, ValueType> &map, ApplyFunction &&f) {
  std::vector<std::future<void>> futures;
  for (const auto &pair : map) {
    futures.emplace_back(async_safe(f, pair.first, pair.second));
  }
  for (auto &f : futures) {
    f.get();
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
inline Grouped<KeyType, ApplyType>
async_apply(const Map<KeyType, ValueType> &map, ApplyFunction &&f) {

  std::map<KeyType, std::future<ApplyType>> futures;
  for (const auto &pair : map) {
    futures[pair.first] = async_safe(f, pair.first, pair.second);
  }

  Grouped<KeyType, ApplyType> output;
  for (auto &pair : futures) {
    output.emplace(pair.first, pair.second.get());
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
inline Grouped<KeyType, ApplyType>
async_apply(const Map<KeyType, ValueType> &map, ApplyFunction &&f) {

  std::map<KeyType, std::future<ApplyType>> futures;
  for (const auto &pair : map) {
    futures[pair.first] = async_safe(f, pair.second);
  }

  Grouped<KeyType, ApplyType> output;
  for (auto &pair : futures) {
    output.emplace(pair.first, pair.second.get());
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
inline void async_apply(const Map<KeyType, ValueType> &map, ApplyFunction &&f) {
  std::vector<std::future<void>> futures;
  for (const auto &pair : map) {
    futures.emplace_back(async_safe(f, pair.second));
  }
  for (auto &f : futures) {
    f.get();
  }
}

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_SRC_UTILS_ASYNC_UTILS_HPP_ */
