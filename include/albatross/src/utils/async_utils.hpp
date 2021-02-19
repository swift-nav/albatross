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
inline void async_apply(const std::vector<ValueType> &xs, const ApplyFunction &func) {
  std::vector<std::future<void>> futures;
  for (const auto &x : xs) {
    futures.emplace_back(async_safe(func, x));
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
inline auto async_apply(const std::vector<ValueType> &xs, const ApplyFunction &func) {
  std::vector<std::future<ApplyType>> futures;
  for (const auto &x : xs) {
    futures.emplace_back(async_safe(func, x));
  }

  std::vector<ApplyType> output;
  for (auto &f : futures) {
    output.emplace_back(f.get());
  }
  return output;
}

template <typename ValueType, typename ApplyFunction,
          typename ApplyType = typename details::value_only_apply_result<
              ApplyFunction, ValueType>::type,
          typename std::enable_if<details::is_valid_value_only_apply_function<
                                      ApplyFunction, ValueType>::value &&
                                      std::is_same<void, ApplyType>::value,
                                  int>::type = 0>
inline void
async_apply(const std::vector<ValueType> &xs, const ApplyFunction &func,
                 std::size_t n_proc) {

  assert(n_proc > 1);

  std::vector<std::size_t> inds(xs.size());
  std::iota(std::begin(inds), std::end(inds), 0);

  auto get_group = [&](std::size_t i) {
    return i % n_proc;
  };

  auto process_group = [&](const std::vector<std::size_t> &group_inds) {
    for (const auto &ind : group_inds) {
      func(xs[ind]);
    }
  };

  group_by(inds, get_group).async_apply(process_group);
}

template <typename ValueType, typename ApplyFunction,
          typename ApplyType = typename details::value_only_apply_result<
              ApplyFunction, ValueType>::type,
          typename std::enable_if<details::is_valid_value_only_apply_function<
                                      ApplyFunction, ValueType>::value &&
                                      !std::is_same<void, ApplyType>::value,
                                  int>::type = 0>
inline auto
async_apply(const std::vector<ValueType> &xs, const ApplyFunction &func,
                 std::size_t n_proc) {

  assert(n_proc > 1);

  std::vector<std::size_t> inds(xs.size());
  std::iota(std::begin(inds), std::end(inds), 0);

  auto get_group = [&](std::size_t i) {
    return i % n_proc;
  };

  std::mutex output_mutex;
  std::vector<ApplyType> output;

  auto process_group = [&](const std::vector<std::size_t> &group_inds) {
    for (const auto &i : group_inds) {
      const ApplyType output_i = func(xs[i]);
      {
        const std::lock_guard<std::mutex> lock(output_mutex);
        output[i] = output_i;
      }
    }
  };

  group_by(inds, get_group).async_apply(process_group);

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
inline void async_apply_map(const Map<KeyType, ValueType> &xs,
                            const ApplyFunction &func) {
  std::vector<std::future<void>> futures;
  for (const auto &x : xs) {
    futures.emplace_back(async_safe(func, x.second));
  }
  for (auto &f : futures) {
    f.get();
  }
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
async_apply_map(const Map<KeyType, ValueType> &xs, const ApplyFunction &func) {
  Grouped<KeyType, std::future<ApplyType>> futures;
  for (const auto &x : xs) {
    futures[x.first] = async_safe(func, x.second);
  }

  Grouped<KeyType, ApplyType> output;
  for (auto &f : futures) {
    output[f.first] = f.second.get();
  }
  return output;
}

template <
    template <typename...> class Map, typename KeyType, typename ValueType,
    typename ApplyFunction,
    typename ApplyType = typename details::key_value_apply_result<
        ApplyFunction, KeyType, ValueType>::type,
    typename std::enable_if<details::is_valid_key_value_apply_function<
                                ApplyFunction, KeyType, ValueType>::value &&
                                std::is_same<void, ApplyType>::value,
                            int>::type = 0>
inline void async_apply_map(const Map<KeyType, ValueType> &xs,
                            const ApplyFunction &func) {
  std::vector<std::future<void>> futures;
  for (const auto &x : xs) {
    futures.emplace_back(async_safe(func, x.first, x.second));
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
async_apply_map(const Map<KeyType, ValueType> &xs, const ApplyFunction &func) {
  Grouped<KeyType, std::future<ApplyType>> futures;
  for (const auto &x : xs) {
    futures[x.first] = async_safe(func, x.first, x.second);
  }

  Grouped<KeyType, ApplyType> output;
  for (auto &f : futures) {
    output[f.first] = f.second.get();
  }
  return output;
}

template <typename KeyType, typename ValueType, typename ApplyFunction>
inline auto async_apply(const std::map<KeyType, ValueType> &map,
                        ApplyFunction &&f) {
  return async_apply_map(map, std::forward<ApplyFunction>(f));
}

template <typename KeyType, typename ValueType, typename ApplyFunction>
inline auto async_apply(const Grouped<KeyType, ValueType> &map,
                        ApplyFunction &&f) {
  return async_apply_map(map, std::forward<ApplyFunction>(f));
}

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_SRC_UTILS_ASYNC_UTILS_HPP_ */
