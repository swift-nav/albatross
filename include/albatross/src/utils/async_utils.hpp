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

namespace detail {

inline bool should_serial_apply(ThreadPool *pool) {
  return (nullptr == pool) || (pool->thread_count() <= 1);
}

} // namespace detail

template <typename ValueType, typename ApplyFunction,
          typename ApplyType = typename details::value_only_apply_result<
              ApplyFunction, ValueType>::type,
          typename std::enable_if<details::is_valid_value_only_apply_function<
                                      ApplyFunction, ValueType>::value &&
                                      std::is_same<void, ApplyType>::value,
                                  int>::type = 0>
inline void apply(const std::vector<ValueType> &xs, const ApplyFunction &func,
                  ThreadPool *pool) {
  if (detail::should_serial_apply(pool)) {
    apply(xs, func);
    return;
  }

  std::vector<std::future<void>> futures;
  for (const auto &x : xs) {
    futures.emplace_back(pool->enqueue(func, x));
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
inline auto apply(const std::vector<ValueType> &xs, const ApplyFunction &func,
                  ThreadPool *pool) {
  if (detail::should_serial_apply(pool)) {
    return apply(xs, func);
  }

  std::vector<std::future<ApplyType>> futures;
  for (const auto &x : xs) {
    futures.emplace_back(pool->enqueue(func, x));
  }

  std::vector<ApplyType> output;
  for (auto &f : futures) {
    output.emplace_back(f.get());
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
inline void apply_map(const Map<KeyType, ValueType> &xs,
                      const ApplyFunction &func, ThreadPool *pool) {
  if (detail::should_serial_apply(pool)) {
    apply_map(xs, func);
    return;
  }

  std::vector<std::future<void>> futures;
  for (const auto &x : xs) {
    futures.emplace_back(pool->enqueue(func, x.second));
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
inline Grouped<KeyType, ApplyType> apply_map(const Map<KeyType, ValueType> &xs,
                                             const ApplyFunction &func,
                                             ThreadPool *pool) {
  if (detail::should_serial_apply(pool)) {
    return apply_map(xs, func);
  }

  Grouped<KeyType, std::future<ApplyType>> futures;
  for (const auto &x : xs) {
    futures[x.first] = pool->enqueue(func, x.second);
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
inline void apply_map(const Map<KeyType, ValueType> &xs,
                      const ApplyFunction &func, ThreadPool *pool) {
  if (detail::should_serial_apply(pool)) {
    apply_map(xs, func);
    return;
  }

  std::vector<std::future<void>> futures;
  for (const auto &x : xs) {
    futures.emplace_back(pool->enqueue(func, x.first, x.second));
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
inline Grouped<KeyType, ApplyType> apply_map(const Map<KeyType, ValueType> &xs,
                                             const ApplyFunction &func,
                                             ThreadPool *pool) {
  if (detail::should_serial_apply(pool)) {
    return apply_map(xs, func);
  }

  Grouped<KeyType, std::future<ApplyType>> futures;
  for (const auto &x : xs) {
    futures[x.first] = pool->enqueue(func, x.first, x.second);
  }

  Grouped<KeyType, ApplyType> output;
  for (auto &f : futures) {
    output[f.first] = f.second.get();
  }
  return output;
}

template <typename KeyType, typename ValueType, typename ApplyFunction>
inline auto apply(const std::map<KeyType, ValueType> &map, ApplyFunction &&f,
                  ThreadPool *pool) {
  return apply_map(map, std::forward<ApplyFunction>(f), pool);
}

template <typename KeyType, typename ValueType, typename ApplyFunction>
inline auto apply(const Grouped<KeyType, ValueType> &map, ApplyFunction &&f,
                  ThreadPool *pool) {
  return apply_map(map, std::forward<ApplyFunction>(f), pool);
}

// Returns the number of threads that the hardware supports cores, or
// 1 if there was a problem calculating that..
inline std::size_t get_default_thread_count() {
  // This standard function is not guaranteed to return nonzero.
  return std::max(std::size_t{1},
                  std::size_t{std::thread::hardware_concurrency()});
}

// A thread pool object that performs normal serial evaluation.
static constexpr std::nullptr_t serial_thread_pool = nullptr;

// Returns a thread pool.  By default, the thread pool has
// `get_default_thread_count()` threads.
inline std::shared_ptr<ThreadPool>
make_shared_thread_pool(std::size_t num_threads = 0,
                        std::size_t stack_size = 0) {
  if (num_threads == 1) {
    return serial_thread_pool;
  }

  if (num_threads < 1) {
    num_threads = get_default_thread_count();
  }

#if defined(EIGEN_USE_MKL_ALL) || defined(EIGEN_USE_MKL_VML)
  const auto init = []() { mkl_set_num_threads_local(1); };
#else  // EIGEN_USE_MKL_ALL || EIGEN_USE_MKL_VML
  const auto init = []() {};
#endif // EIGEN_USE_MKL_ALL || EIGEN_USE_MKL_VML

  return std::make_shared<ThreadPool>(num_threads, init, stack_size);
}

inline std::size_t get_thread_count(const ThreadPool *pool) {
  if (nullptr == pool) {
    return 1;
  }

  return pool->thread_count();
}

inline std::size_t get_thread_count(const std::shared_ptr<ThreadPool> &pool) {
  if (nullptr == pool) {
    return 1;
  }

  return pool->thread_count();
}

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_SRC_UTILS_ASYNC_UTILS_HPP_ */
