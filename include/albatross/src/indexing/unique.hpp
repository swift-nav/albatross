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

#ifndef ALBATROSS_INDEXING_UNIQUE_HPP_
#define ALBATROSS_INDEXING_UNIQUE_HPP_

namespace albatross {

template <typename ValueType>
inline std::set<ValueType> unique_values(const std::vector<ValueType> &values) {
  return std::set<ValueType>(values.begin(), values.end());
}

template <typename ValueType, typename ApplyFunction,
          typename ApplyType = typename details::value_only_apply_result<
              ApplyFunction, ValueType>::type,
          typename std::enable_if<details::is_valid_value_only_apply_function<
                                      ApplyFunction, ValueType>::value &&
                                      !std::is_same<void, ApplyType>::value,
                                  int>::type = 0>
inline std::set<ApplyType> unique_values(const std::vector<ValueType> &xs,
                                         ApplyFunction &&f) {
  std::set<ApplyType> output;
  std::transform(xs.begin(), xs.end(), std::inserter(output, output.begin()),
                 std::forward<ApplyFunction>(f));
  return output;
}

template <typename ValueType, typename ApplyFunction,
          typename ApplyType = typename details::value_only_apply_result<
              ApplyFunction, ValueType>::type,
          typename std::enable_if<details::is_valid_value_only_apply_function<
                                      ApplyFunction, ValueType>::value &&
                                      !std::is_same<void, ApplyType>::value,
                                  int>::type = 0>
inline std::set<ApplyType> unique_values(const std::set<ValueType> &xs,
                                         ApplyFunction &&f) {
  std::set<ApplyType> output;
  std::transform(xs.begin(), xs.end(), std::inserter(output, output.begin()),
                 std::forward<ApplyFunction>(f));
  return output;
}

// unique_value
//   assumes there is one single value in the iterator
//   and will return that value or assert if that assumption is false

template <typename ValueType>
inline ValueType unique_value(const std::set<ValueType> &values) {
  assert(values.size() == 1 && "expected a single unique value");
  return *values.begin();
}

template <typename ValueType>
inline ValueType unique_value(const std::vector<ValueType> &values) {
  const auto unique_vals = unique_values(values);
  return unique_value<ValueType>(unique_vals);
}

template <typename ValueType, typename ApplyFunction,
          typename ApplyType = typename details::value_only_apply_result<
              ApplyFunction, ValueType>::type,
          typename std::enable_if<details::is_valid_value_only_apply_function<
                                      ApplyFunction, ValueType>::value &&
                                      !std::is_same<void, ApplyType>::value,
                                  int>::type = 0>
inline ApplyType unique_value(const std::vector<ValueType> &xs,
                              ApplyFunction &&f) {
  return unique_value(unique_values(xs, std::forward<ApplyFunction>(f)));
}

} // namespace albatross

#endif /* ALBATROSS_INDEXING_UNIQUE_HPP_ */
