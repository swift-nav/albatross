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
inline std::vector<ValueType> filter(const std::vector<ValueType> &values,
                                     ToKeepFunction &&to_keep) {
  std::vector<ValueType> output;
  std::copy_if(values.begin(), values.end(), std::back_inserter(output),
               std::forward<ToKeepFunction>(to_keep));
  return output;
}

// Set

template <typename ToKeepFunction, typename ValueType,
          typename std::enable_if<details::is_valid_value_only_filter_function<
                                      ToKeepFunction, ValueType>::value,
                                  int>::type = 0>
inline std::set<ValueType> filter(const std::set<ValueType> &values,
                                  ToKeepFunction &&to_keep) {
  std::set<ValueType> output;
  std::copy_if(values.begin(), values.end(),
               std::inserter(output, output.begin()),
               std::forward<ToKeepFunction>(to_keep));
  return output;
}

// Map

template <template <typename...> class Map, typename KeyType,
          typename ValueType, typename ToKeepFunction,
          typename std::enable_if<details::is_valid_value_only_filter_function<
                                      ToKeepFunction, ValueType>::value,
                                  int>::type = 0>
inline Grouped<KeyType, ValueType> filter_map(
    const Map<KeyType, ValueType> &map, ToKeepFunction &&to_keep) {
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
inline Grouped<KeyType, ValueType> filter_map(
    const Map<KeyType, ValueType> &map, ToKeepFunction &&to_keep) {
  Grouped<KeyType, ValueType> output;
  for (const auto &pair : map) {
    if (to_keep(pair.first, pair.second)) {
      output.emplace(pair.first, pair.second);
    }
  }
  return output;
}

template <typename KeyType, typename ValueType, typename ToKeepFunction>
inline auto filter(const std::map<KeyType, ValueType> &map,
                   ToKeepFunction &&f) {
  return filter_map(map, std::forward<ToKeepFunction>(f));
}

template <typename KeyType, typename ValueType, typename ToKeepFunction>
inline auto filter(const Grouped<KeyType, ValueType> &map, ToKeepFunction &&f) {
  return filter_map(map, std::forward<ToKeepFunction>(f));
}

// Dataset

template <typename ToKeepFunction, typename FeatureType,
          typename std::enable_if<details::is_valid_value_only_filter_function<
                                      ToKeepFunction, FeatureType>::value,
                                  int>::type = 0>
inline RegressionDataset<FeatureType> filter(
    const RegressionDataset<FeatureType> &dataset, ToKeepFunction &&to_keep) {
  std::vector<std::size_t> inds_to_keep;
  for (std::size_t i = 0; i < dataset.size(); ++i) {
    if (to_keep(dataset.features[i])) {
      inds_to_keep.emplace_back(i);
    }
  }
  return subset(dataset, inds_to_keep);
}

}  // namespace albatross

#endif /* ALBATROSS_INDEXING_FILTER_HPP_ */
