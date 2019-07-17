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

#ifndef ALBATROSS_CORE_CONCATENATE_HPP_
#define ALBATROSS_CORE_CONCATENATE_HPP_

namespace albatross {

namespace internal {

template <typename X, typename Y>
struct concatenation_type {
  using type = variant<X, Y>;
};

template <typename X>
struct concatenation_type<X, X> {
  using type = X;
};

//template <typename X, typename... Ts, typename std::enable_if_t<!is_variant<X>::value && is_in_variant<X, variant<Ts...>>::value>>
//struct concatenation_type<X, variant<Ts...> {
//  using type = variant<Ts...>;
//};
//
//template <typename X, typename... Ts, typename std::enable_if_t<!is_variant<X>::value && !is_in_variant<X, variant<Ts...>>::value>>
//struct concatenation_type<X, variant<Ts...> {
//  using type = variant<Ts...>;
//};

}

template <typename X, typename Y>
inline auto concatenate(const std::vector<X> &xs,
                        const std::vector<Y> &ys) {
  std::vector<typename internal::concatenation_type<X, Y>::type> features(xs.begin(), xs.end());
  for (const auto &y : ys) {
    features.emplace_back(y);
  }
  return features;
}

//template <typename X, typename... Ts, std::enable_if_t<!is_variant<X>::value &&
//                                                       is_in_variant<X, variant<Ts...>>::value>>
//inline std::vector<variant<Ts...>> concatenate(const std::vector<X> &xs,
//                                              const std::vector<variant<Ts...> &ys) {
//  std::vector<variant<Ts...>> features(xs.begin(), xs.end());
//  for (const auto &y : ys) {
//    features.emplace_back(y);
//  }
//  return features;
//}
//
//template <typename X, typename... Ts, std::enable_if_t<is_in_variant<X, variant<Ts...>>::value>>
//inline std::vector<variant<Ts...>> concatenate(const std::vector<X> &xs,
//                                              const std::vector<variant<Ts...> &ys) {
//  std::vector<variant<Ts...>> features(xs.begin(), xs.end());
//  for (const auto &y : ys) {
//    features.emplace_back(y);
//  }
//  return features;
//}
//
//template <typename X>
//inline std::vector<X> concatenate(const std::vector<X> &xs,
//                                  const std::vector<X> &ys) {
//  std::vector<X> features(xs.begin(), xs.end());
//  for (const auto &y : ys) {
//    features.emplace_back(y);
//  }
//  return features;
//}

}

#endif /* ALBATROSS_CORE_CONCATENATE_HPP_ */
