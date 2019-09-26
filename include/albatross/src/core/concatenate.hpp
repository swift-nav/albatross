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

template <typename X, typename Y, typename = void> struct concatenation_type {
  using type = variant<X, Y>;
};

template <typename X> struct concatenation_type<X, X> { using type = X; };

template <typename X, typename... Ts>
struct concatenation_type<
    X, variant<Ts...>,
    std::enable_if_t<!is_variant<X>::value &&
                     is_in_variant<X, variant<Ts...>>::value>> {
  using type = variant<Ts...>;
};

template <typename X, typename... Ts>
struct concatenation_type<
    X, variant<Ts...>,
    std::enable_if_t<!is_variant<X>::value &&
                     !is_in_variant<X, variant<Ts...>>::value>> {
  using type = variant<X, Ts...>;
};

template <typename X, typename... Ts>
struct concatenation_type<
    variant<Ts...>, X,
    std::enable_if_t<!is_variant<X>::value &&
                     is_in_variant<X, variant<Ts...>>::value>> {
  using type = variant<Ts...>;
};

template <typename X, typename... Ts>
struct concatenation_type<
    variant<Ts...>, X,
    std::enable_if_t<!is_variant<X>::value &&
                     !is_in_variant<X, variant<Ts...>>::value>> {
  using type = variant<Ts..., X>;
};

} // namespace internal

/*
 * concatenate with two identical types
 */
template <typename X>
inline auto concatenate(const std::vector<X> &xs, const std::vector<X> &ys) {
  std::vector<X> features(xs.begin(), xs.end());
  for (const auto &y : ys) {
    features.emplace_back(y);
  }
  return features;
}

template <typename X>
inline auto concatenate(const std::vector<std::vector<X>> &all_xs) {
  std::vector<X> features;
  for (const auto &one : all_xs) {
    features.insert(features.end(), one.begin(), one.end());
  }
  return features;
}

/*
 * concatenate with two different non-variant types
 */
template <typename X, typename Y,
          typename std::enable_if<
              !is_variant<X>::value && !is_variant<Y>::value, int>::type = 0>
inline auto concatenate(const std::vector<X> &xs, const std::vector<Y> &ys) {
  std::vector<typename internal::concatenation_type<X, Y>::type> features(
      xs.begin(), xs.end());
  for (const auto &y : ys) {
    features.emplace_back(y);
  }
  return features;
}

/*
 * concatenate with the right hand side a variant.
 */
template <typename X, typename Y,
          typename std::enable_if<!is_variant<X>::value && is_variant<Y>::value,
                                  int>::type = 0>
inline auto concatenate(const std::vector<X> &xs, const std::vector<Y> &ys) {
  using OutputType = typename internal::concatenation_type<X, Y>::type;
  std::vector<OutputType> features(xs.begin(), xs.end());
  for (const auto &y : ys) {
    features.emplace_back(
        y.match([](const auto &yy) { return OutputType(yy); }));
  }
  return features;
}

/*
 * concatenate with the left hand side a variant.
 */
template <typename X, typename Y,
          typename std::enable_if<is_variant<X>::value && !is_variant<Y>::value,
                                  int>::type = 0>
inline auto concatenate(const std::vector<X> &xs, const std::vector<Y> &ys) {
  using OutputType = typename internal::concatenation_type<X, Y>::type;
  std::vector<OutputType> features;
  for (const auto &x : xs) {
    features.emplace_back(
        x.match([](const auto &xx) { return OutputType(xx); }));
  }
  for (const auto &y : ys) {
    features.emplace_back(y);
  }
  return features;
}

} // namespace albatross

#endif /* ALBATROSS_CORE_CONCATENATE_HPP_ */
