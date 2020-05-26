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

#ifndef ALBATROSS_UTILS_VARIANT_UTILS_HPP_
#define ALBATROSS_UTILS_VARIANT_UTILS_HPP_

namespace albatross {

namespace details {
template <typename X> struct ToVariantIdentity {};
} // namespace details

/*
 * In this case X is a sub variant which is compatible so we need to
 * convert it.
 */
template <typename... Ts, typename X>
inline std::enable_if_t<is_sub_variant<X, variant<Ts...>>::value, void>
set_variant(const X &x, variant<Ts...> *to_set) {
  x.match([&to_set](const auto &v) { *to_set = v; });
}

/*
 * In this case X is not a variant so we can simply assign it.
 */
template <typename... Ts, typename X>
inline std::enable_if_t<
    !is_variant<X>::value && is_in_variant<X, variant<Ts...>>::value, void>
set_variant(const X &x, variant<Ts...> *to_set) {
  *to_set = x;
}

template <typename... Ts, typename X>
inline std::enable_if_t<
    !is_variant<X>::value && !is_in_variant<X, variant<Ts...>>::value, void>
set_variant(const X &x, variant<Ts...> *to_set) {
  static_assert(delay_static_assert<X>::value,
                "Incompatible type. X does not belong to variant.");
}

/*
 * Convert a vector of one type
 */
template <typename... Ts, typename X>
inline void set_variants(const std::vector<X> &xs,
                         std::vector<variant<Ts...>> *to_set) {
  to_set->resize(xs.size());
  for (std::size_t i = 0; i < xs.size(); ++i) {
    set_variant(xs[i], &to_set->operator[](i));
  }
}

template <typename OutputType, typename X,
          std::enable_if_t<is_variant<OutputType>::value, int> = 0>
inline std::vector<OutputType>
to_variant_vector(const std::vector<X> &xs,
                  details::ToVariantIdentity<OutputType> && = {}) {
  std::vector<OutputType> output;
  set_variants(xs, &output);
  return output;
}

template <
    typename OutputType, typename... VariantTypes,
    std::enable_if_t<is_in_variant<OutputType, variant<VariantTypes...>>::value,
                     int> = 0>
inline std::vector<OutputType>
extract_from_variants(const std::vector<variant<VariantTypes...>> &xs,
                      details::ToVariantIdentity<OutputType> && = {}) {
  std::vector<OutputType> output;
  for (const auto &x : xs) {
    x.match([&](const OutputType &f) { output.emplace_back(f); },
            [](const auto &) {});
  }

  return output;
}

template <
    typename OutputType, typename... VariantTypes,
    std::enable_if_t<is_in_variant<OutputType, variant<VariantTypes...>>::value,
                     int> = 0>
inline RegressionDataset<OutputType> extract_from_variants(
    const RegressionDataset<variant<VariantTypes...>> &dataset,
    details::ToVariantIdentity<OutputType> && = {}) {
  std::vector<std::size_t> indices;
  std::vector<OutputType> features;
  for (std::size_t i = 0; i < dataset.size(); ++i) {
    dataset.features[i].match(
        [&](const OutputType &f) {
          indices.emplace_back(i);
          features.emplace_back(f);
        },
        [](const auto &) {});
  }
  return RegressionDataset<OutputType>(features,
                                       subset(dataset.targets, indices));
}

} // namespace albatross

#endif /* ALBATROSS_UTILS_VARIANT_UTILS_HPP_ */
