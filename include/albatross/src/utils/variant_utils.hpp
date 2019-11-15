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
 * In this case X is a variant which is compatible so we need to
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

} // namespace albatross

#endif /* ALBATROSS_UTILS_VARIANT_UTILS_HPP_ */
