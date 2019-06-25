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

#ifndef INCLUDE_ALBATROSS_SRC_CEREAL_VARIANT_HPP_
#define INCLUDE_ALBATROSS_SRC_CEREAL_VARIANT_HPP_

/*
 * A lot of this is borrowed from cereal's std::variant.
 */

namespace cereal {

namespace mapbox_variant_detail
{

  template <typename Variant>
  struct variant_size {};

  template <typename... VariantTypes>
  struct variant_size<variant<VariantTypes...>> : public std::tuple_size<std::tuple<VariantTypes...>> {};

  template<int N, class Variant, class ... Args, class Archive,
           typename std::enable_if_t<N == variant_size<Variant>::value, int> = 0>
  void load_variant(Archive &, int, Variant &) {
    assert(false); // load_variant received an out of bounds index.
  }

  template<int N, class Variant, class H, class ... T, class Archive,
           typename std::enable_if_t<N < variant_size<Variant>::value, int> = 0>
  void load_variant(Archive & archive, int target, Variant & variant) {
    // It can get extremely confusing figuring out which type is causing a variant
    // to not be serializable, this static assert should help reveal the culprit.
    static_assert(albatross::valid_input_serializer<H, Archive>::value,
                  "Type is not serializable");
    if(N == target) {
      H value;
      archive(cereal::make_nvp("data", value) );
      variant = std::move(value);
    } else {
      load_variant<N+1, Variant, T...>(archive, target, variant);
    }
  }

} // namespace variant_detail


// This deleted function is here to make sure that rvalues of variants
// can't be serialized, this prevents the compiler from thinking it
// can convert a sub-type into a variant.  Ie, it prevents usage such
// as:
//     X x;
//     save(archive, variant<X, Y>(x));
// without this you'll see some pretty confusing cereal errors.
template <class Archive, typename... VariantTypes>
inline void save(Archive &archive, const variant<VariantTypes...> &&f,
                                      const std::uint32_t) = delete;

template <class Archive, typename... VariantTypes>
inline void save(Archive &archive, const variant<VariantTypes...> &f,
                                      const std::uint32_t) {

  archive(cereal::make_nvp("which", f.which()));
  f.match(
      [&archive](const auto &x) {
        archive(cereal::make_nvp("data", x));
      }
  );
}

template <class Archive, typename... VariantTypes>
inline void load(Archive &archive, variant<VariantTypes...> &v,
                                   const std::uint32_t) {
  decltype(v.which()) which;
  archive(cereal::make_nvp("which", which));
  assert(which < static_cast<decltype(which)>(mapbox_variant_detail::variant_size<variant<VariantTypes...>>::value));
  mapbox_variant_detail::load_variant<0, variant<VariantTypes...>, VariantTypes...>(archive, which, v);

}

}
#endif /* INCLUDE_ALBATROSS_SRC_CEREAL_VARIANT_HPP_ */
