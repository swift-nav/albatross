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

#ifndef VARIANT_SERIALIZATION_VERSION
#define VARIANT_SERIALIZATION_VERSION 1
#endif

namespace cereal {

namespace mapbox_variant_detail {

template <int N, class Variant, class... Args, class Archive,
          typename std::enable_if_t<
              N == albatross::variant_size<Variant>::value, int> = 0>
void load_variant(Archive &, int, Variant &) {
  ALBATROSS_ASSERT(false); // load_variant received an out of bounds index.
}

template <int N, class Variant, class H, class... T, class Archive,
          typename std::enable_if_t<
              N<albatross::variant_size<Variant>::value, int> = 0> void
              load_variant(Archive &archive, int target, Variant &variant) {
  // It can get extremely confusing figuring out which type is causing a variant
  // to not be serializable, this static assert should help reveal the culprit.
  static_assert(albatross::valid_input_serializer<H, Archive>::value,
                "Type is not serializable");
  if (N == target) {
    H value;
    archive(cereal::make_nvp("data", value));
    variant = std::move(value);
  } else {
    load_variant<N + 1, Variant, T...>(archive, target, variant);
  }
}

} // namespace mapbox_variant_detail

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
                 const std::uint32_t version) {
  archive(cereal::make_nvp("which", f.which()));
  if (version > 0) {
    f.match([&](const auto &x) {
      std::string which_typeid = typeid(x).name();
      archive(cereal::make_nvp("which_typeid", which_typeid));
    });
  }
  f.match([&archive](const auto &x) { archive(cereal::make_nvp("data", x)); });
}

template <class Archive, typename... VariantTypes>
inline void load(Archive &archive, variant<VariantTypes...> &v,
                 const std::uint32_t version) {
  int which;
  archive(cereal::make_nvp("which", which));
  if (version > 0) {
    std::string which_typeid;
    archive(cereal::make_nvp("which_typeid", which_typeid));
    ALBATROSS_ASSERT(which < static_cast<int>(sizeof...(VariantTypes)));
  }
  mapbox_variant_detail::load_variant<0, variant<VariantTypes...>,
                                      VariantTypes...>(archive, which, v);
}

// Here we define the version for variant serialization following the
// example given here: https://github.com/USCiLab/cereal/issues/319
namespace detail {
template <typename... VariantTypes> struct Version<variant<VariantTypes...>> {
  static const std::uint32_t version;
  static std::uint32_t registerVersion() {
    ::cereal::detail::StaticObject<Versions>::getInstance().mapping.emplace(
        std::type_index(typeid(variant<VariantTypes...>)).hash_code(),
        VARIANT_SERIALIZATION_VERSION);
    return VARIANT_SERIALIZATION_VERSION;
  }
  static void unused() { (void)version; }
};
template <typename... VariantTypes>
const std::uint32_t Version<variant<VariantTypes...>>::version =
    Version<variant<VariantTypes...>>::registerVersion();
} // namespace detail

} // namespace cereal
#endif /* INCLUDE_ALBATROSS_SRC_CEREAL_VARIANT_HPP_ */
