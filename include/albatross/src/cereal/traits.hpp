/*
 * Copyright (C) 2018 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_CEREAL_TRAITS_H
#define ALBATROSS_CEREAL_TRAITS_H

namespace albatross {

/*
 * The following helper functions let you inspect a type and cereal Archive
 * and determine if the type has a valid serialization method for that Archive
 * type.
 */
template <typename X, typename Archive> class valid_output_serializer {
  template <typename T>
  static typename std::enable_if<
      1 == cereal::traits::detail::count_output_serializers<T, Archive>::value,
      std::true_type>::type
  test(int);
  template <typename T> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<X>(0))::value;
};

template <typename X, typename Archive> class valid_input_serializer {
  template <typename T>
  static typename std::enable_if<
      1 == cereal::traits::detail::count_input_serializers<T, Archive>::value,
      std::true_type>::type
  test(int);
  template <typename T> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<X>(0))::value;
};

template <typename X, typename Archive> class valid_in_out_serializer {
  template <typename T>
  static typename std::enable_if<valid_input_serializer<T, Archive>::value &&
                                     valid_output_serializer<T, Archive>::value,
                                 std::true_type>::type
  test(int);
  template <typename T> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<X>(0))::value;
};

} // namespace albatross

namespace cereal {
namespace detail {

// The Following makes it so you can set a static member in a class
// in order to define the cereal serialization version that will be used.
//
//   struct Foo {
//     static const std::uint32_t serialization_version = 2;
//   }

template <typename X> class has_serialization_version {
  template <typename T>
  static std::enable_if_t<
      std::is_same<std::uint32_t, typename std::decay<decltype(
                                      T::serialization_version)>::type>::value,
      std::true_type>
  test(int);
  template <typename T> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<X>(0))::value;
};

// This default definition mimics the one in helpers.hpp
template <typename T, bool> struct VersionedAlbatrossType {
  static const std::uint32_t version = 0;
};

// Here we define the version for serialization following the
// example given here: https://github.com/USCiLab/cereal/issues/319
template <typename T> struct VersionedAlbatrossType<T, true> {
  static const std::uint32_t version = T::serialization_version;

  static std::uint32_t registerVersion() {
    ::cereal::detail::StaticObject<Versions>::getInstance().mapping.emplace(
        std::type_index(typeid(T)).hash_code(), version);
    return version;
  }
};

// Propagate the `serialization_version` member into the cereal version.
template <typename T>
struct Version<T>
    : public VersionedAlbatrossType<T, has_serialization_version<T>::value> {};

} // namespace detail

} // namespace cereal

#endif
