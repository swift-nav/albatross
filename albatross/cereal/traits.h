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

#include "cereal/details/traits.hpp"

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

}

#endif
