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

#ifndef INCLUDE_ALBATROSS_SRC_DETAILS_TRAITS_HPP_
#define INCLUDE_ALBATROSS_SRC_DETAILS_TRAITS_HPP_

namespace albatross {

/*
 * We frequently inspect for definitions of functions which
 * must be defined for const references to objects
 * (so that repeated evaluations return the same thing
 *  and so the computations are not repeatedly copying.)
 * This type conversion utility will turn a type `T` into `const T&`
 */
template <class T> struct const_ref {
  typedef
      typename std::add_lvalue_reference<typename std::add_const<T>::type>::type
          type;
};

/*
 * This little trick was borrowed from cereal, you can think of it as
 * a function that will always return false ... but that doesn't
 * get resolved until template instantiation, which when combined
 * with a static assert let's you include a static assert that
 * only triggers with a particular template parameter is used.
 */
template <class T> struct delay_static_assert : std::false_type {};

/*
 * Checks if a class type is complete by using sizeof.
 *
 * https://stackoverflow.com/questions/25796126/static-assert-that-template-typename-t-is-not-complete
 */
template <typename X> class is_complete {
  template <typename T, typename = decltype(!sizeof(T))>
  static std::true_type test(int);
  template <typename T> static std::false_type test(...);

public:
  static constexpr bool value =
      decltype(test<typename std::decay<X>::type>(0))::value;
};

/*
 * is_vector
 */

template <typename T> struct is_vector : public std::false_type {};

template <typename T>
struct is_vector<std::vector<T>> : public std::true_type {};

/*
 * is_variant
 */

template <typename T> struct is_variant : public std::false_type {};

template <typename... Ts>
struct is_variant<variant<Ts...>> : public std::true_type {};

/*
 * is_in_variant
 */

template <typename T, typename A>
struct is_in_variant : public std::false_type {};

template <typename T, typename A>
struct is_in_variant<T, variant<A>> : public std::is_same<T, A> {};

template <typename T, typename A, typename... Ts>
struct is_in_variant<T, variant<A, Ts...>> {
  static constexpr bool value =
      (std::is_same<T, A>::value || is_in_variant<T, variant<Ts...>>::value);
};

/*
 * variant_size
 */
template <typename T> struct variant_size {};

template <typename... Ts>
struct variant_size<variant<Ts...>>
    : public std::tuple_size<std::tuple<Ts...>> {};

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_SRC_DETAILS_TRAITS_HPP_ */
