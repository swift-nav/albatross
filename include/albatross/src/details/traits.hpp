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
 * map helpers
 */

template <typename Map, typename K, typename = void>
struct has_find_key : std::false_type {};

template <typename Map, typename K>
struct has_find_key<Map, K,
                    std::void_t<decltype(std::declval<const Map &>().find(
                        std::declval<const K &>()))>> : std::true_type {};

template <typename Map, typename K>
inline constexpr bool has_find_key_v = has_find_key<Map, K>::value;

template <typename T, typename = void>
struct has_key_compare : std::false_type {};

template <typename T>
struct has_key_compare<T, std::void_t<typename T::key_compare>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_key_compare_v = has_key_compare<T>::value;

template <typename T, typename U, typename = void>
struct has_same_key_compare : std::false_type {};

template <typename T, typename U>
struct has_same_key_compare<
    T, U, std::void_t<typename T ::key_compare, typename U::key_compare>>
    : std::is_same<typename T::key_compare, typename U::key_compare> {};

template <typename T, typename U>
inline constexpr bool has_same_key_compare_v =
    has_same_key_compare<T, U>::value;

template <typename Project, typename K, typename K2, typename = void>
struct can_project_key : std::false_type {};

template <typename Project, typename K, typename K2>
struct can_project_key<Project, K, K2,
                       std::void_t<std::is_invocable_r<K, Project, K2>>>
    : std::true_type {};

template <typename Project, typename K, typename K2>
inline constexpr bool can_project_key_v =
    can_project_key<Project, K, K2>::value;

template <typename F, typename K, typename V,
          bool WithKey =
              std::is_invocable_r_v<V, F &, const K &, const V &, const V &>,
          bool WithoutKey = std::is_invocable_r_v<V, F &, const V &, const V &>>
struct map_union_traits;

template <typename F, typename K, typename V, bool WithoutKey>
struct map_union_traits<F, K, V, true, WithoutKey> {
  static constexpr const bool with_key = true;
  static constexpr const bool without_key = WithoutKey;
};

template <typename F, typename K, typename V>
struct map_union_traits<F, K, V, false, true> {
  static constexpr const bool with_key = false;
  static constexpr const bool without_key = true;
};

template <typename OnMatch, typename K, typename V>
inline constexpr bool can_call_map_union_v =
    map_union_traits<std::decay_t<OnMatch>, K, V>::with_key ||
    map_union_traits<std::decay_t<OnMatch>, K, V>::without_key;

template <
    typename OnMatch, typename K, typename V,
    typename = std::enable_if_t<can_call_map_union_v<OnMatch, K, V>, void>>
auto call_map_union(OnMatch &&f, const K &k, const V &va, const V &vb) {
  using traits = map_union_traits<std::decay_t<OnMatch>, K, V>;
  if constexpr (traits::with_key) {
    return std::invoke(std::forward<OnMatch>(f), k, va, vb);
  } else {
    static_assert(traits::without_key,
                  "Map union helper must be callable either as f(K, V, "
                  "V) or f(V, V)!");
    return std::invoke(std::forward<OnMatch>(f), va, vb);
  }
}

template <typename F, typename K, typename V, typename V2,
          bool WithKey =
              std::is_invocable_v<F &, const K &, const V &, const V2 &>,
          bool WithoutKey = std::is_invocable_v<F &, const V &, const V2 &>>
struct map_intersection_traits;

template <typename F, typename K, typename V, typename V2, bool WithoutKey>
struct map_intersection_traits<F, K, V, V2, true, WithoutKey> {
  static constexpr const bool with_key = true;
  static constexpr const bool without_key = WithoutKey;
  using result_t =
      std::decay_t<std::invoke_result_t<F &, const K &, const V &, const V2 &>>;
};

template <typename F, typename K, typename V, typename V2>
struct map_intersection_traits<F, K, V, V2, false, true> {
  static constexpr const bool with_key = false;
  static constexpr const bool without_key = true;
  using result_t =
      std::decay_t<std::invoke_result_t<F &, const V &, const V2 &>>;
};

template <typename F, typename K, typename V, typename V2>
struct map_intersection_traits<F, K, V, V2, false, false> {
  static constexpr const bool with_key = false;
  static constexpr const bool without_key = false;
};

template <typename OnMatch, typename K, typename V, typename V2>
using MapIntersectionResultType =
    typename map_intersection_traits<std::decay_t<OnMatch>, K, V, V2>::result_t;

template <typename OnMatch, typename K, typename V, typename V2>
inline constexpr bool can_call_map_intersection_v =
    map_intersection_traits<std::decay_t<OnMatch>, K, V, V2>::with_key ||
    map_intersection_traits<std::decay_t<OnMatch>, K, V, V2>::without_key;

template <typename OnMatch, typename K, typename V, typename V2,
          typename = std::enable_if_t<
              can_call_map_intersection_v<OnMatch, K, V, V2>, void>>
auto call_map_intersection(OnMatch &&f, const K &k, const V &v, const V2 &v2) {
  using traits = map_intersection_traits<std::decay_t<OnMatch>, K, V, V2>;
  if constexpr (traits::with_key) {
    return std::invoke(std::forward<OnMatch>(f), k, v, v2);
  } else {
    static_assert(traits::without_key,
                  "Map intersection helper must be callable either as f(K, V, "
                  "V2) or f(V, V2)!");
    return std::invoke(std::forward<OnMatch>(f), v, v2);
  }
}

template <template <typename...> typename Map, typename K, typename V,
          typename V2, typename OnMatch, typename... More>
using IntersectedMapType = std::enable_if_t<
    can_call_map_intersection_v<OnMatch, K, V, V2>,
    Map<K, MapIntersectionResultType<OnMatch, K, V, V2>, More...>>;

/*
 * is_templated_type
 */
template <template <typename...> class Wrapper, typename T>
struct is_templated_type : public std::false_type {};

template <template <typename...> class Wrapper, typename... Ts>
struct is_templated_type<Wrapper, Wrapper<Ts...>> : public std::true_type {};

/*
 * is_variant
 */

template <typename T>
struct is_variant : public is_templated_type<variant, T> {};

/*
 * is_measurement
 */

template <typename T>
struct is_measurement : public is_templated_type<Measurement, T> {};

/*
 * is_streamable
 */

template <typename T> class is_streamable {
  template <typename C>
  static auto test(int)
      -> decltype(std::declval<std::ostream &>() << std::declval<C>(),
                  std::true_type());

  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename T> constexpr bool is_streamable<T>::value;

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
 * Checks if Expression<A>::value is true for any of the types in a variant.
 */
template <template <typename...> class Expression, typename A, typename = void>
struct variant_any : public std::false_type {};

template <template <typename...> class Expression, typename A>
struct variant_any<Expression, variant<A>,
                   std::enable_if_t<!is_variant<A>::value>> {
  static constexpr bool value = Expression<A>::value;
};

template <template <typename...> class Expression, typename A, typename... Ts>
struct variant_any<Expression, variant<A, Ts...>,
                   std::enable_if_t<!is_variant<A>::value>> {
  static constexpr bool value =
      Expression<A>::value || variant_any<Expression, variant<Ts...>>::value;
};

/*
 * Checks if Expression<A>::value is true for all of the types in a variant.
 */
template <template <typename...> class Expression, typename A, typename = void>
struct variant_all : public std::true_type {};

template <template <typename...> class Expression, typename A>
struct variant_all<Expression, variant<A>,
                   std::enable_if_t<!is_variant<A>::value>> {
  static constexpr bool value = Expression<A>::value;
};

template <template <typename...> class Expression, typename A, typename... Ts>
struct variant_all<Expression, variant<A, Ts...>,
                   std::enable_if_t<!is_variant<A>::value>> {
  static constexpr bool value =
      Expression<A>::value && variant_all<Expression, variant<Ts...>>::value;
};

/*
 * Checks if one variant contains all the types of another variant.
 */
template <typename X, typename Y>
struct is_sub_variant : public std::false_type {};

template <typename Y, typename... Ts> struct is_sub_variant<variant<Ts...>, Y> {
private:
  template <typename T> struct is_in_y : public is_in_variant<T, Y> {};

public:
  static constexpr bool value = variant_all<is_in_y, variant<Ts...>>::value;
};

/*
 * variant_size
 */
template <typename T> struct variant_size {};

template <typename... Ts>
struct variant_size<variant<Ts...>>
    : public std::tuple_size<std::tuple<Ts...>> {};

/*
 * Eigen helpers
 */
template <typename T> class is_eigen_plain_object {
  template <typename C>
  static typename std::enable_if<
      std::is_base_of<Eigen::PlainObjectBase<C>, C>::value,
      std::true_type>::type
  test(int);

  template <typename C> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename T> class is_eigen_xpr {
  // Here we check if T is a generic eigen expression, but we then
  // want to rule out cases where T is a plain_object such as
  // an Eigen::MatrixXd.
  template <typename C,
            typename IsGenericXpr =
                typename Eigen::internal::generic_xpr_base<C>::type,
            std::enable_if_t<!is_eigen_plain_object<C>::value, int> = 0>
  static std::true_type test(int);

  template <typename C> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

/*
 * invocable and invoke result
 *
 * both copied from the possible implementation provided here:
 *   https://en.cppreference.com/w/cpp/types/result_of
 */

namespace detail {
template <class T> struct is_reference_wrapper : std::false_type {};
template <class U>
struct is_reference_wrapper<std::reference_wrapper<U>> : std::true_type {};

template <class T> struct invoke_impl {
  template <class F, class... Args>
  static auto call(F &&f, Args &&...args)
      -> decltype(std::forward<F>(f)(std::forward<Args>(args)...));
};

template <class B, class MT> struct invoke_impl<MT B::*> {
  template <
      class T, class Td = typename std::decay<T>::type,
      class = typename std::enable_if<std::is_base_of<B, Td>::value>::type>
  static auto get(T &&t) -> T &&;

  template <
      class T, class Td = typename std::decay<T>::type,
      class = typename std::enable_if<is_reference_wrapper<Td>::value>::type>
  static auto get(T &&t) -> decltype(t.get());

  template <
      class T, class Td = typename std::decay<T>::type,
      class = typename std::enable_if<!std::is_base_of<B, Td>::value>::type,
      class = typename std::enable_if<!is_reference_wrapper<Td>::value>::type>
  static auto get(T &&t) -> decltype(*std::forward<T>(t));

  template <class T, class... Args, class MT1,
            class = typename std::enable_if<std::is_function<MT1>::value>::type>
  static auto call(MT1 B::*pmf, T &&t, Args &&...args)
      -> decltype((invoke_impl::get(std::forward<T>(t)).*
                   pmf)(std::forward<Args>(args)...));

  template <class T>
  static auto call(MT B::*pmd, T &&t)
      -> decltype(invoke_impl::get(std::forward<T>(t)).*pmd);
};

template <class F, class... Args, class Fd = typename std::decay<F>::type>
auto INVOKE(F &&f, Args &&...args)
    -> decltype(invoke_impl<Fd>::call(std::forward<F>(f),
                                      std::forward<Args>(args)...));

template <typename AlwaysVoid, typename, typename...> struct invoke_result {};
template <typename F, typename... Args>
struct invoke_result<decltype(void(detail::INVOKE(std::declval<F>(),
                                                  std::declval<Args>()...))),
                     F, Args...> {
  using type =
      decltype(detail::INVOKE(std::declval<F>(), std::declval<Args>()...));
};

} // namespace detail

template <class F, class... Args>
struct invoke_result : detail::invoke_result<void, F, Args...> {};

template <class F, class... Args> class is_invocable {
  template <typename T, typename = typename invoke_result<T, Args...>::type>
  static std::true_type test(int);
  template <typename T> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<F>(0))::value;
};

template <class F, class ReturnType, class... Args>
struct is_invocable_with_result {
  static constexpr bool value = std::is_same<
      ReturnType,
      typename detail::invoke_result<void, F, Args...>::type>::value;
};

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_SRC_DETAILS_TRAITS_HPP_ */
