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

#ifndef ALBATROSS_COVARIANCE_FUNCTIONS_TRAITS_H
#define ALBATROSS_COVARIANCE_FUNCTIONS_TRAITS_H

namespace albatross {

MAKE_HAS_ANY_TRAIT(_call_impl);

// A helper rename to avoid duplicate underscores.
template <typename U> class has_any_call_impl : public has_any__call_impl<U> {};

HAS_METHOD_WITH_RETURN_TYPE(_call_impl);

template <typename U, typename... Args>
class has_valid_call_impl : public has__call_impl_with_return_type<
                                U, double, typename const_ref<Args>::type...> {
};

HAS_METHOD(_call_impl);

template <typename U, typename... Args>
class has_possible_call_impl : public has__call_impl<U, Args &...> {};

HAS_METHOD_WITH_RETURN_TYPE(call);

template <typename U, typename Caller, typename... Args>
class has_valid_cov_caller
    : public has_call_with_return_type<Caller, double,
                                       typename const_ref<U>::type,
                                       typename const_ref<Args>::type...> {};

template <typename U, typename Caller, typename A, typename B>
class has_valid_cross_cov_caller {
public:
  static constexpr bool value = has_valid_cov_caller<U, Caller, A, A>::value &&
                                has_valid_cov_caller<U, Caller, A, B>::value &&
                                has_valid_cov_caller<U, Caller, B, B>::value;
};

template <typename U, typename Caller, typename A, typename B>
struct has_valid_cross_cov_caller<U, Caller, A, variant<B>>
    : public has_valid_cross_cov_caller<U, Caller, A, B> {
  ;
};

template <typename U, typename Caller, typename A, typename B, typename... Ts>
struct has_valid_cross_cov_caller<U, Caller, A, variant<B, Ts...>> {
  static constexpr bool value =
      has_valid_cross_cov_caller<U, Caller, A, B>::value ||
      has_valid_cross_cov_caller<U, Caller, A, variant<Ts...>>::value;
};

/*
 * This determines whether or not a class has a method defined for,
 *   `operator() (const X &x, const Y &y, const Z &z, ...)`
 * The result of the inspection gets stored in the member `value`.
 */
template <typename T, typename... Args> class has_call_operator {

  template <typename C, typename = decltype(std::declval<C>()(
                            std::declval<typename const_ref<Args>::type>()...))>
  static std::true_type test(C *);
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename T, typename... Args> class has_invalid_call_impl {

public:
  static constexpr bool value = (has_possible_call_impl<T, Args...>::value &&
                                 !has_valid_call_impl<T, Args...>::value);
};

HAS_METHOD(solve);

template <typename T> struct is_variant : public std::false_type {};

template <typename... Ts>
struct is_variant<variant<Ts...>> : public std::true_type {};

/*
 * Is in variant
 */

template <typename T, typename P>
struct is_in_variant  : public std::false_type {};

template <typename T, typename A>
struct is_in_variant<T, variant<A>> : public std::is_same<T, A> {};

template <typename T, typename A, typename... Ts>
struct is_in_variant<T, variant<A, Ts...>> {
  static constexpr bool value = (std::is_same<T, A>::value || is_in_variant<T, variant<Ts...>>::value);
};

/*
 * Has valid caller for all variants
 */

template <typename U, typename Caller, typename A>
struct has_valid_caller_for_all_variants : public std::false_type {};

template <typename U, typename Caller, typename A>
struct has_valid_caller_for_all_variants<U, Caller, variant<A>>
    : public has_valid_cov_caller<U, Caller, A, A> {};

template <typename U, typename Caller, typename A, typename... Ts>
struct has_valid_caller_for_all_variants<U, Caller, variant<A, Ts...>> {
  static constexpr bool value =
      has_valid_cov_caller<U, Caller, A, A>::value &&
      has_valid_caller_for_all_variants<U, Caller, variant<Ts...>>::value;
};

template <typename U, typename Caller, typename... Ts>
struct has_valid_cov_caller<U, Caller, variant<Ts...>, variant<Ts...>>
    : public has_valid_caller_for_all_variants<U, Caller, variant<Ts...>> {};

/*
 * Checks if a type has a valid cov call for any of the types in a variant.
 */
template <typename U, typename Caller, typename A, typename B, typename = void>
struct has_valid_variant_cov_caller : public std::false_type {};

/*
 * Collapse from the right
 */
template <typename U, typename Caller, typename A, typename B>
struct has_valid_variant_cov_caller<U, Caller, A, variant<B>, std::enable_if_t<!is_variant<A>::value>> {
  static constexpr bool value = has_valid_cov_caller<U, Caller, A, B>::value;
};

template <typename U, typename Caller, typename A, typename B, typename... Ts>
struct has_valid_variant_cov_caller<U, Caller, A, variant<B, Ts...>, std::enable_if_t<!is_variant<A>::value>> {
  static constexpr bool value =
      has_valid_cov_caller<U, Caller, A, B>::value ||
      has_valid_variant_cov_caller<U, Caller, A, variant<Ts...>>::value;
};

// Collapse from the left
template <typename U, typename Caller, typename A, typename B>
struct has_valid_variant_cov_caller<U, Caller, variant<A>, B, std::enable_if_t<!is_variant<B>::value>> {
  static constexpr bool value = has_valid_cov_caller<U, Caller, A, B>::value;
};

template <typename U, typename Caller, typename A, typename B, typename... Ts>
struct has_valid_variant_cov_caller<U, Caller, variant<A, Ts...>, B, std::enable_if_t<!is_variant<B>::value>> {
  static constexpr bool value =
      has_valid_cov_caller<U, Caller, A, B>::value ||
      has_valid_variant_cov_caller<U, Caller, variant<Ts...>, B>::value;
};

// For some mysterious reason this only works if the types in the variant are
// split into two parts (Ts...).
template <typename U, typename Caller, typename... Ts>
struct has_valid_variant_cov_caller<U, Caller, variant<Ts...>,
                                    variant<Ts...>> {
  static constexpr bool value =
      has_valid_cov_caller<U, Caller, variant<Ts...>,
                           variant<Ts...>>::value;
};

} // namespace albatross

#endif
