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

DEFINE_CLASS_METHOD_TRAITS(_call_impl);

template <typename U, typename... Args>
class has_valid_call_impl
    : public has__call_impl_with_return_type<
          const U, double, typename const_ref<Args>::type...> {};

template <typename U, typename... Args>
class has_possible_call_impl : public has__call_impl<U, Args &...> {};

// Here we check if a class U has a call_impl which can be reused
// inside things like sums and products where we may need to
// leverage equivalencies like cov(X, Y) == cov(Y, X) but don't
// want to worry about all the caller logic.
template <typename U, typename X, typename Y> class has_usable_call_impl {
public:
  static constexpr bool value = has_valid_call_impl<U, X, Y>::value ||
                                has_valid_call_impl<U, Y, X>::value;
};

/*
 * has_valid_cov_caller
 */

DEFINE_CLASS_METHOD_TRAITS(call);

template <typename CovFunc, typename Caller, typename... Args>
class has_valid_cov_caller
    : public has_call_with_return_type<Caller, double,
                                       typename const_ref<CovFunc>::type,
                                       typename const_ref<Args>::type...> {};

/*
 * has_valid_cross_cov_caller
 */
template <typename U, typename Caller, typename A, typename B>
class has_valid_cross_cov_caller {
public:
  static constexpr bool value = has_valid_cov_caller<U, Caller, A, A>::value &&
                                has_valid_cov_caller<U, Caller, A, B>::value &&
                                has_valid_cov_caller<U, Caller, B, B>::value;
};

template <typename MeanFunc, typename Caller, typename FeatureType>
class has_valid_mean_caller
    : public has_call_with_return_type<Caller, double,
                                       typename const_ref<MeanFunc>::type,
                                       typename const_ref<FeatureType>::type> {
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

DEFINE_CLASS_METHOD_TRAITS(solve);

DEFINE_CLASS_METHOD_TRAITS(_ssr_impl);

template <typename T, typename FeatureType> class has_valid_ssr_impl {
  using SsrCall = class_method__ssr_impl_traits<T, std::vector<FeatureType>>;

public:
  static constexpr bool value =
      (SsrCall::is_defined && is_vector<typename SsrCall::return_type>::value);
};

DEFINE_CLASS_METHOD_TRAITS(state_space_representation);

template <typename T, typename FeatureType>
struct has_valid_state_space_representation {

  using SsrCall =
      class_method_state_space_representation_traits<T,
                                                     std::vector<FeatureType>>;

public:
  static constexpr bool value =
      (SsrCall::is_defined && is_vector<typename SsrCall::return_type>::value);
};

/*
 * Has valid caller for all variants
 */

template <typename U, typename Caller, typename A>
struct has_valid_cov_caller_for_all_variants : public std::false_type {};

template <typename U, typename Caller, typename A>
struct has_valid_cov_caller_for_all_variants<U, Caller, variant<A>>
    : public has_valid_cov_caller<U, Caller, A, A> {};

template <typename U, typename Caller, typename A, typename... Ts>
struct has_valid_cov_caller_for_all_variants<U, Caller, variant<A, Ts...>> {
  static constexpr bool value =
      has_valid_cov_caller<U, Caller, A, A>::value &&
      has_valid_cov_caller_for_all_variants<U, Caller, variant<Ts...>>::value;
};

/*
 * A specialization of has_valid_cov_caller for variants in which
 * the call is valid if all the types involved are valid.
 */
template <typename CovFunc, typename Caller, typename... Ts, typename... Ys>
struct has_valid_cov_caller<CovFunc, Caller, variant<Ts...>, variant<Ys...>> {
  static constexpr bool value =
      (has_valid_cov_caller_for_all_variants<CovFunc, Caller,
                                             variant<Ts...>>::value &&
       has_valid_cov_caller_for_all_variants<CovFunc, Caller,
                                             variant<Ys...>>::value);
};

/*
 * A specialization of has_valid_cross_cov_caller in which all variant
 * types must be valid and at least one cross covariance must be valid.
 */
template <typename U, typename Caller, typename A, typename B>
struct has_valid_cross_cov_caller<U, Caller, A, variant<B>>
    : public has_valid_cross_cov_caller<U, Caller, A, B> {
  ;
};

template <typename U, typename Caller, typename A, typename B, typename... Ts>
struct has_valid_cross_cov_caller<U, Caller, A, variant<B, Ts...>> {
  static constexpr bool value =
      has_valid_cov_caller_for_all_variants<U, Caller,
                                            variant<B, Ts...>>::value &&
      (has_valid_cross_cov_caller<U, Caller, A, B>::value ||
       has_valid_cross_cov_caller<U, Caller, A, variant<Ts...>>::value);
};

/*
 * Checks if a type has a valid cov call for any of the types in a variant.
 */
template <typename U, typename Caller, typename A, typename B, typename = void>
struct has_valid_variant_cov_caller : public std::false_type {};

/*
 * Collapse from the right
 */
template <typename U, typename Caller, typename A, typename B>
struct has_valid_variant_cov_caller<U, Caller, A, variant<B>,
                                    std::enable_if_t<!is_variant<A>::value>> {
  static constexpr bool value = has_valid_cov_caller<U, Caller, A, B>::value;
};

template <typename U, typename Caller, typename A, typename B, typename... Ts>
struct has_valid_variant_cov_caller<U, Caller, A, variant<B, Ts...>,
                                    std::enable_if_t<!is_variant<A>::value>> {
  static constexpr bool value =
      has_valid_cov_caller<U, Caller, A, B>::value ||
      has_valid_variant_cov_caller<U, Caller, A, variant<Ts...>>::value;
};

// Collapse from the left
template <typename U, typename Caller, typename A, typename B>
struct has_valid_variant_cov_caller<U, Caller, variant<A>, B,
                                    std::enable_if_t<!is_variant<B>::value>> {
  static constexpr bool value = has_valid_cov_caller<U, Caller, A, B>::value;
};

template <typename U, typename Caller, typename A, typename B, typename... Ts>
struct has_valid_variant_cov_caller<U, Caller, variant<A, Ts...>, B,
                                    std::enable_if_t<!is_variant<B>::value>> {
  static constexpr bool value =
      has_valid_cov_caller<U, Caller, A, B>::value ||
      has_valid_variant_cov_caller<U, Caller, variant<Ts...>, B>::value;
};

template <typename U, typename Caller, typename... Ts>
struct has_valid_variant_cov_caller<U, Caller, variant<Ts...>, variant<Ts...>> {
  static constexpr bool value =
      has_valid_cov_caller<U, Caller, variant<Ts...>, variant<Ts...>>::value;
};

/*
 * Checks if a type has a valid mean call for any of the types in a variant.
 */
template <typename U, typename Caller, typename A, typename = void>
struct has_valid_variant_mean_caller : public std::false_type {};

/*
 * Collapse from the right
 */
template <typename U, typename Caller, typename A>
struct has_valid_variant_mean_caller<U, Caller, variant<A>,
                                     std::enable_if_t<!is_variant<A>::value>> {
  static constexpr bool value = has_valid_mean_caller<U, Caller, A>::value;
};

template <typename U, typename Caller, typename A, typename... Ts>
struct has_valid_variant_mean_caller<U, Caller, variant<A, Ts...>,
                                     std::enable_if_t<!is_variant<A>::value>> {
  static constexpr bool value =
      has_valid_mean_caller<U, Caller, A>::value ||
      has_valid_variant_mean_caller<U, Caller, variant<Ts...>>::value;
};

} // namespace albatross

#endif
