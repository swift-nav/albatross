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

// Batch covariance method traits
DEFINE_CLASS_METHOD_TRAITS(_call_impl_vector);
DEFINE_CLASS_METHOD_TRAITS(_call_impl_vector_diagonal);

template <typename U, typename... Args>
class has_valid_call_impl
    : public has__call_impl_with_return_type<
          const U, double, typename const_ref<Args>::type...> {};

template <typename U, typename... Args>
class has_possible_call_impl : public has__call_impl<U, Args &...> {};

/*
 * Batch covariance traits - detect _call_impl_vector methods
 */

// Detect _call_impl_vector for cross-covariance (different types)
template <typename U, typename X, typename Y>
class has_valid_call_impl_vector
    : public has__call_impl_vector_with_return_type<
          const U, Eigen::MatrixXd, typename const_ref<std::vector<X>>::type,
          typename const_ref<std::vector<Y>>::type, ThreadPool *> {};

// Detect _call_impl_vector for symmetric case (same type, two vector args)
template <typename U, typename X>
class has_valid_call_impl_vector_symmetric
    : public has__call_impl_vector_with_return_type<
          const U, Eigen::MatrixXd, typename const_ref<std::vector<X>>::type,
          typename const_ref<std::vector<X>>::type, ThreadPool *> {};

// Detect _call_impl_vector for optimized symmetric case (single vector arg)
// Signature: Eigen::MatrixXd _call_impl_vector(const std::vector<X>&, ThreadPool*) const
template <typename U, typename X>
class has_valid_call_impl_vector_single_arg
    : public has__call_impl_vector_with_return_type<
          const U, Eigen::MatrixXd, typename const_ref<std::vector<X>>::type,
          ThreadPool *> {};

// Detect _call_impl_vector_diagonal for diagonal extraction
template <typename U, typename X>
class has_valid_call_impl_vector_diagonal
    : public has__call_impl_vector_diagonal_with_return_type<
          const U, Eigen::VectorXd, typename const_ref<std::vector<X>>::type,
          ThreadPool *> {};

/*
 * Measurement inner type extraction - extracts X from Measurement<X>
 * For non-Measurement types, returns the type itself.
 */
template <typename T, typename = void> struct measurement_inner {
  using type = T;
};

template <typename X>
struct measurement_inner<Measurement<X>, void> {
  using type = X;
};

template <typename T>
using measurement_inner_t = typename measurement_inner<T>::type;

/*
 * has_valid_batch_or_measurement_batch
 *
 * Checks if a covariance function has batch support for types X, Y, either:
 * 1. Directly via _call_impl_vector(vector<X>, vector<Y>, pool), OR
 * 2. Via Measurement unwrapping: if X=Measurement<X'> and/or Y=Measurement<Y'>,
 *    check for _call_impl_vector(vector<X'>, vector<Y'>, pool)
 *
 * This trait is used by Sum/Product compositions to enable batch methods
 * when the underlying covariance has batch support for unwrapped types.
 */
template <typename CovFunc, typename X, typename Y>
class has_valid_batch_or_measurement_batch {
  // Get the inner types (same type if not a Measurement)
  using InnerX = measurement_inner_t<X>;
  using InnerY = measurement_inner_t<Y>;

public:
  static constexpr bool value =
      // Direct batch support for X, Y
      has_valid_call_impl_vector<CovFunc, X, Y>::value ||
      // Batch support for unwrapped types (only relevant if X or Y is a
      // Measurement) Note: if neither is a Measurement, InnerX=X and InnerY=Y,
      // so this duplicates the first check, which is fine.
      ((!std::is_same<X, InnerX>::value || !std::is_same<Y, InnerY>::value) &&
       has_valid_call_impl_vector<CovFunc, InnerX, InnerY>::value);
};

/*
 * has_valid_batch_or_measurement_batch_single_arg
 *
 * Checks if a covariance function has single-arg symmetric batch support for type X, either:
 * 1. Directly via _call_impl_vector(vector<X>, pool), OR
 * 2. Via Measurement unwrapping: if X=Measurement<X'>,
 *    check for _call_impl_vector(vector<X'>, pool)
 *
 * This trait is used by Sum/Product compositions to enable single-arg batch methods
 * when the underlying covariance has single-arg batch support for unwrapped types.
 */
template <typename CovFunc, typename X>
class has_valid_batch_or_measurement_batch_single_arg {
  // Get the inner type (same type if not a Measurement)
  using InnerX = measurement_inner_t<X>;

public:
  static constexpr bool value =
      // Direct single-arg batch support for X
      has_valid_call_impl_vector_single_arg<CovFunc, X>::value ||
      // Single-arg batch support for unwrapped type (only relevant if X is a Measurement)
      (!std::is_same<X, InnerX>::value &&
       has_valid_call_impl_vector_single_arg<CovFunc, InnerX>::value);
};

/*
 * has_valid_cov_caller
 */

DEFINE_CLASS_METHOD_TRAITS(call);
DEFINE_CLASS_METHOD_TRAITS(call_vector);

template <typename CovFunc, typename Caller, typename... Args>
class has_valid_cov_caller
    : public has_call_with_return_type<Caller, double,
                                       typename const_ref<CovFunc>::type,
                                       typename const_ref<Args>::type...> {};

// Detect if caller supports batch operations (call_vector)
template <typename CovFunc, typename Caller, typename X, typename Y>
class has_valid_cov_caller_vector
    : public has_call_vector_with_return_type<
          Caller, Eigen::MatrixXd, typename const_ref<CovFunc>::type,
          typename const_ref<std::vector<X>>::type,
          typename const_ref<std::vector<Y>>::type, ThreadPool *> {};

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
    : public has_valid_cross_cov_caller<U, Caller, A, B> {};

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
 * Batch variant traits - checks if a caller has valid batch (call_vector)
 * support for any of the types in a variant.
 *
 * These mirror the pointwise variant traits above, but check for call_vector
 * instead of call.
 */
template <typename U, typename Caller, typename A, typename B, typename = void>
struct has_valid_variant_cov_caller_vector : public std::false_type {};

// Base case: single element variant (right side)
template <typename U, typename Caller, typename A, typename B>
struct has_valid_variant_cov_caller_vector<
    U, Caller, A, variant<B>, std::enable_if_t<!is_variant<A>::value>> {
  static constexpr bool value =
      has_valid_cov_caller_vector<U, Caller, A, B>::value;
};

// Recursive case: check first element, then rest (right side)
template <typename U, typename Caller, typename A, typename B, typename... Ts>
struct has_valid_variant_cov_caller_vector<
    U, Caller, A, variant<B, Ts...>, std::enable_if_t<!is_variant<A>::value>> {
  static constexpr bool value =
      has_valid_cov_caller_vector<U, Caller, A, B>::value ||
      has_valid_variant_cov_caller_vector<U, Caller, A, variant<Ts...>>::value;
};

// Base case: single element variant (left side)
template <typename U, typename Caller, typename A, typename B>
struct has_valid_variant_cov_caller_vector<
    U, Caller, variant<A>, B, std::enable_if_t<!is_variant<B>::value>> {
  static constexpr bool value =
      has_valid_cov_caller_vector<U, Caller, A, B>::value;
};

// Recursive case: check first element, then rest (left side)
template <typename U, typename Caller, typename A, typename B, typename... Ts>
struct has_valid_variant_cov_caller_vector<
    U, Caller, variant<A, Ts...>, B, std::enable_if_t<!is_variant<B>::value>> {
  static constexpr bool value =
      has_valid_cov_caller_vector<U, Caller, A, B>::value ||
      has_valid_variant_cov_caller_vector<U, Caller, variant<Ts...>, B>::value;
};

// Both sides are variants - check if any valid pair exists
// This is used when both arguments are variant vectors
template <typename U, typename Caller, typename XVariant, typename YVariant>
struct has_valid_variant_cov_caller_vector_both : public std::false_type {};

template <typename U, typename Caller, typename... Xs, typename... Ys>
struct has_valid_variant_cov_caller_vector_both<U, Caller, variant<Xs...>,
                                                variant<Ys...>> {
  // Use same logic as single-sided: any valid pair is sufficient
  // because we assert homogeneity at runtime
  static constexpr bool value =
      has_valid_variant_cov_caller_vector<U, Caller, variant<Xs...>,
                                          variant<Ys...>>::value;
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

/*
 * Sometimes we need to distinguish between raw types for which we would
 * explicitly write covariance function _call_impl methods and wrapped
 * types which get handled by the callers.hpp logic.  For example, if
 * you write a covariance function with _call_impl(X, X) you don't need
 * to also write one which handles LinearCombination<X> types, that is
 * handled by default inside the caller logic. Here we add a trait to
 * be able to distinguish between a basic type and a composite type.
 */
template <typename X> struct is_basic_type : std::true_type {};

/*
 * measurement_inner_type - extract the inner type from Measurement<X>
 */
template <typename T> struct measurement_inner_type;

template <typename X> struct measurement_inner_type<Measurement<X>> {
  using type = X;
};

template <typename T>
using measurement_inner_type_t = typename measurement_inner_type<T>::type;

template <typename X>
struct is_basic_type<LinearCombination<X>> : std::false_type {};

template <typename X> struct is_basic_type<Measurement<X>> : std::false_type {};

template <typename... Ts>
struct is_basic_type<variant<Ts...>> : std::false_type {};

} // namespace albatross

#endif
