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

#ifndef ALBATROSS_CORE_TRAITS_H
#define ALBATROSS_CORE_TRAITS_H

#include "cereal/details/traits.hpp"
#include "core/declarations.h"
#include <utility>

namespace albatross {

/*
 * This little trick was borrowed from cereal, you an think of it as
 * a function that will always return false ... but that doesn't
 * get resolved until template instantiation, which when combined
 * with a static assert let's you include a static assert that
 * only triggers with a particular template parameter is used.
 */
template <class T> struct delay_static_assert : std::false_type {};

/*
 * In CovarianceFunction we frequently inspect for definitions of
 * call_impl_ which MUST be defined for const references to objects
 * (so that repeated covariance matrix evaluations return the same thing
 *  and so the computations are not repeatedly copying.)
 * This type conversion utility will turn a type `T` into `const T&`
 */
template <class T> struct call_impl_arg_type {
  typedef
      typename std::add_lvalue_reference<typename std::add_const<T>::type>::type
          type;
};

/*
 * This determines whether or not a class has a method defined for,
 *   `operator() (const X &x, const Y &y, const Z &z, ...)`
 * The result of the inspection gets stored in the member `value`.
 */
template <typename T, typename... Args> class has_call_operator {

  template <typename C,
            typename = decltype(std::declval<C>()(
                std::declval<typename call_impl_arg_type<Args>::type>()...))>
  static std::true_type test(C *);
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

/*
 * This determines whether or not a class has a method defined for,
 *   `double call_impl_(const X &x, const Y &y, const Z &z, ...)`
 * The result of the inspection gets stored in the member `value`.
 */
template <typename T, typename... Args> class has_valid_call_impl {

  template <typename C>
  static typename std::is_same<
      decltype(std::declval<const C>().call_impl_(
          std::declval<typename call_impl_arg_type<Args>::type>()...)),
      double>::type
  test(C *);
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

/*
 * This determines whether or not a class has a method defined for
 * something close to, but not quite, a valid call_impl_.  For example
 * if a class has:
 *   double call_impl_(const X x)
 * or
 *   double call_impl_(X &x)
 * or
 *   int call_impl_(const X &x)
 * those are nearly correct but the required `const X &x` in which
 * case this trait can be used to warn the user.
 */
template <typename T, typename... Args> class has_possible_call_impl {
  template <typename C, typename = decltype(std::declval<C>().call_impl_(
                            std::declval<Args &>()...))>
  static std::true_type test(int);
  template <typename C> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename T, typename... Args> class has_invalid_call_impl {
public:
  static constexpr bool value = (has_possible_call_impl<T, Args...>::value &&
                                 !has_valid_call_impl<T, Args...>::value);
};

/*
 * This determines whether or not a class, T, has a member,
 *   `T.name_`
 * The result of the inspection gets stored in the member `value`.
 */
template <typename T> class has_name_ {
  template <typename C, typename = decltype(std::declval<C>().name_)>
  static std::true_type test(int);
  template <typename C> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

/*
 * Inspects T to see if it has a class level type called FitType,
 * the result is returned in ::value.
 */
template <typename T> class has_fit_type {
  template <typename C, typename = typename C::FitType>
  static std::true_type test(int);
  template <typename C> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

/*
 * One way to tell the difference between a RegressionModel
 * and a SerializableRegressionModel is by inspecting for a
 * FitType.
 */
template <typename T> using is_serializable_regression_model = has_fit_type<T>;

/*
 * This traits helper class defines `::type` to be `T::FitType`
 * if a type with that name has been defined for T and will
 * otherwise be `void`.
 */
template <typename T> class fit_type_or_void {
  template <typename C, typename = typename C::FitType>
  static typename C::FitType test(int);
  template <typename C> static void test(...);

public:
  typedef decltype(test<T>(0)) type;
};

/*
 * Helper function for enable_if and is_serializable_regression_model
 */
template <typename X, typename T>
using enable_if_serializable =
    std::enable_if<is_serializable_regression_model<X>::value, T>;

/*
 * Will result in substitution failure if X is not serializable and
 * otherwise resolves to X::FitType.
 */
template <typename X>
using fit_type_if_serializable =
    typename enable_if_serializable<X,
                                    typename fit_type_or_void<X>::type>::type;

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

/*
 * This helper function takes a model and decides which base class it extends
 * by inspecting whether or not the ModelType has a FitType.
 */
template <typename FeatureType, typename ModelType>
class choose_regression_model_implementation {
  template <typename C, typename = typename C::FitType>
  static SerializableRegressionModel<FeatureType, typename C::FitType> *
  test(int);

  template <typename C> static RegressionModel<FeatureType> *test(...);

public:
  typedef typename std::remove_pointer<decltype(test<ModelType>(0))>::type type;
};

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
  static constexpr bool value = decltype(test<X>(0))::value;
};

/*
 * This set of trait logic checks if a type has any call_impl_ method
 * implemented (including private methods) by hijacking name hiding.
 * Namely if a derived class overloads a method the base methods will
 * be hidden.  So by starting with a base class with a known method
 * then extending that class you can determine if the derived class
 * included any other methods with that name.
 *
 * https://stackoverflow.com/questions/1628768/why-does-an-overridden-function-in-the-derived-class-hide-other-overloads-of-the
 */
namespace detail {

struct DummyType {};

struct BaseWithPublicCallImpl {
  // This method will be accessible in `MultiInherit` only if
  // the class U doesn't contain any methods with the same name.
  double call_impl_(const DummyType &) const { return -1.; }
};

template <typename U>
struct MultiInherit : public U, public BaseWithPublicCallImpl {};
}

template <typename U> class has_any_call_impl {
  template <typename T>
  static typename std::enable_if<
      has_valid_call_impl<detail::MultiInherit<T>, detail::DummyType>::value,
      std::false_type>::type
  test(int);
  template <typename T> static std::true_type test(...);

public:
  static constexpr bool value = decltype(test<U>(0))::value;
};

} // namespace albatross

#endif
