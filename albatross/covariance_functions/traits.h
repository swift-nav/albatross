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
struct MultiInheritCallImpl : public U, public BaseWithPublicCallImpl {};
}

template <typename U> class has_any_call_impl {
  template <typename T>
  static typename std::enable_if<
      has_valid_call_impl<detail::MultiInheritCallImpl<T>,
                          detail::DummyType>::value,
      std::false_type>::type
  test(int);
  template <typename T> static std::true_type test(...);

public:
  static constexpr bool value = decltype(test<U>(0))::value;
};
}

#endif
