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

#ifndef ALBATROSS_CORE_MAGIC_H
#define ALBATROSS_CORE_MAGIC_H

#include <utility>

namespace albatross {

/*
 * This determines whether or not a class has a method defined for,
 *   `operator() (X x, Y y, Z z, ...)`
 * The result of the inspection gets stored in the member `value`.
 */
template <typename T, typename... Args> class has_call_operator {
  template <typename C,
            typename = decltype(std::declval<C>()(std::declval<Args>()...))>
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

} // namespace albatross

#endif
