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
 * This determines whether or not a class has a method defined for,
 *   `FitType fit_(const std::vector<FeatureType>& features,
 *                 const MarginalDistribution &)`
 * The result of the inspection gets stored in the member `value`.
 */
// template <typename T, typename FeatureType> class has_fit {
//
//  template <typename C, typename = decltype(std::declval<const C>().fit_(
//                            const std::vector<FeatureType> &,
//                            const MarginalDistribution &))>
//  static std::true_type test(C *);
//  template <typename> static std::false_type test(...);
//
// public:
//  static constexpr bool value = decltype(test<T>(0))::value;
//};

/*
 * This determines FitType from a method with signature,
 *   `FitType fit_(const std::vector<FeatureType>& features,
 *                 const MarginalDistribution &)`
 * The result of the inspection is the FitType stored in `type`.
 */
// template <typename T, typename FeatureType> class fit_type {
//
//  template <typename C>
//  static typename decltype(std::declval<const C>().fit_(
//      const std::vector<FeatureType> &, const MarginalDistribution &))
//  test(C *);
//
// public:
//  typedef decltype(test<T>(0)) type;
//};

///*
// * Inspects T to see if it has a class level type called FitType,
// * the result is returned in ::value.
// */
// template <typename T> class has_fit_type {
//  template <typename C, typename = typename C::FitType>
//  static std::true_type test(int);
//  template <typename C> static std::false_type test(...);
//
// public:
//  static constexpr bool value = decltype(test<T>(0))::value;
//};
//
///*
// * One way to tell the difference between a RegressionModel
// * and a SerializableRegressionModel is by inspecting for a
// * FitType.
// */
// template <typename T> using is_serializable_regression_model =
// has_fit_type<T>;
//
///*
// * This traits helper class defines `::type` to be `T::FitType`
// * if a type with that name has been defined for T and will
// * otherwise be `void`.
// */
// template <typename T> class fit_type_or_void {
//  template <typename C, typename = typename C::FitType>
//  static typename C::FitType test(int);
//  template <typename C> static void test(...);
//
// public:
//  typedef decltype(test<T>(0)) type;
//};
//
///*
// * Helper function for enable_if and is_serializable_regression_model
// */
// template <typename X, typename T>
// using enable_if_serializable =
//    std::enable_if<is_serializable_regression_model<X>::value, T>;
//
///*
// * Will result in substitution failure if X is not serializable and
// * otherwise resolves to X::FitType.
// */
// template <typename X>
// using fit_type_if_serializable =
//    typename enable_if_serializable<X,
//                                    typename fit_type_or_void<X>::type>::type;

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

} // namespace albatross

#endif
