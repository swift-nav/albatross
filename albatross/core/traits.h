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

namespace albatross {

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

template <typename T, typename U>
struct template_param_is_base_of : public std::false_type {};

template <template <typename> class Base, typename T, typename U>
struct template_param_is_base_of<Base<T>, Base<U>>
    : public std::is_base_of<T, U> {};

template <typename ModelType, typename FitType>
struct is_valid_fit_type
    : public template_param_is_base_of<FitType, Fit<ModelType>> {};

/*
 * This determines whether or not a class (T) has a method defined for,
 *   `Fit<T> fit_impl_(const std::vector<FeatureType>&,
 *                const MarginalDistribution &)`
 * The result of the inspection gets stored in the member `value`.
 */
template <typename T, typename FeatureType> class has_valid_fit_impl {
  template <typename C,
            typename FitType = decltype(std::declval<const C>().fit_impl_(
                std::declval<const std::vector<FeatureType> &>(),
                std::declval<const MarginalDistribution &>()))>
  static typename is_valid_fit_type<T, FitType>::type test(C *);
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

/*
 * This determines whether or not a class (T) has a method defined for,
 *   `Anything fit_impl_(std::vector<FeatureType>&,
 *                       MarginalDistribution &)`
 * The result of the inspection gets stored in the member `value`.
 */
template <typename T, typename FeatureType> class has_possible_fit_impl {
  template <typename C, typename = decltype(std::declval<C>().fit_impl_(
                            std::declval<std::vector<FeatureType> &>(),
                            std::declval<MarginalDistribution &>()))>
  static std::true_type test(C *);
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename T, typename FeatureType, typename PredictType>
class has_valid_predict_ {
  template <typename C,
            typename ReturnType = decltype(std::declval<const C>().predict_(
                std::declval<const std::vector<FeatureType> &>(),
                std::declval<PredictTypeIdentity<PredictType> &&>()))>
  static typename std::enable_if<std::is_same<PredictType, ReturnType>::value,
                                 std::true_type>::type
  test(C *);
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename T, typename FeatureType>
using has_valid_predict_mean =
    has_valid_predict_<T, FeatureType, Eigen::VectorXd>;

template <typename T, typename FeatureType>
using has_valid_predict_marginal =
    has_valid_predict_<T, FeatureType, MarginalDistribution>;

template <typename T, typename FeatureType>
using has_valid_predict_joint =
    has_valid_predict_<T, FeatureType, JointDistribution>;

} // namespace albatross

#endif
