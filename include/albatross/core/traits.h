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

template <typename T> struct is_vector : public std::false_type {};

template <typename T>
struct is_vector<std::vector<T>> : public std::true_type {};

/*
 * This determines whether or not a class, T, has a method,
 *   `std::string T.name() const`
 */
template <typename T> class has_name {
  template <typename C,
            typename ReturnType = decltype(std::declval<const C>().name())>
  static typename std::enable_if<std::is_same<std::string, ReturnType>::value,
                                 std::true_type>::type
  test(int);

  template <typename C> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

/*
 * Like std::is_base_of<T, U> except compares the first template parameter.  Ie,
 *
 *  first_template_param_is_base_of<C<T, X>, C<U, Y>>::value == is_base_of<T,
 * U>::value
 */
template <typename T, typename U>
struct first_template_param_is_base_of : public std::false_type {};

template <template <typename, typename> class Base, typename T, typename U,
          typename P, typename Q>
struct first_template_param_is_base_of<Base<T, P>, Base<U, Q>>
    : public std::is_base_of<T, U> {};

struct is_valid_fit_type_dummy_type {};

/*
 * A valid fit for a ModelType is defined as Fit<AnyBaseOfModelType,
 * AnyFeatureType>.
 */
template <typename ModelType, typename FitType>
struct is_valid_fit_type
    : public first_template_param_is_base_of<
          FitType, Fit<ModelType, is_valid_fit_type_dummy_type>> {};

/*
 * This determines whether or not a class (T) has a method defined for,
 *   `Fit<U, FeatureType> _fit_impl(const std::vector<FeatureType>&,
 *                            const MarginalDistribution &)`
 * where U is a base of T.
 */
template <typename T, typename FeatureType> class has_valid_fit {
  template <typename C,
            typename FitType = decltype(std::declval<const C>()._fit_impl(
                std::declval<const std::vector<FeatureType> &>(),
                std::declval<const MarginalDistribution &>()))>
  static typename is_valid_fit_type<T, FitType>::type test(C *);
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

/*
 * This determines whether or not a class (T) has a method defined for,
 *   `Anything _fit_impl(std::vector<FeatureType>&,
 *                 MarginalDistribution &)`
 */
template <typename T, typename FeatureType> class has_possible_fit {
  template <typename C,
            typename = decltype(std::declval<C>()._fit_impl(
                std::declval<std::vector<FeatureType> &>(),
                std::declval<MarginalDistribution &>()))>
  static std::true_type test(C *);
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

/*
 * Determines which object would be returned from a call to:
 *
 *   T::fit(features, targets);
 */
template <typename T, typename FeatureType> class fit_model_type {
  template <typename C,
            typename FitModelType = decltype(std::declval<const C>().fit(
                std::declval<const std::vector<FeatureType> &>(),
                std::declval<const MarginalDistribution &>()))>
  static FitModelType test(C *);
  template <typename> static void test(...);

public:
  typedef decltype(test<T>(0)) type;
};

/*
 * fit_type
 *
 * Determines which Fit specialization will be returned if you
 * call ModelType::fit.
 */
template <typename ModelType, typename FeatureType, int = 0> struct fit_type {
  typedef void type;
};

template <typename ModelType, typename FeatureType, typename FitType>
struct fit_type<FitModel<ModelType, FitType>, FeatureType> {
  typedef FitType type;
};

template <typename M, typename F>
struct fit_type<M, F>
    : public fit_type<typename fit_model_type<M, F>::type, F> {};

/*
 * A valid predict method has a signature that looks like:
 *
 *   PredictType _predict_impl(const std::vector<FeatureType> &,
 *                       const FitType &,
 *                       const PredictTypeIdentity<PredictType>) const;
 */
template <typename T, typename FeatureType, typename FitType,
          typename PredictType>
class has_valid_predict {
  template <
      typename C,
      typename ReturnType = decltype(std::declval<const C>()._predict_impl(
          std::declval<const std::vector<FeatureType> &>(),
          std::declval<const FitType &>(),
          std::declval<PredictTypeIdentity<PredictType>>()))>
  static typename std::enable_if<std::is_same<PredictType, ReturnType>::value,
                                 std::true_type>::type
  test(C *);
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename T, typename FeatureType, typename FitType>
using has_valid_predict_mean =
    has_valid_predict<T, FeatureType, FitType, Eigen::VectorXd>;

template <typename T, typename FeatureType, typename FitType>
using has_valid_predict_marginal =
    has_valid_predict<T, FeatureType, FitType, MarginalDistribution>;

template <typename T, typename FeatureType, typename FitType>
using has_valid_predict_joint =
    has_valid_predict<T, FeatureType, FitType, JointDistribution>;

/*
 * Methods for inspecting `Prediction` types.
 */
template <typename T> class has_mean {
  template <typename C,
            typename ReturnType = decltype(std::declval<const C>().mean())>
  static
      typename std::enable_if<std::is_same<Eigen::VectorXd, ReturnType>::value,
                              std::true_type>::type
      test(C *);
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename T> class has_marginal {
  template <typename C,
            typename ReturnType = decltype(std::declval<const C>().marginal())>
  static typename std::enable_if<
      std::is_same<MarginalDistribution, ReturnType>::value,
      std::true_type>::type
  test(C *);
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename T> class has_joint {
  template <typename C,
            typename ReturnType = decltype(std::declval<const C>().joint())>
  static typename std::enable_if<
      std::is_same<JointDistribution, ReturnType>::value, std::true_type>::type
  test(C *);
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

} // namespace albatross

#endif
