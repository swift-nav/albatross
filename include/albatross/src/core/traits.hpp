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
 * A valid fit is defined as simply anything which matches the pattern:
 *
 *     Fit<Anything>.
 */
template <typename FitType>
struct is_valid_fit_type : public std::false_type {};

template <typename FitParameter>
struct is_valid_fit_type<Fit<FitParameter>> : public std::true_type {};

DEFINE_CLASS_METHOD_TRAITS(_fit_impl);

/*
 * This determines whether or not a class (T) has a method defined for,
 *   `Fit<U, FeatureType> _fit_impl(const std::vector<FeatureType>&,
 *                            const MarginalDistribution &)`
 * where U is a base of T.
 */
template <typename T, typename FeatureType> class has_valid_fit {
  template <typename C,
            typename FitType = typename class_method__fit_impl_traits<
                C, const std::vector<FeatureType> &,
                const MarginalDistribution &>::return_type>
  static typename is_valid_fit_type<FitType>::type test(C *);
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename T, typename FeatureType>
class has_possible_fit : public has__fit_impl<T, std::vector<FeatureType> &,
                                              MarginalDistribution &> {};

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
 * Determines which object would be returned from a call to:
 *
 *   T::fit_from_prediction(features, joint_prediction);
 */
template <typename T, typename FeatureType> class fit_from_prediction_type {
  template <typename C,
            typename FitModelType =
                decltype(std::declval<const C>().fit_from_prediction(
                    std::declval<const std::vector<FeatureType> &>(),
                    std::declval<const JointDistribution &>()))>
  static FitModelType test(C *);
  template <typename> static void test(...);

public:
  typedef decltype(test<T>(0)) type;
};

/*
 * Determines the type of updated_fit in a call along the lines of :
 *
 *   auto fit = model.fit(dataset);
 *   auto updated_fit = update(fit, other_dataset);
 */
template <typename T, typename FeatureType> class updated_fit_type {
  template <typename C,
            typename UpdatedFitType = decltype(
                update(std::declval<const C>(),
                       std::declval<const RegressionDataset<FeatureType> &>()))>
  static UpdatedFitType test(C *);
  template <typename> static void test(...);

public:
  typedef decltype(test<T>(0)) type;
};

DEFINE_CLASS_METHOD_TRAITS(_predict_impl);

template <typename T, typename FeatureType, typename FitType,
          typename PredictType>
class has_valid_predict
    : public has__predict_impl_with_return_type<
          T, PredictType, typename const_ref<std::vector<FeatureType>>::type,
          typename const_ref<FitType>::type, PredictTypeIdentity<PredictType>> {
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

DEFINE_CLASS_METHOD_TRAITS(_mean);

template <typename T, typename ModelType, typename FeatureType,
          typename FitType>
struct can_predict_mean
    : public has__mean<T, typename const_ref<ModelType>::type,
                       typename const_ref<FitType>::type,
                       typename const_ref<std::vector<FeatureType>>::type> {};

DEFINE_CLASS_METHOD_TRAITS(_marginal);

template <typename T, typename ModelType, typename FeatureType,
          typename FitType>
struct can_predict_marginal
    : public has__marginal<T, typename const_ref<ModelType>::type,
                           typename const_ref<FitType>::type,
                           typename const_ref<std::vector<FeatureType>>::type> {
};

DEFINE_CLASS_METHOD_TRAITS(_joint);

template <typename T, typename ModelType, typename FeatureType,
          typename FitType>
struct can_predict_joint
    : public has__joint<T, typename const_ref<ModelType>::type,
                        typename const_ref<FitType>::type,
                        typename const_ref<std::vector<FeatureType>>::type> {};

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
