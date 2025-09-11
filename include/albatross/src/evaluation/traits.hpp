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

#ifndef ALBATROSS_EVALUATION_TRAITS_H
#define ALBATROSS_EVALUATION_TRAITS_H

namespace albatross {

/*
 * Cross validation traits
 */
template <typename T, typename FeatureType, typename PredictType,
          typename GroupKey>
class has_valid_cross_validated_predictions {
  template <typename C,
            typename ReturnType =
                decltype(std::declval<const C>().cross_validated_predictions(
                    std::declval<const RegressionDataset<FeatureType> &>(),
                    std::declval<const GroupIndexer<GroupKey> &>(),
                    std::declval<PredictTypeIdentity<PredictType>>()))>
  static typename std::enable_if<
      std::is_same<std::map<GroupKey, PredictType>, ReturnType>::value,
      std::true_type>::type
  test(C *);
  template <typename>
  static std::false_type test(...);

 public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename T, typename FeatureType, typename GroupKey>
using has_valid_cv_mean =
    has_valid_cross_validated_predictions<T, FeatureType, Eigen::VectorXd,
                                          GroupKey>;

template <typename T, typename FeatureType, typename GroupKey>
using has_valid_cv_marginal =
    has_valid_cross_validated_predictions<T, FeatureType, MarginalDistribution,
                                          GroupKey>;

template <typename T, typename FeatureType, typename GroupKey>
using has_valid_cv_joint =
    has_valid_cross_validated_predictions<T, FeatureType, JointDistribution,
                                          GroupKey>;

/*
 * PredictionMetrics
 */
template <typename T, typename PredictType>
class is_specific_prediction_metric {
  template <typename C,
            typename ReturnType = decltype(std::declval<const C>().operator()(
                std::declval<const PredictType &>(),
                std::declval<const MarginalDistribution &>()))>
  static typename std::enable_if<std::is_same<double, ReturnType>::value,
                                 std::true_type>::type
  test(C *);
  template <typename>
  static std::false_type test(...);

 public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename T>
struct is_prediction_metric {
  static constexpr bool value =
      (is_specific_prediction_metric<T, Eigen::VectorXd>::value ||
       is_specific_prediction_metric<T, MarginalDistribution>::value ||
       is_specific_prediction_metric<T, JointDistribution>::value);
};

/*
 * ModelMetric
 */
template <typename T, typename FeatureType, typename ModelType>
class is_model_metric {
  template <typename C,
            typename ReturnType = decltype(std::declval<const C>().operator()(
                std::declval<const RegressionDataset<FeatureType> &>(),
                std::declval<const ModelType &>()))>
  static typename std::enable_if<std::is_same<double, ReturnType>::value,
                                 std::true_type>::type
  test(C *);
  template <typename>
  static std::false_type test(...);

 public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

}  // namespace albatross

#endif /* ALBATROSS_EVALUATION_TRAITS_H */
