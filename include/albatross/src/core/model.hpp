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

#ifndef ALBATROSS_CORE_MODEL_H
#define ALBATROSS_CORE_MODEL_H

namespace albatross {

using Insights = std::map<std::string, std::string>;

constexpr bool DEFAULT_USE_ASYNC = false;

template <typename ModelType> class ModelBase : public ParameterHandlingMixin {

  friend class JointPredictor;
  friend class MarginalPredictor;
  friend class MeanPredictor;

  template <typename T, typename FeatureType> friend class fit_model_type;

protected:
  ModelBase() : insights(), use_async_(DEFAULT_USE_ASYNC){};

  /*
   * Fit
   */
  template <typename FeatureType,
            typename std::enable_if<
                has_valid_fit<ModelType, FeatureType>::value, int>::type = 0>
  auto _fit(const std::vector<FeatureType> &features,
            const MarginalDistribution &targets) const {
    auto fit_output = derived()._fit_impl(features, targets);
    return FitModel<ModelType, decltype(fit_output)>(derived(),
                                                     std::move(fit_output));
  }

  template <
      typename FeatureType,
      typename std::enable_if<has_possible_fit<ModelType, FeatureType>::value &&
                                  !has_valid_fit<ModelType, FeatureType>::value,
                              int>::type = 0>
  void _fit(const std::vector<FeatureType> &features,
            const MarginalDistribution &targets) const
      ALBATROSS_FAIL(FeatureType,
                     "The ModelType *almost* has a _fit_impl method for "
                     "FeatureType, but it appears to be invalid");

  template <typename FeatureType,
            typename std::enable_if<
                !has_possible_fit<ModelType, FeatureType>::value &&
                    !has_valid_fit<ModelType, FeatureType>::value,
                int>::type = 0>
  void _fit(const std::vector<FeatureType> &features,
            const MarginalDistribution &targets) const
      ALBATROSS_FAIL(
          FeatureType,
          "The ModelType is missing a _fit_impl method for FeatureType.");

  template <
      typename PredictFeatureType, typename FitType, typename PredictType,
      typename std::enable_if<has_valid_predict<ModelType, PredictFeatureType,
                                                FitType, PredictType>::value,
                              int>::type = 0>
  PredictType predict_(const std::vector<PredictFeatureType> &features,
                       const FitType &fit_,
                       PredictTypeIdentity<PredictType> &&) const {
    std::cout << "predict_" << std::endl;
    return derived()._predict_impl(features, fit_,
                                   PredictTypeIdentity<PredictType>());
  }

  template <
      typename PredictFeatureType, typename FitType, typename PredictType,
      typename std::enable_if<!has_valid_predict<ModelType, PredictFeatureType,
                                                 FitType, PredictType>::value,
                              int>::type = 0>
  PredictType predict_(const std::vector<PredictFeatureType> &features,
                       const FitType &fit,
                       PredictTypeIdentity<PredictType> &&) const
      ALBATROSS_FAIL(PredictFeatureType,
                     "The ModelType is missing a _predict_impl method for "
                     "PredictFeatureType, FitType, PredictType.");

public:
  /*
   * CRTP Helpers
   */
  ModelType &derived() { return *static_cast<ModelType *>(this); }
  const ModelType &derived() const {
    std::cout << "derived()" << std::endl;
    return *static_cast<const ModelType *>(this);
  }

  bool operator==(const ModelType &other) const {
    return (derived().get_params() == other.get_params() &&
            derived().get_name() == other.get_name() &&
            derived().insights == other.insights);
  }

  template <typename DummyType = ModelType,
            typename std::enable_if<!has_name<DummyType>::value, int>::type = 0>
  std::string get_name() {
    return typeid(ModelType).name();
  }

  template <typename DummyType = ModelType,
            typename std::enable_if<has_name<DummyType>::value, int>::type = 0>
  std::string get_name() {
    return derived().name();
  }

  void set_async_flag(const bool use_async) { use_async_ = use_async; }

  template <typename FeatureType>
  auto fit(const std::vector<FeatureType> &features,
           const MarginalDistribution &targets) const {
    return _fit(features, targets);
  }

  template <typename FeatureType>
  auto fit(const RegressionDataset<FeatureType> &dataset) const {
    return _fit(dataset.features, dataset.targets);
  }

  template <typename FeatureX, typename FeatureY>
  auto fit(const RegressionDataset<FeatureX> &x,
           const RegressionDataset<FeatureY> &y) const {
    return fit(concatenate_datasets(x, y));
  }

  CrossValidation<ModelType> cross_validate() const;

  template <typename Strategy>
  Ransac<ModelType, Strategy>
  ransac(const Strategy &strategy, double inlier_threshold,
         std::size_t random_sample_size, std::size_t min_consensus_size,
         std::size_t max_iteration) const;

  template <typename Strategy>
  Ransac<ModelType, Strategy> ransac(const Strategy &strategy,
                                     const RansacConfig &) const;

  Insights insights;
  bool use_async_;
};
} // namespace albatross
#endif
