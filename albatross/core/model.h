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

template <typename ModelType> class ModelBase : public ParameterHandlingMixin {

  template <typename X, typename Y, typename Z> friend class Prediction;

  template <typename T, typename FeatureType> friend class fit_model_type;

private:
  // Declaring these private makes it impossible to accidentally do things like:
  //     class A : public ModelBase<B> {}
  // or
  //     using A = ModelBase<B>;
  //
  // which if unchecked can lead to some very strange behavior.
  ModelBase() : insights_(){};
  friend ModelType;
  Insights insights_;

  /*
   * Fit
   */
  template <typename FeatureType,
            typename std::enable_if<
                has_valid_fit<ModelType, FeatureType>::value, int>::type = 0>
  auto fit_(const std::vector<FeatureType> &features,
            const MarginalDistribution &targets) const {
    auto fit = derived().fit(features, targets);
    return FitModel<ModelType, decltype(fit)>(derived(), std::move(fit));
  }

  template <
      typename FeatureType,
      typename std::enable_if<has_possible_fit<ModelType, FeatureType>::value &&
                                  !has_valid_fit<ModelType, FeatureType>::value,
                              int>::type = 0>
  void fit_(const std::vector<FeatureType> &features,
            const MarginalDistribution &targets) const = delete; // Invalid fit

  template <typename FeatureType,
            typename std::enable_if<
                !has_possible_fit<ModelType, FeatureType>::value &&
                    !has_valid_fit<ModelType, FeatureType>::value,
                int>::type = 0>
  void
  fit_(const std::vector<FeatureType> &features,
       const MarginalDistribution &targets) const = delete; // No fit found.

  template <
      typename PredictFeatureType, typename FitType, typename PredictType,
      typename std::enable_if<has_valid_predict<ModelType, PredictFeatureType,
                                                FitType, PredictType>::value,
                              int>::type = 0>
  PredictType predict_(const std::vector<PredictFeatureType> &features,
                       const FitType &fit,
                       PredictTypeIdentity<PredictType> &&) const {
    return derived().predict(features, fit, PredictTypeIdentity<PredictType>());
  }

  template <
      typename PredictFeatureType, typename FitType, typename PredictType,
      typename std::enable_if<!has_valid_predict<ModelType, PredictFeatureType,
                                                 FitType, PredictType>::value,
                              int>::type = 0>
  PredictType predict_(
      const std::vector<PredictFeatureType> &features, const FitType &fit,
      PredictTypeIdentity<PredictType> &&) const = delete; // No valid predict.

  /*
   * CRTP Helpers
   */
  ModelType &derived() { return *static_cast<ModelType *>(this); }
  const ModelType &derived() const {
    return *static_cast<const ModelType *>(this);
  }

public:
  template <class Archive> void save(Archive &archive) const {
    archive(cereal::make_nvp("params", derived().get_params()));
    archive(cereal::make_nvp("insights", insights_));
  }

  template <class Archive> void load(Archive &archive) {
    ParameterStore params;
    archive(cereal::make_nvp("params", params));
    derived().set_params(params);
    archive(cereal::make_nvp("insights", insights_));
  }

  bool operator==(const ModelType &other) const {
    return (derived().get_params() == other.get_params() &&
            derived().get_name() == other.get_name() &&
            derived().insights_ == other.insights_);
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

  template <typename FeatureType>
  auto get_fit_model(const std::vector<FeatureType> &features,
                     const MarginalDistribution &targets) const {
    return fit_(features, targets);
  }

  template <typename FeatureType>
  auto get_fit_model(const RegressionDataset<FeatureType> &dataset) const {
    return fit_(dataset.features, dataset.targets);
  }

  CrossValidation<ModelType> cross_validate() const;

  template <typename MetricType>
  Ransac<ModelType, MetricType>
  ransac(const MetricType &metric, double inlier_threshold,
         std::size_t min_inliers, std::size_t random_sample_size,
         std::size_t max_iterations) const;
};
}
#endif
