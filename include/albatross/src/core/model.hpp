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

  friend class JointPredictor;
  friend class MarginalPredictor;
  friend class MeanPredictor;

  template <typename T, typename FeatureType> friend class fit_model_type;

protected:
  ModelBase() : insights_(){};
  Insights insights_;

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
    return *static_cast<const ModelType *>(this);
  }

  template <class Archive>
  void save(Archive &archive, const std::uint32_t) const {
    archive(cereal::make_nvp("params", derived().get_params()));
    archive(cereal::make_nvp("insights", insights_));
  }

  template <class Archive> void load(Archive &archive, const std::uint32_t) {
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
  auto fit(const std::vector<FeatureType> &features,
           const MarginalDistribution &targets) const {
    return _fit(features, targets);
  }

  template <typename FeatureType>
  auto fit(const RegressionDataset<FeatureType> &dataset) const {
    return _fit(dataset.features, dataset.targets);
  }

  CrossValidation<ModelType> cross_validate() const;

  template <typename Strategy>
  Ransac<ModelType, Strategy>
  ransac(const Strategy &strategy, double inlier_threshold,
         std::size_t random_sample_size, std::size_t min_consensus_size,
         std::size_t max_iteration) const;
};
} // namespace albatross
#endif
