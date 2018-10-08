/*
 * Copyright (C) 2018 Swift Navigation Inc
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_CORE_MODEL_ADAPTER_H
#define ALBATROSS_CORE_MODEL_ADAPTER_H

#include "functional_model.h"
#include "model.h"
#include "serialize.h"
#include "traits.h"

namespace albatross {

/*
 * This provides a way of creating a RegressionModel<X> which
 * wraps a RegressionModel<Y> as long as you provide a way of
 * converting from X to Y.
 *
 * A good example can be found in the definition of LinearRegression.
 *
 * Note that RegressionModelImplementation exists in case one wants
 * to adapt something that has extended RegressionModel.
 */
template <typename FeatureType, typename SubModelType>
class AdaptedRegressionModel
    : public choose_regression_model_implementation<FeatureType,
                                                    SubModelType>::type {

public:
  using SubFeature = typename SubModelType::Feature;
  using RegressionModelImplementation =
      typename choose_regression_model_implementation<FeatureType,
                                                      SubModelType>::type;

  static_assert(std::is_same<RegressionModelImplementation,
                             RegressionModel<FeatureType>>::value ||
                    std::is_base_of<RegressionModel<FeatureType>,
                                    RegressionModelImplementation>::value,
                "The template parameter RegressionModelImplementation must be "
                "derived from RegressionModel<FeatureType>");

  static_assert(
      !has_fit_type<RegressionModelImplementation>::value ||
          std::is_same<
              typename fit_type_or_void<RegressionModelImplementation>::type,
              typename fit_type_or_void<SubModelType>::type>::value,
      "If the RegressionModelImplementation is serializable, it must have the "
      "same FitType as the sub_model");

  AdaptedRegressionModel() : sub_model_(){};
  AdaptedRegressionModel(const SubModelType &sub_model)
      : sub_model_(sub_model){};
  virtual ~AdaptedRegressionModel(){};

  // This function will often be required by AdaptedModels
  // The default implementation is a null operation.
  virtual const SubFeature
  convert_feature(const FeatureType &parent_feature) const = 0;

  std::string get_name() const override { return sub_model_.get_name(); };

  bool has_been_fit() const override { return sub_model_.has_been_fit(); }

  ParameterStore get_params() const override {
    return map_join(this->params_, sub_model_.get_params());
  }

  template <class Archive> void save(Archive &archive) const {
    archive(cereal::make_nvp(
        "base_class", cereal::base_class<RegressionModelImplementation>(this)));
    archive(cereal::make_nvp("sub_model", sub_model_));
  }

  template <class Archive> void load(Archive &archive) {
    archive(cereal::make_nvp(
        "base_class", cereal::base_class<RegressionModelImplementation>(this)));
    archive(cereal::make_nvp("sub_model", sub_model_));
  }

  void unchecked_set_param(const std::string &name,
                           const Parameter &param) override {

    if (map_contains(this->params_, name)) {
      this->params_[name] = param;
    } else {
      sub_model_.set_param(name, param);
    }
  }

  fit_type_if_serializable<SubModelType> get_fit() const override {
    return sub_model_.get_fit();
  }

  /*
   * Cross validation specializations
   */
  virtual std::vector<JointDistribution> cross_validated_predictions_(
      const RegressionDataset<FeatureType> &dataset,
      const FoldIndexer &fold_indexer,
      const detail::PredictTypeIdentity<JointDistribution> &identity) override {
    const RegressionDataset<SubFeature> converted = convert_dataset(dataset);
    return sub_model_.template cross_validated_predictions<JointDistribution>(
        converted, fold_indexer);
  }

  virtual std::vector<MarginalDistribution> cross_validated_predictions_(
      const RegressionDataset<FeatureType> &dataset,
      const FoldIndexer &fold_indexer,
      const detail::PredictTypeIdentity<MarginalDistribution> &identity)
      override {
    const RegressionDataset<SubFeature> converted = convert_dataset(dataset);
    return sub_model_
        .template cross_validated_predictions<MarginalDistribution>(
            converted, fold_indexer);
  }

  virtual std::vector<Eigen::VectorXd> cross_validated_predictions_(
      const RegressionDataset<FeatureType> &dataset,
      const FoldIndexer &fold_indexer,
      const detail::PredictTypeIdentity<PredictMeanOnly> &identity) override {
    const RegressionDataset<SubFeature> converted = convert_dataset(dataset);
    return sub_model_.template cross_validated_predictions<Eigen::VectorXd>(
        converted, fold_indexer);
  }

  virtual std::unique_ptr<RegressionModel<FeatureType>>
  ransac_model(double inlier_threshold, std::size_t min_inliers,
               std::size_t random_sample_size,
               std::size_t max_iterations) override {

    using FitType = std::unique_ptr<RegressionModel<SubFeature>>;
    GenericModelFunctions<FeatureType, FitType> funcs;

    decltype(funcs.fitter) fitter =
        [&](const std::vector<FeatureType> &features,
            const MarginalDistribution &targets) {
          std::unique_ptr<RegressionModel<SubFeature>> sub_ransac =
              sub_model_.ransac_model(inlier_threshold, min_inliers,
                                      random_sample_size, max_iterations);
          sub_ransac->fit(convert_features(features), targets);
          return std::move(sub_ransac);
        };

    decltype(funcs.predictor) predictor =
        [&](const std::vector<FeatureType> &features, const FitType &model_) {
          return model_->predict(convert_features(features));
        };

    std::unique_ptr<RegressionModel<FeatureType>> adapted_ransac =
        std::make_unique<FunctionalRegressionModel<FeatureType, FitType>>(
            fitter, predictor);
    return adapted_ransac;
  }

protected:
  void fit_(const std::vector<FeatureType> &features,
            const MarginalDistribution &targets) override {
    this->sub_model_.fit(convert_features(features), targets);
  }

  /*
   * In order to make it possible for this model adapter to extend
   * a SerializableRegressionModel we need to define the proper pure virtual
   * serializable_fit_ method.
   */
  fit_type_if_serializable<RegressionModelImplementation>
  serializable_fit_(const std::vector<FeatureType> &features,
                    const MarginalDistribution &targets) const override {
    assert(false &&
           "serializable_fit_ for an adapted model should never be called");
    typename fit_type_or_void<RegressionModelImplementation>::type dummy;
    return dummy;
  }

  JointDistribution
  predict_(const std::vector<FeatureType> &features) const override {
    return sub_model_.template predict<JointDistribution>(
        convert_features(features));
  }

  MarginalDistribution
  predict_marginal_(const std::vector<FeatureType> &features) const override {
    return sub_model_.template predict<MarginalDistribution>(
        convert_features(features));
  }

  Eigen::VectorXd
  predict_mean_(const std::vector<FeatureType> &features) const override {
    return sub_model_.template predict<Eigen::VectorXd>(
        convert_features(features));
  }

  const std::vector<SubFeature>
  convert_features(const std::vector<FeatureType> &parent_features) const {
    std::vector<SubFeature> converted;
    for (const auto &f : parent_features) {
      converted.push_back(convert_feature(f));
    }
    return converted;
  }

  const RegressionDataset<SubFeature>
  convert_dataset(const RegressionDataset<FeatureType> &parent_dataset) const {
    const auto converted_features = convert_features(parent_dataset.features);
    RegressionDataset<SubFeature> converted(converted_features,
                                            parent_dataset.targets);
    converted.metadata = parent_dataset.metadata;
    return converted;
  }

  SubModelType sub_model_;
};

} // namespace albatross

#endif
