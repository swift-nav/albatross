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

#include "model.h"
#include "serialize.h"
#include "traits.h"

namespace albatross {

/*
 * This helper function takes a model which is being adapted (SubModelType)
 * and decides which base class to extend by inspecting whether or not the
 * SubModelType is a pure RegressionModel or a SerializableRegressionModel.
 */
template <typename FeatureType, typename SubModelType>
class choose_regression_model_implementation {
  template <typename C, typename = typename C::FitType>
  static SerializableRegressionModel<FeatureType, typename C::FitType> *
  test(int);

  template <typename C> static RegressionModel<FeatureType> *test(...);

public:
  typedef
      typename std::remove_pointer<decltype(test<SubModelType>(0))>::type type;
};

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

  SubModelType sub_model_;
};
} // namespace albatross

#endif
