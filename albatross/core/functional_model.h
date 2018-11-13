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

#ifndef ALBATROSS_CORE_FUNCTIONAL_MODEL_H
#define ALBATROSS_CORE_FUNCTIONAL_MODEL_H

#include "core/model.h"

namespace albatross {

// This struct is just a type helper to make it obvious that
// the `FitType` used in the Fitter needs to be the same as
// the one used in `Predictor`
template <typename FeatureType, typename FitType> struct GenericModelFunctions {
  // A function which takes a bunch of indices and fits a model
  // to the corresponding subset of data.
  using Fitter = std::function<FitType(const std::vector<FeatureType> &,
                                       const MarginalDistribution &)>;
  // A function which takes a fit and a set of indices
  // and returns a metric which represents how well the model
  // predicted the subset corresponding to the indices.
  using Predictor = std::function<JointDistribution(
      const std::vector<FeatureType> &, const FitType &)>;
  Fitter fitter;
  Predictor predictor;
};

/*
 * A model that uses a single Feature to estimate the value of a double typed
 * target.
 */
template <typename FeatureType, typename FitType>
class FunctionalRegressionModel : public RegressionModel<FeatureType> {
public:
  using Feature = FeatureType;
  FunctionalRegressionModel(
      const typename GenericModelFunctions<FeatureType, FitType>::Fitter
          &fitter,
      const typename GenericModelFunctions<FeatureType, FitType>::Predictor
          &predictor,
      const std::string &&model_name = "generic_functional_regression_model")
      : RegressionModel<FeatureType>(), fitter_(fitter), predictor_(predictor),
        model_name_(model_name){};
  virtual ~FunctionalRegressionModel(){};

  virtual std::string get_name() const override { return model_name_; }

protected:
  virtual void fit_(const std::vector<FeatureType> &features,
                    const MarginalDistribution &targets) override {
    model_fit_ = fitter_(features, targets);
  }

  virtual JointDistribution
  predict_(const std::vector<FeatureType> &features) const override {
    return predictor_(features, model_fit_);
  }

  FitType model_fit_;
  const typename GenericModelFunctions<FeatureType, FitType>::Fitter fitter_;
  const typename GenericModelFunctions<FeatureType, FitType>::Predictor
      predictor_;
  std::string model_name_;
};

template <typename ModelType>
GenericModelFunctions<typename ModelType::Feature, ModelType>
get_generic_functions(ModelType &model) {

  using FeatureType = typename ModelType::Feature;

  GenericModelFunctions<FeatureType, ModelType> funcs;

  funcs.fitter = [&](const std::vector<FeatureType> &features,
                     const MarginalDistribution &targets) {
    model.fit(features, targets);
    return model;
  };

  funcs.predictor = [&](const std::vector<FeatureType> &features,
                        const ModelType &model_) {
    return model_.predict(features);
  };
  return funcs;
}

} // namespace albatross

#endif
