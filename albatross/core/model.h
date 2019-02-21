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
private:
  // Declaring these private makes it impossible to accidentally do things like:
  //     class A : public ModelBase<B> {}
  // or
  //     using A = ModelBase<B>;
  //
  // which if unchecked can lead to some very strange behavior.
  ModelBase() : has_been_fit_(false), insights_(){};
  friend ModelType;

  bool has_been_fit_;
  Insights insights_;
  Fit<ModelType> model_fit_;

public:
  /*
   * Fit
   */
  template <
      typename FeatureType,
      typename std::enable_if<has_valid_fit_impl<ModelType, FeatureType>::value,
                              int>::type = 0>
  Fit<ModelType> fit(const std::vector<FeatureType> &features,
                     const MarginalDistribution &targets) {
    model_fit_ = derived().fit_impl_(features, targets);
    has_been_fit_ = true;
    return model_fit_;
  }

  template <typename FeatureType,
            typename std::enable_if<
                has_possible_fit_impl<ModelType, FeatureType>::value &&
                    !has_valid_fit_impl<ModelType, FeatureType>::value,
                int>::type = 0>
  Fit<ModelType>
  fit(const std::vector<FeatureType> &features,
      const MarginalDistribution &targets) = delete; // Invalid fit_impl_

  template <typename FeatureType,
            typename std::enable_if<
                !has_possible_fit_impl<ModelType, FeatureType>::value &&
                    !has_valid_fit_impl<ModelType, FeatureType>::value,
                int>::type = 0>
  Fit<ModelType>
  fit(const std::vector<FeatureType> &features,
      const MarginalDistribution &targets) = delete; // No fit_impl_ found.

  /*
   * Predict
   */
  template <typename FeatureType>
  Prediction<ModelType, FeatureType>
  predict(const std::vector<FeatureType> &features) const {
    return Prediction<ModelType, FeatureType>(derived(), features);
  }

  /*
   * CRTP Helpers
   */
  ModelType &derived() { return *static_cast<ModelType *>(this); }
  const ModelType &derived() const {
    return *static_cast<const ModelType *>(this);
  }
};
}
#endif
