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

template <typename ModelType> class ModelBase {
private:
  // Declaring these private makes it impossible to accidentally do things like:
  //     class A : public ModelBase<B> {}
  // or
  //     using A = ModelBase<B>;
  //
  // which if unchecked can lead to some very strange behavior.
  ModelBase() : has_been_fit_(false), insights_(){};
  friend ModelType;

public:
  template <typename FeatureType, typename TargetType>
  void fit(const std::vector<FeatureType> &features, const TargetType &targets);

  template <typename FeatureType>
  Eigen::VectorXd predict(const std::vector<FeatureType> &features) const;

  /*
   * CRTP Helpers
   */

  ModelType &derived() { return *static_cast<ModelType *>(this); }
  const ModelType &derived() const {
    return *static_cast<const ModelType *>(this);
  }

  bool has_been_fit_;
  Insights insights_;
};

template <typename ModelType>
template <typename FeatureType, typename TargetType>
void ModelBase<ModelType>::fit(const std::vector<FeatureType> &features,
                               const TargetType &targets) {
  derived().fit_(features, targets);
  has_been_fit_ = true;
}

template <typename ModelType>
template <typename FeatureType>
Eigen::VectorXd
ModelBase<ModelType>::predict(const std::vector<FeatureType> &features) const {
  assert(has_been_fit_);
  return derived().predict_(features);
}
}
#endif
