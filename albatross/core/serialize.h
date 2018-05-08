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

#ifndef ALBATROSS_CORE_SERIALIZE_H
#define ALBATROSS_CORE_SERIALIZE_H

#include <cereal/archives/json.hpp>
#include <cereal/types/polymorphic.hpp>
#include <iostream>
#include <memory>

#include <core/model.h>

namespace albatross {

template <typename FeatureType, typename ModelFit>
class SerializableRegressionModel : public RegressionModel<FeatureType> {

public:
  using FitType = ModelFit;
  SerializableRegressionModel() : model_fit_(){};
  virtual ~SerializableRegressionModel(){};

  bool operator==(
      const SerializableRegressionModel<FeatureType, ModelFit> &other) const {
    return (this->get_name() == other.get_name() &&
            this->get_params() == other.get_params() &&
            this->has_been_fit() == other.has_been_fit() &&
            model_fit_ == other.get_fit());
  }

  // todo: enable if ModelFit is serializable.
  template <class Archive> void save(Archive &archive) const {
    archive(cereal::make_nvp(
        "model_definition",
        cereal::base_class<RegressionModel<FeatureType>>(this)));
    archive(cereal::make_nvp("model_fit", this->model_fit_));
  }

  template <class Archive> void load(Archive &archive) {
    archive(cereal::make_nvp(
        "model_definition",
        cereal::base_class<RegressionModel<FeatureType>>(this)));
    archive(cereal::make_nvp("model_fit", this->model_fit_));
  }

  ModelFit get_fit() const { return model_fit_; }

protected:
  void fit_(const std::vector<FeatureType> &features,
            const TargetDistribution &targets) {
    model_fit_ = serializable_fit_(features, targets);
  }

  virtual ModelFit
  serializable_fit_(const std::vector<FeatureType> &features,
                    const TargetDistribution &targets) const = 0;

  ModelFit model_fit_;
};
}

#endif
