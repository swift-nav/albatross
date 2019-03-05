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

#ifndef ALBATROSS_CORE_FIT_H
#define ALBATROSS_CORE_FIT_H

namespace albatross {

template <typename ModelType, typename Fit>
class FitModel {

  template <typename X, typename Y, typename Z>
  friend class Prediction;

 public:

  static_assert(std::is_move_constructible<Fit>::value,
                "Fit type must be move constructible to avoid unexpected copying.");

  FitModel(const ModelType &model,
           const Fit &&fit)
      : model_(model), fit_(std::move(fit)) {}

 public:

  template <typename PredictFeatureType>
  Prediction<ModelType, PredictFeatureType, Fit>
  get_prediction(const std::vector<PredictFeatureType> &features) const {
    return Prediction<ModelType, PredictFeatureType, Fit>(*this, features);
  }

  const ModelType &model_;
  const Fit fit_;

};

}
#endif
