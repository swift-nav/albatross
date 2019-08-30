/*
 * Copyright (C) 2019 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_CORE_FIT_MODEL_H
#define ALBATROSS_CORE_FIT_MODEL_H

namespace albatross {

template <typename ModelType, typename Fit> class FitModel {
public:
  template <typename X, typename Y, typename Z> friend class Prediction;

  static_assert(
      std::is_move_constructible<Fit>::value,
      "Fit type must be move constructible to avoid unexpected copying.");

  FitModel(){};

  FitModel(const ModelType &model, const Fit &fit) = delete;

  FitModel(const ModelType &model, Fit &&fit)
      : model_(model), fit_(std::move(fit)) {}

  template <typename PredictFeatureType>
  Prediction<ModelType, PredictFeatureType, Fit>
  predict(const std::vector<PredictFeatureType> &features) const {
    return Prediction<ModelType, PredictFeatureType, Fit>(model_, fit_,
                                                          features);
  }

  template <typename PredictFeatureType>
  Prediction<ModelType, Measurement<PredictFeatureType>, Fit>
  predict_with_measurement_noise(
      const std::vector<PredictFeatureType> &features) const {
    std::vector<Measurement<PredictFeatureType>> measurements;
    for (const auto &f : features) {
      measurements.emplace_back(Measurement<PredictFeatureType>(f));
    }
    return Prediction<ModelType, Measurement<PredictFeatureType>, Fit>(
        model_, fit_, measurements);
  }

  Fit get_fit() const { return fit_; }

  Fit &get_fit() { return fit_; }

  ModelType get_model() const { return model_; };

  ModelType &get_model() { return model_; };

  bool operator==(const FitModel &other) const {
    return (model_ == other.model_ && fit_ == other.fit_);
  }

private:
  ModelType model_;
  Fit fit_;
};
} // namespace albatross
#endif
