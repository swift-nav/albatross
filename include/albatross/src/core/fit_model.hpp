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

  // When FitModel is an lvalue we store a reference to the fit
  // inside the resulting Prediction class.
  template <typename PredictFeatureType>
  const PredictionReference<ModelType, PredictFeatureType, Fit>
  predict(const std::vector<PredictFeatureType> &features) const & {
    return PredictionReference<ModelType, PredictFeatureType, Fit>(model_, fit_,
                                                                   features);
  }

  // When FitModel is an rvalue the Fit will be a temporary so
  // we move it into the Prediction class to be stored there.
  template <typename PredictFeatureType>
  Prediction<ModelType, PredictFeatureType, Fit>
  predict(const std::vector<PredictFeatureType> &features) && {
    return Prediction<ModelType, PredictFeatureType, Fit>(
        std::move(model_), std::move(fit_), features);
  }

  template <typename PredictFeatureType>
  auto predict_with_measurement_noise(
      const std::vector<PredictFeatureType> &features) const {
    std::vector<Measurement<PredictFeatureType>> measurements;
    for (const auto &f : features) {
      measurements.emplace_back(Measurement<PredictFeatureType>(f));
    }
    return predict(measurements);
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
