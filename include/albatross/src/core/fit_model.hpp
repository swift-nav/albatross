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

template <typename ModelType, typename Fit>
class FitModel {
 public:
  template <typename X, typename Y, typename Z>
  friend class Prediction;

  static_assert(
      std::is_move_constructible<Fit>::value,
      "Fit type must be move constructible to avoid unexpected copying.");

  typedef ModelType model_type;
  typedef Fit fit_type;

  FitModel() {}

  FitModel(const ModelType &model, const Fit &fit) = delete;

  FitModel(const ModelType &model, Fit &&fit)
      : model_(model), fit_(std::move(fit)) {}

  // When FitModel is an lvalue we store a reference to the fit
  // inside the resulting Prediction class.
  template <typename PredictFeatureType>
  const PredictionReference<ModelType, PredictFeatureType, Fit> predict(
      const std::vector<PredictFeatureType> &features) const & {
    return PredictionReference<ModelType, PredictFeatureType, Fit>(model_, fit_,
                                                                   features);
  }

  // When FitModel is an rvalue the Fit will be a temporary so
  // we move it into the Prediction class to be stored there.
  template <typename PredictFeatureType>
  Prediction<ModelType, PredictFeatureType, Fit> predict(
      const std::vector<PredictFeatureType> &features) && {
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

  template <
      typename FeatureType,
      typename std::enable_if<
          has_valid_update<ModelType, Fit, FeatureType>::value, int>::type = 0>
  auto update(const std::vector<FeatureType> &features,
              const MarginalDistribution &targets) const {
    auto updated_fit = model_._update_impl(Fit(fit_), features, targets);
    return FitModel<ModelType, decltype(updated_fit)>(model_,
                                                      std::move(updated_fit));
  }

  template <
      typename FeatureType,
      typename std::enable_if<
          has_valid_update<ModelType, Fit, FeatureType>::value, int>::type = 0>
  auto update(const RegressionDataset<FeatureType> &dataset) const {
    return update(dataset.features, dataset.targets);
  }

  template <typename FeatureType,
            typename std::enable_if<
                can_update_in_place<ModelType, Fit, FeatureType>::value,
                int>::type = 0>
  void update_in_place(const std::vector<FeatureType> &features,
                       const MarginalDistribution &targets) {
    fit_ = model_._update_impl(fit_, features, targets);
  }

  template <typename FeatureType>
  void update_in_place(const RegressionDataset<FeatureType> &dataset) {
    update_in_place(dataset.features, dataset.targets);
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

template <typename ModelType, typename FitType, typename FeatureType>
auto update(const FitModel<ModelType, FitType> &fit_model,
            const RegressionDataset<FeatureType> &dataset) {
  return fit_model.update(dataset);
}

}  // namespace albatross
#endif
