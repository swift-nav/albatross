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

#ifndef ALBATROSS_CORE_PREDICTION_H
#define ALBATROSS_CORE_PREDICTION_H

namespace albatross {

// This is effectively just a container that allows us to develop methods
// which behave different conditional on the type of predictions desired.
template <typename T>
struct PredictTypeIdentity {
  typedef T type;
};

/*
 * MeanPredictor is responsible for determining if a valid form of
 * predicting exists for a given set of model, feature, fit.  The
 * primary goal of the class is to consolidate all the logic required
 * to decide if different predict types are available.  For example,
 * by inspecting this class for a _mean method you can determine if
 * any valid mean prediction method exists.
 */
class MeanPredictor {
 public:
  template <typename ModelType, typename FeatureType, typename FitType,
            typename std::enable_if<
                has_valid_predict_mean<ModelType, FeatureType, FitType>::value,
                int>::type = 0>
  Eigen::VectorXd _mean(const ModelType &model, const FitType &fit,
                        const std::vector<FeatureType> &features) const {
    return model.predict_(features, fit,
                          PredictTypeIdentity<Eigen::VectorXd>());
  }

  template <
      typename ModelType, typename FeatureType, typename FitType,
      typename std::enable_if<
          !has_valid_predict_mean<ModelType, FeatureType, FitType>::value &&
              has_valid_predict_marginal<ModelType, FeatureType,
                                         FitType>::value,
          int>::type = 0>
  Eigen::VectorXd _mean(const ModelType &model, const FitType &fit,
                        const std::vector<FeatureType> &features) const {
    return model
        .predict_(features, fit, PredictTypeIdentity<MarginalDistribution>())
        .mean;
  }

  template <
      typename ModelType, typename FeatureType, typename FitType,
      typename std::enable_if<
          !has_valid_predict_mean<ModelType, FeatureType, FitType>::value &&
              !has_valid_predict_marginal<ModelType, FeatureType,
                                          FitType>::value &&
              has_valid_predict_joint<ModelType, FeatureType, FitType>::value,
          int>::type = 0>
  Eigen::VectorXd _mean(const ModelType &model, const FitType &fit,
                        const std::vector<FeatureType> &features) const {
    return model
        .predict_(features, fit, PredictTypeIdentity<JointDistribution>())
        .mean;
  }
};

class MarginalPredictor {
 public:
  template <typename ModelType, typename FeatureType, typename FitType,
            typename std::enable_if<has_valid_predict_marginal<
                                        ModelType, FeatureType, FitType>::value,
                                    int>::type = 0>
  MarginalDistribution _marginal(
      const ModelType &model, const FitType &fit,
      const std::vector<FeatureType> &features) const {
    return model.predict_(features, fit,
                          PredictTypeIdentity<MarginalDistribution>());
  }

  template <
      typename ModelType, typename FeatureType, typename FitType,
      typename std::enable_if<
          !has_valid_predict_marginal<ModelType, FeatureType, FitType>::value &&
              has_valid_predict_joint<ModelType, FeatureType, FitType>::value,
          int>::type = 0>
  MarginalDistribution _marginal(
      const ModelType &model, const FitType &fit,
      const std::vector<FeatureType> &features) const {
    const auto joint_pred =
        model.predict_(features, fit, PredictTypeIdentity<JointDistribution>());
    return joint_pred.marginal();
  }
};

class JointPredictor {
 public:
  template <typename ModelType, typename FeatureType, typename FitType,
            typename std::enable_if<
                has_valid_predict_joint<ModelType, FeatureType, FitType>::value,
                int>::type = 0>
  JointDistribution _joint(const ModelType &model, const FitType &fit,
                           const std::vector<FeatureType> &features) const {
    return model.predict_(features, fit,
                          PredictTypeIdentity<JointDistribution>());
  }
};

template <typename ModelType, typename FeatureType, typename FitType>
class Prediction {
  using PlainModelType = typename std::decay<ModelType>::type;
  using PlainFitType = typename std::decay<FitType>::type;

 public:
  Prediction(const PlainModelType &model, const PlainFitType &fit,
             const std::vector<FeatureType> &features)
      : model_(model), fit_(fit), features_(features) {}

  Prediction(PlainModelType &&model, PlainFitType &&fit,
             const std::vector<FeatureType> &features)
      : model_(std::move(model)), fit_(std::move(fit)), features_(features) {}

  // Mean
  template <typename DummyType = FeatureType,
            typename std::enable_if<can_predict_mean<MeanPredictor, ModelType,
                                                     DummyType, FitType>::value,
                                    int>::type = 0>
  Eigen::VectorXd mean() const {
    static_assert(std::is_same<DummyType, FeatureType>::value,
                  "never do prediction.mean<T>()");
    if (features_.size() == 0) {
      return Eigen::VectorXd(0);
    }
    return MeanPredictor()._mean(model_, fit_, features_);
  }

  template <
      typename DummyType = FeatureType,
      typename std::enable_if<!can_predict_mean<MeanPredictor, ModelType,
                                                DummyType, FitType>::value,
                              int>::type = 0>
  Eigen::VectorXd mean() const = delete;  // No valid predict method found.

  // Marginal
  template <
      typename DummyType = FeatureType,
      typename std::enable_if<can_predict_marginal<MarginalPredictor, ModelType,
                                                   DummyType, FitType>::value,
                              int>::type = 0>
  MarginalDistribution marginal() const {
    static_assert(std::is_same<DummyType, FeatureType>::value,
                  "never do prediction.mean<T>()");
    if (features_.size() == 0) {
      return MarginalDistribution();
    }
    return MarginalPredictor()._marginal(model_, fit_, features_);
  }

  template <typename DummyType = FeatureType,
            typename std::enable_if<
                !can_predict_marginal<MarginalPredictor, ModelType, DummyType,
                                      FitType>::value,
                int>::type = 0>
  MarginalDistribution marginal() const =
      delete;  // No valid predict method found.

  // Joint
  template <
      typename DummyType = FeatureType,
      typename std::enable_if<can_predict_joint<JointPredictor, ModelType,
                                                DummyType, FitType>::value,
                              int>::type = 0>
  JointDistribution joint() const {
    static_assert(std::is_same<DummyType, FeatureType>::value,
                  "never do prediction.mean<T>()");
    if (features_.size() == 0) {
      return JointDistribution();
    }
    return JointPredictor()._joint(model_, fit_, features_);
  }

  template <
      typename DummyType = FeatureType,
      typename std::enable_if<!can_predict_joint<JointPredictor, ModelType,
                                                 DummyType, FitType>::value,
                              int>::type = 0>
  JointDistribution joint() const = delete;  // No valid predict method found.

  template <typename PredictType>
  PredictType get(PredictTypeIdentity<PredictType> =
                      PredictTypeIdentity<PredictType>()) const {
    return get(get_type<PredictType>());
  }

  std::size_t size() const { return features_.size(); }

  explicit operator JointDistribution() const { return this->joint(); };

  explicit operator MarginalDistribution() const { return this->marginal(); };

 private:
  template <typename T>
  struct get_type {};

  Eigen::VectorXd get(get_type<Eigen::VectorXd>) const { return this->mean(); }

  MarginalDistribution get(get_type<MarginalDistribution>) const {
    return this->marginal();
  }

  JointDistribution get(get_type<JointDistribution>) const {
    return this->joint();
  }

  const ModelType model_;
  const FitType fit_;
  const std::vector<FeatureType> features_;
};

template <typename ModelType, typename FeatureType, typename FitType>
auto get_prediction(const ModelType &model, const FitType &fit,
                    const std::vector<FeatureType> &features) {
  return Prediction<ModelType, FeatureType, FitType>(model, fit, features);
}

template <typename ModelType, typename FeatureType, typename FitType>
auto get_prediction_reference(const ModelType &model, const FitType &fit,
                              const std::vector<FeatureType> &features) {
  return PredictionReference<ModelType, FeatureType, FitType>(model, fit,
                                                              features);
}

}  // namespace albatross
#endif
