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
template <typename T> struct PredictTypeIdentity { typedef T type; };

template <typename ModelType, typename FeatureType, typename FitType>
class Prediction {

public:
  Prediction(const ModelType &model, const FitType &fit,
             const std::vector<FeatureType> &features)
      : model_(model), fit_(fit), features_(features) {}

  Prediction(const ModelType &model, const FitType &fit,
             std::vector<FeatureType> &&features)
      : model_(model), fit_(fit), features_(std::move(features)) {}

  // Mean
  template <typename DummyType = FeatureType,
            typename std::enable_if<
                has_valid_predict_mean<ModelType, DummyType, FitType>::value,
                int>::type = 0>
  Eigen::VectorXd mean() const {
    static_assert(std::is_same<DummyType, FeatureType>::value,
                  "never do prediction.mean<T>()");
    return model_.predict_(features_, fit_,
                           PredictTypeIdentity<Eigen::VectorXd>());
  }

  template <
      typename DummyType = FeatureType,
      typename std::enable_if<
          !has_valid_predict_mean<ModelType, DummyType, FitType>::value &&
              has_valid_predict_marginal<ModelType, DummyType, FitType>::value,
          int>::type = 0>
  Eigen::VectorXd mean() const {
    static_assert(std::is_same<DummyType, FeatureType>::value,
                  "never do prediction.mean<T>()");
    return model_
        .predict_(features_, fit_, PredictTypeIdentity<MarginalDistribution>())
        .mean;
  }

  template <
      typename DummyType = FeatureType,
      typename std::enable_if<
          !has_valid_predict_mean<ModelType, DummyType, FitType>::value &&
              !has_valid_predict_marginal<ModelType, DummyType,
                                          FitType>::value &&
              has_valid_predict_joint<ModelType, DummyType, FitType>::value,
          int>::type = 0>
  Eigen::VectorXd mean() const {
    static_assert(std::is_same<DummyType, FeatureType>::value,
                  "never do prediction.mean<T>()");
    return model_
        .predict_(features_, fit_, PredictTypeIdentity<JointDistribution>())
        .mean;
  }

  // Marginal
  template <typename DummyType = FeatureType,
            typename std::enable_if<has_valid_predict_marginal<
                                        ModelType, DummyType, FitType>::value,
                                    int>::type = 0>
  MarginalDistribution marginal() const {
    static_assert(std::is_same<DummyType, FeatureType>::value,
                  "never do prediction.marginal<T>()");
    return model_.predict_(features_, fit_,
                           PredictTypeIdentity<MarginalDistribution>());
  }

  template <
      typename DummyType = FeatureType,
      typename std::enable_if<
          !has_valid_predict_marginal<ModelType, DummyType, FitType>::value &&
              has_valid_predict_joint<ModelType, DummyType, FitType>::value,
          int>::type = 0>
  MarginalDistribution marginal() const {
    static_assert(std::is_same<DummyType, FeatureType>::value,
                  "never do prediction.marginal<T>()");
    const auto joint_pred = model_.predict_(
        features_, fit_, PredictTypeIdentity<JointDistribution>());
    if (joint_pred.has_covariance()) {
      Eigen::VectorXd diag = joint_pred.covariance.diagonal();
      return MarginalDistribution(joint_pred.mean, diag.asDiagonal());
    } else {
      return MarginalDistribution(joint_pred.mean);
    }
  }

  // Joint
  template <typename DummyType = FeatureType,
            typename std::enable_if<
                has_valid_predict_joint<ModelType, DummyType, FitType>::value,
                int>::type = 0>
  JointDistribution joint() const {
    static_assert(std::is_same<DummyType, FeatureType>::value,
                  "never do prediction.joint<T>()");
    return model_.predict_(features_, fit_,
                           PredictTypeIdentity<JointDistribution>());
  }

  // CATCH FAILURE MODES
  template <
      typename DummyType = FeatureType,
      typename std::enable_if<
          !has_valid_predict_mean<ModelType, DummyType, FitType>::value &&
              !has_valid_predict_marginal<ModelType, DummyType,
                                          FitType>::value &&
              !has_valid_predict_joint<ModelType, DummyType, FitType>::value,
          int>::type = 0>
  Eigen::VectorXd mean() const = delete; // No valid predict method found.

  template <
      typename DummyType = FeatureType,
      typename std::enable_if<
          !has_valid_predict_marginal<ModelType, DummyType, FitType>::value &&
              !has_valid_predict_joint<ModelType, DummyType, FitType>::value,
          int>::type = 0>
  Eigen::VectorXd
  marginal() const = delete; // No valid predict marginal method found.

  template <typename DummyType = FeatureType,
            typename std::enable_if<
                !has_valid_predict_joint<ModelType, DummyType, FitType>::value,
                int>::type = 0>
  Eigen::VectorXd
  joint() const = delete; // No valid predict joint method found.

  template <typename PredictType>
  PredictType get(PredictTypeIdentity<PredictType> =
                      PredictTypeIdentity<PredictType>()) const {
    return get(get_type<PredictType>());
  }

  std::size_t size() const { return features_.size(); }

private:
  template <typename T> struct get_type {};

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
}
#endif
