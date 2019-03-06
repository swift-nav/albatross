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

#ifndef ALBATROSS_CORE_PREDICTION_H
#define ALBATROSS_CORE_PREDICTION_H

namespace albatross {

// This is effectively just a container that allows us to develop methods
// which behave different conditional on the type of predictions desired.
template <typename T> struct PredictTypeIdentity { typedef T type; };

template <typename ModelType, typename FeatureType, typename FitType>
class Prediction {

public:
  Prediction(const FitModel<ModelType, FitType> &fit_model,
             const std::vector<FeatureType> &features)
      : fit_model_(fit_model), features_(features) {}

  /*
   * MEAN
   */
  template <
      typename DummyType = FeatureType,
      typename std::enable_if<
          has_valid_predict_mean<ModelType, DummyType, FitType>::value, int>::type = 0>
  Eigen::VectorXd mean() const {
    static_assert(std::is_same<DummyType, FeatureType>::value,
                  "never do prediction.mean<T>()");
    return fit_model_.model_.predict_(features_, fit_model_.fit_, PredictTypeIdentity<Eigen::VectorXd>());
  }

  template <typename DummyType = FeatureType,
            typename std::enable_if<
                !has_valid_predict_mean<ModelType, DummyType, FitType>::value &&
                 has_valid_predict_marginal<ModelType, DummyType, FitType>::value,
                int>::type = 0>
  Eigen::VectorXd mean() const {
    static_assert(std::is_same<DummyType, FeatureType>::value,
                  "never do prediction.mean<T>()");
    return fit_model_.model_.predict_(features_, fit_model_.fit_, PredictTypeIdentity<MarginalDistribution>())
        .mean;
  }

  template <typename DummyType = FeatureType,
            typename std::enable_if<
                !has_valid_predict_mean<ModelType, DummyType, FitType>::value &&
                    !has_valid_predict_marginal<ModelType, DummyType, FitType>::value &&
                    has_valid_predict_joint<ModelType, DummyType, FitType>::value,
                int>::type = 0>
  Eigen::VectorXd mean() const {
    static_assert(std::is_same<DummyType, FeatureType>::value,
                  "never do prediction.mean<T>()");
    return fit_model_.model_.predict_(features_, fit_model_.fit_, PredictTypeIdentity<JointDistribution>())
        .mean;
  }

  /*
   * MARGINAL
   */
  template <typename DummyType = FeatureType,
            typename std::enable_if<
                has_valid_predict_marginal<ModelType, DummyType, FitType>::value,
                int>::type = 0>
  MarginalDistribution marginal() const {
    static_assert(std::is_same<DummyType, FeatureType>::value,
                  "never do prediction.marginal<T>()");
    return fit_model_.model_.predict_(features_,
                                      fit_model_.fit_,
                           PredictTypeIdentity<MarginalDistribution>());
  }

  template <typename DummyType = FeatureType,
            typename std::enable_if<
                !has_valid_predict_marginal<ModelType, DummyType, FitType>::value &&
                    has_valid_predict_joint<ModelType, DummyType, FitType>::value,
                int>::type = 0>
  MarginalDistribution marginal() const {
    static_assert(std::is_same<DummyType, FeatureType>::value,
                  "never do prediction.marginal<T>()");
    const auto joint_pred =
        fit_model_.model_.predict_(features_, fit_model_, PredictTypeIdentity<JointDistribution>());
    if (joint_pred.has_covariance()) {
      Eigen::VectorXd diag = joint_pred.covariance.diagonal();
      return MarginalDistribution(joint_pred.mean, diag.asDiagonal());
    } else {
      return MarginalDistribution(joint_pred.mean);
    }
  }

  /*
   * JOINT
   */
  template <
      typename DummyType = FeatureType,
      typename std::enable_if<
          has_valid_predict_joint<ModelType, DummyType, FitType>::value, int>::type = 0>
  JointDistribution joint() const {
    static_assert(std::is_same<DummyType, FeatureType>::value,
                  "never do prediction.joint<T>()");
    return fit_model_.model_.predict_(features_, fit_model_, PredictTypeIdentity<JointDistribution>());
  }

  /*
   * CATCH FAILURE MODES
   */
  template <typename DummyType = FeatureType,
            typename std::enable_if<
                !has_valid_predict_mean<ModelType, DummyType, FitType>::value &&
                    !has_valid_predict_marginal<ModelType, DummyType, FitType>::value &&
                    !has_valid_predict_joint<ModelType, DummyType, FitType>::value,
                int>::type = 0>
  Eigen::VectorXd mean() const = delete; // No valid predict method found.

  template <typename DummyType = FeatureType,
            typename std::enable_if<
                    !has_valid_predict_marginal<ModelType, DummyType, FitType>::value &&
                    !has_valid_predict_joint<ModelType, DummyType, FitType>::value,
                int>::type = 0>
  Eigen::VectorXd marginal() const = delete; // No valid predict marginal method found.

  template <typename DummyType = FeatureType,
            typename std::enable_if<
                    !has_valid_predict_joint<ModelType, DummyType, FitType>::value,
                int>::type = 0>
  Eigen::VectorXd joint() const = delete; // No valid predict joint method found.

private:
  const FitModel<ModelType, FitType> &fit_model_;
  const std::vector<FeatureType> &features_;
};

}
#endif
