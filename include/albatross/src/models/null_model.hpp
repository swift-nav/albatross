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

#ifndef ALBATROSS_SRC_MODELS_NULL_MODEL_HPP_
#define ALBATROSS_SRC_MODELS_NULL_MODEL_HPP_

namespace albatross {

class NullModel;

template <> struct Fit<NullModel> {
  template <typename Archive>
  void serialize(Archive &archive, const std::uint32_t){};

  bool operator==(const Fit<NullModel> &other) const { return true; }
};

class NullModel : public ModelBase<NullModel> {

public:
  NullModel(){};
  NullModel(const ParameterStore &param_store) : params_(param_store){};

  std::string get_name() const { return "null_model"; };

  /*
   * The Gaussian Process Regression model derives its parameters from
   * the covariance functions.
   */
  ParameterStore get_params() const override { return params_; }

  void unchecked_set_param(const std::string &name,
                           const Parameter &param) override {
    params_[name] = param;
  }

  // If the implementing class doesn't have a fit method for this
  // FeatureType but the CovarianceFunction does.
  template <typename FeatureType>
  Fit<NullModel> _fit_impl(const std::vector<FeatureType> &features,
                           const MarginalDistribution &targets) const {
    return {};
  }

  template <typename FeatureType>
  auto fit_from_prediction(const std::vector<FeatureType> &features,
                           const JointDistribution &prediction) const {
    const NullModel m(*this);
    FitModel<NullModel, Fit<NullModel>> fit_model(m, Fit<NullModel>());
    return fit_model;
  }

  template <typename FeatureType>
  JointDistribution
  _predict_impl(const std::vector<FeatureType> &features,
                const Fit<NullModel> &fit,
                PredictTypeIdentity<JointDistribution> &&) const {
    const Eigen::Index n = static_cast<Eigen::Index>(features.size());
    const Eigen::VectorXd mean = Eigen::VectorXd::Zero(n);
    const Eigen::MatrixXd cov = 1.e4 * Eigen::MatrixXd::Identity(n, n);
    return JointDistribution(mean, cov);
  }

  template <typename FeatureType>
  MarginalDistribution
  _predict_impl(const std::vector<FeatureType> &features,
                const Fit<NullModel> &fit,
                PredictTypeIdentity<MarginalDistribution> &&) const {
    const Eigen::Index en = static_cast<Eigen::Index>(features.size());
    const Eigen::VectorXd mean = Eigen::VectorXd::Zero(en);
    const Eigen::VectorXd diag = 1.e4 * Eigen::VectorXd::Ones(en);
    return MarginalDistribution(mean, diag.asDiagonal());
  }

private:
  ParameterStore params_;
};

} // namespace albatross

#endif /* THIRD_PARTY_ALBATROSS_INCLUDE_ALBATROSS_SRC_MODELS_NULL_MODEL_HPP_   \
        */
