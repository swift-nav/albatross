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

#ifndef ALBATROSS_GP_GP_H
#define ALBATROSS_GP_GP_H

#include <functional>
#include <memory>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include "stdio.h"
#include "core/model.h"

namespace albatross {

template <typename CovarianceFunction, typename Feature>
class GaussianProcessRegression : public RegressionModel<Feature> {
 public:
  GaussianProcessRegression(CovarianceFunction& covariance_function)
      : covariance_function_(covariance_function),
        train_features_(),
        ldlt_(),
        information_(){};
  ~GaussianProcessRegression(){};

  std::string get_name() const override {
    return "gaussian_process_regression";
  };

  void fit_(const std::vector<Feature>& features,
            const Eigen::VectorXd& targets) override {
    train_features_ = features;
    Eigen::MatrixXd cov = symmetric_covariance(covariance_function_, train_features_);
    // Precompute the information vector which is all we need in
    // order to make predictions.
    ldlt_ = cov.ldlt();
    information_ = ldlt_.solve(targets);
  }

  PredictionDistribution predict_(
      const std::vector<Feature>& features) const override {

    const auto cross_cov = asymmetric_covariance(covariance_function_,
                                                 features, train_features_);

    // Then we can use the information vector to determine the posterior
    const Eigen::VectorXd pred = cross_cov * information_;

    Eigen::MatrixXd pred_cov = symmetric_covariance(covariance_function_, features);
    pred_cov -= cross_cov * ldlt_.solve(cross_cov.transpose());

    return PredictionDistribution(pred, pred_cov);
  }

  template <typename OtherFeature>
  PredictionDistribution inspect(
      const std::vector<OtherFeature>& features) const {
    assert(this->has_been_fit_);
    const auto cross_cov = asymmetric_covariance(covariance_function_,
                                                 features, train_features_);
    // Then we can use the information vector to determine the posterior
    const Eigen::VectorXd pred = cross_cov * information_;
    Eigen::MatrixXd pred_cov = symmetric_covariance(covariance_function_, features);
    pred_cov -= cross_cov * ldlt_.solve(cross_cov.transpose());
    assert(static_cast<s32>(pred.size()) ==
           static_cast<s32>(features.size()));
    return PredictionDistribution(pred, pred_cov);
  }

  ParameterStore get_params() const override {
    return covariance_function_.get_params();
  }

  void unchecked_set_param(const std::string& name,
                           const double value) override {
    covariance_function_.set_param(name, value);
  }

 private:
  CovarianceFunction covariance_function_;
  std::vector<Feature> train_features_;
  Eigen::LDLT<Eigen::MatrixXd> ldlt_;
  Eigen::VectorXd information_;
};

template <typename CovFunc>
GaussianProcessRegression<CovFunc, double> gp_from_covariance(
    CovFunc covariance_function) {
  return GaussianProcessRegression<CovFunc, double>(covariance_function);
};

}

#endif
