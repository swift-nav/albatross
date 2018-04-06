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

template <typename Feature>
struct GaussianProcessFit {
  std::vector<Feature> train_features;
  Eigen::VectorXd information;
  Eigen::LDLT<Eigen::MatrixXd> ldlt;
};

template <typename CovarianceFunction, typename Feature>
class GaussianProcessRegression : public RegressionModel<Feature, GaussianProcessFit<Feature>> {
 public:
  GaussianProcessRegression(CovarianceFunction& covariance_function)
      : covariance_function_(covariance_function) {};
  ~GaussianProcessRegression(){};

  std::string get_name() const override {
    return "gaussian_process_regression";
  };

  GaussianProcessFit<Feature> fit_(const std::vector<Feature>& features,
                                   const Eigen::VectorXd& targets) const override {
    GaussianProcessFit<Feature> model_fit;
    Eigen::MatrixXd cov = symmetric_covariance(covariance_function_, features);
    // Precompute the information vector which is all we need in
    // order to make predictions.
    model_fit.train_features = features;
    model_fit.information = cov.ldlt().solve(targets);
    model_fit.ldlt = cov.ldlt();
    return model_fit;
  }

  PredictionDistribution predict_(
      const std::vector<Feature>& features) const override {

    const auto cross_cov = asymmetric_covariance(covariance_function_,
                                                 features,
                                                 this->model_fit_->train_features);
    // Then we can use the information vector to determine the posterior
    const Eigen::VectorXd pred = cross_cov * this->model_fit_->information;

    Eigen::MatrixXd pred_cov = symmetric_covariance(covariance_function_, features);
    pred_cov -= cross_cov * this->model_fit_->ldlt.solve(cross_cov.transpose());

    return PredictionDistribution(pred, pred_cov);
  }

  template <typename OtherFeature>
  PredictionDistribution inspect(
      const std::vector<OtherFeature>& features) const {
    assert(this->model_fit_);
    const auto cross_cov = asymmetric_covariance(covariance_function_,
                                                 features,
                                                 this->model_fit_->train_features);
    // Then we can use the information vector to determine the posterior
    const Eigen::VectorXd pred = cross_cov * this->model_fit_->information;
    Eigen::MatrixXd pred_cov = symmetric_covariance(covariance_function_, features);
    pred_cov -= cross_cov * this->model_fit_->ldlt.solve(cross_cov.transpose());
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
};

template <typename CovFunc>
GaussianProcessRegression<CovFunc, double> gp_from_covariance(
    CovFunc covariance_function) {
  return GaussianProcessRegression<CovFunc, double>(covariance_function);
};

}

#endif
