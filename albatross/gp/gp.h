

#ifndef ALBATROSS_GP_GP_H
#define ALBATROSS_GP_GP_H

#include <functional>
#include <memory>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include "stdio.h"
#include "core/model.h"


namespace albatross {

template <typename CovarianceFunction, typename Predictor>
class GaussianProcessRegression : public RegressionModel<Predictor> {
 public:
  GaussianProcessRegression(CovarianceFunction& covariance_function)
      : covariance_function_(covariance_function),
        train_predictors_(),
        ldlt_(),
        information_(){};
  ~GaussianProcessRegression(){};

  std::string get_name() const override {
    return "gaussian_process_regression";
  };

  void fit_(const std::vector<Predictor>& predictors,
            const Eigen::VectorXd& targets) override {
    train_predictors_ = predictors;
    Eigen::MatrixXd cov = symmetric_covariance(covariance_function_, train_predictors_);
    // Precompute the information vector which is all we need in
    // order to make predictions.
    ldlt_ = cov.ldlt();
    information_ = ldlt_.solve(targets);
  }

  PredictionDistribution predict_(
      const std::vector<Predictor>& predictors) const override {

    const auto cross_cov = asymmetric_covariance(covariance_function_,
                                                 predictors, train_predictors_);

    // Then we can use the information vector to determine the posterior
    const Eigen::VectorXd pred = cross_cov * information_;

    Eigen::MatrixXd pred_cov = symmetric_covariance(covariance_function_, predictors);
    pred_cov -= cross_cov * ldlt_.solve(cross_cov.transpose());

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
  std::vector<Predictor> train_predictors_;
  Eigen::LDLT<Eigen::MatrixXd> ldlt_;
  Eigen::VectorXd information_;
};

}

#endif
