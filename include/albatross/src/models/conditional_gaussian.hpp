/*
 * Copyright (C) 2021 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef ALBATROSS_SRC_MODELS_CONDITIONAL_GAUSSIAN_HPP_
#define ALBATROSS_SRC_MODELS_CONDITIONAL_GAUSSIAN_HPP_

namespace albatross {

struct ConditionalFit {
  Eigen::VectorXd information;
  Eigen::LDLT<Eigen::MatrixXd> cov_ldlt;
  GroupIndices indices;
};

class ConditionalGaussian : public ModelBase<ConditionalGaussian> {
public:
  ConditionalGaussian(const JointDistribution &prior,
                      const MarginalDistribution &truth)
      : prior_(prior), truth_(truth) {}

  ConditionalFit fit_from_indices(const GroupIndices &indices) const {
    const JointDistribution train_prior = prior_.subset(indices);
    const MarginalDistribution train_truth = truth_.subset(indices);

    const Eigen::VectorXd truth_minus_prior =
        train_truth.mean - train_prior.mean;
    Eigen::MatrixXd cov = train_prior.covariance;
    cov += train_truth.covariance;

    const Eigen::LDLT<Eigen::MatrixXd> train_ldlt = cov.ldlt();
    const Eigen::VectorXd information = train_ldlt.solve(truth_minus_prior);

    const ConditionalFit fit = {information, train_ldlt, indices};
    return fit;
  }

  FitModel<ConditionalGaussian, ConditionalFit>
  fit(const std::vector<std::size_t> &indices) const {
    return FitModel<ConditionalGaussian, ConditionalFit>(
        *this, fit_from_indices(indices));
  }

  JointDistribution get_prior(const GroupIndices &indices) const {
    return prior_.subset(indices);
  }

  MarginalDistribution get_truth(const GroupIndices &indices) const {
    return truth_.subset(indices);
  }

  JointDistribution
  _predict_impl(const GroupIndices &predict_indices, const ConditionalFit &fit,
                PredictTypeIdentity<JointDistribution> &&) const {
    const JointDistribution predict_prior = prior_.subset(predict_indices);

    const Eigen::MatrixXd cross =
        subset(prior_.covariance, fit.indices, predict_indices);

    JointDistribution conditional_pred = gp_joint_prediction(
        cross, predict_prior.covariance, fit.information, fit.cov_ldlt);

    conditional_pred.mean += predict_prior.mean;
    return conditional_pred;
  }

  MarginalDistribution
  _predict_impl(const GroupIndices &predict_indices, const ConditionalFit &fit,
                PredictTypeIdentity<MarginalDistribution> &&) const {
    const MarginalDistribution predict_prior =
        prior_.marginal().subset(predict_indices);
    const Eigen::MatrixXd cross =
        subset(prior_.covariance, fit.indices, predict_indices);
    MarginalDistribution conditional_pred =
        gp_marginal_prediction(cross, predict_prior.covariance.diagonal(),
                               fit.information, fit.cov_ldlt);

    conditional_pred.mean += predict_prior.mean;
    return conditional_pred;
  }

  Eigen::VectorXd _predict_impl(const GroupIndices &predict_indices,
                                const ConditionalFit &fit,
                                PredictTypeIdentity<Eigen::VectorXd> &&) const {
    const Eigen::MatrixXd cross =
        subset(prior_.covariance, fit.indices, predict_indices);
    Eigen::VectorXd conditional_pred =
        gp_mean_prediction(cross, fit.information);
    conditional_pred += subset(prior_.mean, predict_indices);
    return conditional_pred;
  }

private:
  JointDistribution prior_;
  MarginalDistribution truth_;
};

} // namespace albatross

#endif /* ALBATROSS_SRC_MODELS_CONDITIONAL_GAUSSIAN_HPP_ */
