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

#ifndef ALBATROSS_EVALUATE_H
#define ALBATROSS_EVALUATE_H

namespace albatross {

/*
 * An EvaluationMetric is basically just a wrapper around a
 * function which enforces a signature used in evaluation.  One
 * alternative would be to use std::function.  Ie,
 *
 *   template <typename RequiredPredictType>
 *   using EvaluationMetric =
 *         std::function<double(const RequiredPredictType &,
 *                              const MarginalDistribution &);
 *
 * the problem with this is that when you pass the std::function
 * object as an argument the RequiredPredictType cannot be
 * inferred.
 */
template <typename RequiredPredictType> struct EvaluationMetric {

  virtual double operator()(const RequiredPredictType &,
                            const MarginalDistribution &) const = 0;

  template <typename ModelType, typename FeatureType, typename FitType>
  double
  operator()(const Prediction<ModelType, FeatureType, FitType> &prediction,
             const MarginalDistribution &truth) const {
    return this->operator()(prediction.template get<RequiredPredictType>(),
                            truth);
  }
};

static inline double root_mean_square_error(const Eigen::VectorXd &prediction,
                                            const Eigen::VectorXd &truth) {
  const Eigen::VectorXd error = prediction - truth;
  double mse = error.dot(error) / static_cast<double>(error.size());
  return sqrt(mse);
}

struct RootMeanSquareError : public EvaluationMetric<Eigen::VectorXd> {
  double operator()(const Eigen::VectorXd &prediction,
                    const MarginalDistribution &truth) const override {
    return root_mean_square_error(prediction, truth.mean);
  }
};

static inline double standard_deviation(const Eigen::VectorXd &prediction,
                                        const Eigen::VectorXd &truth) {
  Eigen::VectorXd error = prediction - truth;
  const auto n_elements = static_cast<double>(error.size());
  const double mean_error = error.sum() / n_elements;
  error.array() -= mean_error;
  return std::sqrt(error.dot(error) / (n_elements - 1));
}

struct StandardDeviation : public EvaluationMetric<Eigen::VectorXd> {
  double operator()(const Eigen::VectorXd &prediction,
                    const MarginalDistribution &truth) const override {
    return standard_deviation(prediction, truth.mean);
  }
};

/*
 * Computes the negative log likelihood under the assumption that the
 * predictive
 * distribution is multivariate normal.
 */
static inline double
negative_log_likelihood(const JointDistribution &prediction,
                        const MarginalDistribution &truth) {
  const Eigen::VectorXd mean = prediction.mean - truth.mean;
  Eigen::MatrixXd covariance(prediction.covariance);
  if (truth.has_covariance()) {
    covariance += truth.covariance;
  }
  return albatross::negative_log_likelihood(mean, covariance);
}

static inline double
negative_log_likelihood(const MarginalDistribution &prediction,
                        const MarginalDistribution &truth) {
  const Eigen::VectorXd mean = prediction.mean - truth.mean;
  Eigen::VectorXd variance(prediction.covariance.diagonal());
  if (truth.has_covariance()) {
    variance += truth.covariance.diagonal();
  }
  return albatross::negative_log_likelihood(mean, variance.asDiagonal());
}

template <typename PredictType = JointDistribution>
struct NegativeLogLikelihood : public EvaluationMetric<PredictType> {
  double operator()(const PredictType &prediction,
                    const MarginalDistribution &truth) const override {
    return negative_log_likelihood(prediction, truth);
  }
};

} // namespace albatross

#endif
