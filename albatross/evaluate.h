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

#include "core/model.h"
#include "crossvalidation.h"
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <functional>
#include <map>
#include <math.h>
#include <memory>

namespace albatross {

/*
 * Computes the negative log likelihood under the assumption that the predcitve
 * distribution is multivariate normal.
 */
static inline double
negative_log_likelihood(const Eigen::VectorXd &mean,
                        const Eigen::MatrixXd &covariance) {
  auto llt = covariance.llt();
  auto cholesky = llt.matrixL();
  double det = cholesky.determinant();
  double log_det = log(det);
  Eigen::VectorXd normalized_residuals(mean.size());
  normalized_residuals = cholesky.solve(mean);
  double residuals = normalized_residuals.dot(normalized_residuals);
  return 0.5 *
         (log_det + residuals + static_cast<double>(mean.size()) * 2 * M_PI);
}

namespace evaluation_metrics {

static inline double
root_mean_square_error(const PredictDistribution &prediction,
                       const TargetDistribution &truth) {
  const Eigen::VectorXd error = prediction.mean - truth.mean;
  double mse = error.dot(error) / static_cast<double>(error.size());
  return sqrt(mse);
}

/*
 * Takes output from a model (PredictionDistribution)
 * and the corresponding truth and uses them to compute the stddev.
 */
static inline double standard_deviation(const PredictDistribution &prediction,
                                        const TargetDistribution &truth) {
  Eigen::VectorXd error = prediction.mean - truth.mean;
  const auto n_elements = static_cast<double>(error.size());
  const double mean_error = error.sum() / n_elements;
  error.array() -= mean_error;
  return std::sqrt(error.dot(error) / (n_elements - 1));
}

/*
 * Computes the negative log likelihood under the assumption that the predictive
 * distribution is multivariate normal.
 */
static inline double
negative_log_likelihood(const PredictDistribution &prediction,
                        const TargetDistribution &truth) {
  const Eigen::VectorXd mean = prediction.mean - truth.mean;
  Eigen::MatrixXd covariance(prediction.covariance);
  if (truth.has_covariance()) {
    covariance += truth.covariance;
  }
  return albatross::negative_log_likelihood(mean, covariance);
}

} // namespace evaluation_metrics
} // namespace albatross

#endif
