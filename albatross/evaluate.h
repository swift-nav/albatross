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
 * Negative log likelihood of a univariate normal.
 */
static inline double negative_log_likelihood(double deviation,
                                             double variance) {
  double nll = deviation;
  nll *= nll;
  nll /= (2 * variance);
  nll += 0.5 * log(2 * M_PI * variance);
  return nll;
}

static inline double log_sum(const Eigen::VectorXd &x) {
  double sum = 0.;
  for (Eigen::Index i = 0; i < x.size(); i++) {
    sum += log(x[i]);
  }
  return sum;
}

/*
 * Negative log likelihood of a pre decomposed multivariate
 * normal.
 */
template <typename _MatrixType, int _UpLo>
static inline double
negative_log_likelihood(const Eigen::VectorXd &deviation,
                        const Eigen::LDLT<_MatrixType, _UpLo> &ldlt) {
  const auto diag = ldlt.vectorD();
  const auto L = ldlt.matrixL();
  const double rank = static_cast<double>(diag.size());
  const double mahalanobis = deviation.dot(ldlt.solve(deviation));
  const double log_det = log_sum(diag);
  return -0.5 * (log_det + mahalanobis + rank * log(2 * M_PI));
}

/*
 * Computes the negative log likelihood under the assumption that the predcitve
 * distribution is multivariate normal.
 */
static inline double
negative_log_likelihood(const Eigen::VectorXd &deviation,
                        const Eigen::MatrixXd &covariance) {
  assert(deviation.size() == covariance.rows());
  assert(covariance.cols() == covariance.rows());
  if (deviation.size() == 1) {
    // Looks like we have a univariate distribution, skipping
    // all the matrix decomposition steps should speed this up.
    return negative_log_likelihood(deviation[0], covariance(0, 0));
  } else {
    const auto ldlt = covariance.ldlt();
    return negative_log_likelihood(deviation, ldlt);
  }
}

/*
 * This handles the case where the covariance matrix is diagonal, which
 * means makes the computation a lot simpler since all variables are
 * independent.
 */
static inline double
negative_log_likelihood(const Eigen::VectorXd &deviation,
                        const DiagonalMatrixXd &diagonal_covariance) {
  const auto variances = diagonal_covariance.diagonal();
  double nll = 0.;
  for (Eigen::Index i = 0; i < deviation.size(); i++) {
    nll += negative_log_likelihood(deviation[i], variances[i]);
  }
  return nll;
}

/*
 * Evaluation metrics are best kept in a separate namespace since
 * the compiler can get confused with the use of std::function
 * (which is used in the definition of an EvaluationMetric) and
 * overloaded functions.
 *
 * https://stackoverflow.com/questions/30393285/stdfunction-fails-to-distinguish-overloaded-functions
 */
namespace evaluation_metrics {

static inline double root_mean_square_error(const JointDistribution &prediction,
                                            const MarginalDistribution &truth) {
  const Eigen::VectorXd error = prediction.mean - truth.mean;
  double mse = error.dot(error) / static_cast<double>(error.size());
  return sqrt(mse);
}

/*
 * Takes output from a model (PredictionDistribution)
 * and the corresponding truth and uses them to compute the stddev.
 */
static inline double standard_deviation(const JointDistribution &prediction,
                                        const MarginalDistribution &truth) {
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
negative_log_likelihood(const JointDistribution &prediction,
                        const MarginalDistribution &truth) {
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
