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

#ifndef ALBATROSS_EVALUATION_LIKELIHOOD_H
#define ALBATROSS_EVALUATION_LIKELIHOOD_H

namespace albatross {

/*
 * Negative log likelihood of a univariate normal.
 */
static inline double negative_log_likelihood(double deviation,
                                             double variance) {
  return -albatross::gaussian::log_pdf(deviation, variance);
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
static inline double negative_log_likelihood(
    const Eigen::VectorXd &deviation,
    const Eigen::LDLT<_MatrixType, _UpLo> &ldlt) {
  const auto diag = ldlt.vectorD();
  const double rank = cast::to_double(diag.size());
  const double mahalanobis = deviation.dot(ldlt.solve(deviation));
  const double log_det = log_sum(diag);
  return 0.5 * (log_det + mahalanobis + rank * log(2 * M_PI));
}

/*
 * Computes the negative log likelihood under the assumption that the predcitve
 * distribution is multivariate normal.
 */
static inline double negative_log_likelihood(
    const Eigen::VectorXd &deviation, const Eigen::MatrixXd &covariance) {
  ALBATROSS_ASSERT(deviation.size() == covariance.rows());
  ALBATROSS_ASSERT(covariance.cols() == covariance.rows());
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
 * makes the computation a lot simpler since all variables are
 * independent.
 */
template <typename _Scalar, int SizeAtCompileTime>
static inline double negative_log_likelihood(
    const Eigen::VectorXd &deviation,
    const Eigen::DiagonalMatrix<_Scalar, SizeAtCompileTime>
        &diagonal_covariance) {
  ALBATROSS_ASSERT(deviation.size() == diagonal_covariance.diagonal().size());
  const auto variances = diagonal_covariance.diagonal();
  double nll = 0.;
  for (Eigen::Index i = 0; i < deviation.size(); i++) {
    nll += negative_log_likelihood(deviation[i], variances[i]);
  }
  return nll;
}

}  // namespace albatross

#endif
