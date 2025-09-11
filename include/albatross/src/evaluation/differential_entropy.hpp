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

#ifndef INCLUDE_ALBATROSS_EVALUATION_DIFFERENTIAL_ENTROPY_H_
#define INCLUDE_ALBATROSS_EVALUATION_DIFFERENTIAL_ENTROPY_H_

namespace albatross {

// Calculate a numerically stable log determinant of a symmetric matrix using
// the Cholesky (LDLT) decomposition.
inline double log_determinant_of_symmetric(
    const Eigen::LDLT<Eigen::MatrixXd> &ldlt) {
  double log_determinant = 0;
  const auto diagonal = ldlt.vectorD();
  for (Eigen::Index i = 0; i < diagonal.size(); ++i) {
    log_determinant += log(diagonal(i));
  }
  return log_determinant;
}

/*
 * Loosely describes the entropy of a model given the
 * covariance matrix.  This can be thought of as describing
 * the dispersion of the data.
 *
 * https://en.wikipedia.org/wiki/Differential_entropy
 */
inline double differential_entropy(
    const Eigen::LDLT<Eigen::MatrixXd> &cov_ldlt) {
  double k = cast::to_double(cov_ldlt.rows());
  double log_det = log_determinant_of_symmetric(cov_ldlt);
  return 0.5 * (k * (1 + log(2 * M_PI) + log_det));
}

inline double differential_entropy(const Eigen::MatrixXd &cov) {
  Eigen::LDLT<Eigen::MatrixXd> ldlt(cov);
  return differential_entropy(ldlt);
}
}  // namespace albatross

#endif /* INCLUDE_ALBATROSS_EVALUATION_DIFFERENTIAL_ENTROPY_H_ */
