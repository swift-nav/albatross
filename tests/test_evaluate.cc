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

#include <gtest/gtest.h>

#include "test_utils.h"
#include <albatross/Evaluation>

namespace albatross {

/* Make sure the multivariate negative log likelihood
 * matches python.
 *
 * import numpy as np
 * from scipy import stats
 *
 * x = np.array([-1, 0., 1])
 * cov = np.array([[1., 0.9, 0.8],
 *                 [0.9, 1., 0.9],
 *                 [0.8, 0.9, 1.]])
 * stats.multivariate_normal.logpdf(x, np.zeros(x.size), cov)
 * -6.0946974293510134
 *
 */
TEST(test_evaluate, test_negative_log_likelihood) {
  Eigen::VectorXd x(3);
  x << -1., 0., 1.;
  Eigen::MatrixXd cov(3, 3);
  cov << 1., 0.9, 0.8, 0.9, 1., 0.9, 0.8, 0.9, 1.;

  const auto nll = negative_log_likelihood(x, cov);
  EXPECT_NEAR(nll, 6.0946974293510134, 1e-6);

  const auto ldlt_nll = negative_log_likelihood(x, cov.ldlt());
  EXPECT_NEAR(nll, ldlt_nll, 1e-6);

  const DiagonalMatrixXd diagonal_matrix = cov.diagonal().asDiagonal();
  const Eigen::MatrixXd dense_diagonal = diagonal_matrix.toDenseMatrix();
  const auto diag_nll = negative_log_likelihood(x, diagonal_matrix);
  const auto dense_diag_nll = negative_log_likelihood(x, dense_diagonal);
  EXPECT_NEAR(diag_nll, dense_diag_nll, 1e-6);

  JointDistribution pred(x, cov);
  MarginalDistribution truth(Eigen::VectorXd::Zero(x.size()));

  const NegativeLogLikelihood<JointDistribution> joint_nll;
  const auto joint_nll_value = joint_nll(pred, truth);
  EXPECT_NEAR(joint_nll_value, nll, 1e-6);

  const NegativeLogLikelihood<MarginalDistribution> marginal_nll;
  MarginalDistribution marginal_pred(pred.mean,
                                     pred.covariance.diagonal().asDiagonal());
  const auto marginal_nll_value = marginal_nll(marginal_pred, truth);
  EXPECT_NEAR(marginal_nll_value, dense_diag_nll, 1e-6);
}

} // namespace albatross
