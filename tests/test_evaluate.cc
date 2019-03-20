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

  const auto nll = albatross::negative_log_likelihood(x, cov);
  EXPECT_NEAR(nll, 6.0946974293510134, 1e-6);

  const auto ldlt_nll = albatross::negative_log_likelihood(x, cov.ldlt());
  EXPECT_NEAR(nll, ldlt_nll, 1e-6);

  const DiagonalMatrixXd diagonal_matrix = cov.diagonal().asDiagonal();
  const Eigen::MatrixXd dense_diagonal = diagonal_matrix.toDenseMatrix();
  const auto diag_nll = albatross::negative_log_likelihood(x, diagonal_matrix);
  const auto dense_diag_nll =
      albatross::negative_log_likelihood(x, dense_diagonal);
  EXPECT_NEAR(diag_nll, dense_diag_nll, 1e-6);

  JointDistribution pred(x, dense_diagonal);
  MarginalDistribution truth(Eigen::VectorXd::Zero(x.size()));

  const auto dist_nll =
      evaluation_metrics::negative_log_likelihood(pred, truth);
  EXPECT_NEAR(dist_nll, dense_diag_nll, 1e-6);
}

TEST_F(LinearRegressionTest, test_leave_one_out) {
  model_ptr_->fit(dataset_);
  Eigen::VectorXd preds =
      model_ptr_->predict<Eigen::VectorXd>(dataset_.features);
  double in_sample_rmse = root_mean_square_error(preds, dataset_.targets);
  const auto folds = leave_one_out(dataset_);

  EvaluationMetric<Eigen::VectorXd> rmse = root_mean_square_error;
  Eigen::VectorXd rmses = cross_validated_scores(rmse, folds, model_ptr_.get());
  double out_of_sample_rmse = rmses.mean();

  // Make sure the RMSE computed doing leave one out cross validation is larger
  // than the in sample version.  This should always be true as the in sample
  // version has already seen the values we're trying to predict.
  EXPECT_LT(in_sample_rmse, out_of_sample_rmse);
}

TEST_F(LinearRegressionTest, test_cross_validated_predict) {
  const auto folds = leave_one_group_out<double>(dataset_, group_by_interval);

  const auto preds = cross_validated_predict(folds, model_ptr_.get());

  // Make sure the group cross validation resulted in folds that
  // are out of order
  EXPECT_TRUE(folds[0].name == "1");
  // And that cross_validate_predict put them back in order.
  EXPECT_TRUE(is_monotonic_increasing(preds.mean));
}

TEST_F(LinearRegressionTest, test_leave_one_group_out) {
  const auto folds = leave_one_group_out<double>(dataset_, group_by_interval);
  EvaluationMetric<Eigen::VectorXd> rmse = root_mean_square_error;
  Eigen::VectorXd rmses = cross_validated_scores(rmse, folds, model_ptr_.get());

  // Make sure we get a single RMSE for each of the three groups.
  EXPECT_EQ(rmses.size(), 3);
}
} // namespace albatross
