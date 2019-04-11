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

#include <gtest/gtest.h>

#include "test_models.h"

#include <albatross/models/sparse_gp.h>

namespace albatross {

TEST(test_sparse_gp, test_sanity) {
  auto covariance = make_simple_covariance_function();
  auto dataset = make_toy_linear_data();

  auto direct = gp_from_covariance(covariance, "direct");

  LeaveOneOut loo;
  UniformlySpacedInducingPoints strategy(10);
  auto sparse = sparse_gp_from_covariance(covariance, strategy, loo, "sparse");

  UniformlySpacedInducingPoints bad_strategy(2);
  auto really_sparse =
      sparse_gp_from_covariance(covariance, bad_strategy, loo, "really_sparse");

  auto test_features = linspace(0.01, 9.9, 11);

  auto sparse_pred = sparse.fit(dataset).predict(test_features).joint();
  auto really_sparse_pred =
      really_sparse.fit(dataset).predict(test_features).joint();
  auto direct_pred = direct.fit(dataset).predict(test_features).joint();

  double sparse_error = (sparse_pred.mean - direct_pred.mean).norm();
  double really_sparse_error =
      (really_sparse_pred.mean - direct_pred.mean).norm();
  EXPECT_LT(sparse_error, 1e-2);
  EXPECT_LT(really_sparse_error, 0.5);
  EXPECT_GT(really_sparse_error, sparse_error);

  double sparse_cov_diff =
      (sparse_pred.covariance - direct_pred.covariance).norm();
  double really_sparse_cov_diff =
      (really_sparse_pred.covariance - direct_pred.covariance).norm();

  EXPECT_LT(sparse_cov_diff, 1e-2);
  EXPECT_LT(really_sparse_cov_diff, 0.5);
  EXPECT_GT(really_sparse_cov_diff, sparse_cov_diff);
}
} // namespace albatross
