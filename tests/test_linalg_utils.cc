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

#include <albatross/Core>
#include <albatross/linalg/Utils>
#include <gtest/gtest.h>

namespace albatross {

TEST(test_linalg_utils, test_qr_sqrt_solve) {
  const int n = 5;
  const Eigen::MatrixXd A = Eigen::MatrixXd::Random(2 * n, n);

  const auto qr = A.colPivHouseholderQr();

  const Eigen::MatrixXd rhs = Eigen::MatrixXd::Random(n, 3);
  const Eigen::MatrixXd expected_quad =
      rhs.transpose() * (A.transpose() * A).ldlt().solve(rhs);
  const Eigen::MatrixXd sqrt = sqrt_solve(qr, rhs);
  const Eigen::MatrixXd actual_quad = sqrt.transpose() * sqrt;

  EXPECT_LT((actual_quad - expected_quad).norm(), 1e-14);
}

TEST(test_linalg_utils, test_spqr_pivot_threshold_small_matrix) {
  // Regression test: matrices smaller than the minimum sparse pivot
  // size must use the tight small-problem threshold (1e-15) instead of
  // falling through to the large-problem SuiteSparseQR policy.
  const Eigen::Index small_size = kMinSparsePivotSize - 1;
  const Eigen::MatrixXd small =
      Eigen::MatrixXd::Identity(small_size, small_size);
  EXPECT_EQ(
      SPQR_pivot_threshold(small, kMinSparsePivotSize, kSPQRPivotCoefficient),
      1e-15);

  // Large matrices should keep using the (scaled) default policy.
  const Eigen::Index large_size = kMinSparsePivotSize;
  const Eigen::MatrixXd large =
      Eigen::MatrixXd::Identity(large_size, large_size);
  const double expected_large = kSPQRPivotCoefficient *
                                cast::to_double(large_size + large_size) *
                                Eigen::NumTraits<double>::epsilon();
  EXPECT_DOUBLE_EQ(
      SPQR_pivot_threshold(large, kMinSparsePivotSize, kSPQRPivotCoefficient),
      expected_large);
}

TEST(test_linalg_utils, test_print_eigen_values) {

  constexpr Eigen::Index k = 10;
  Eigen::MatrixXd random = Eigen::MatrixXd::Random(k, k);
  random = random * random.transpose();

  std::vector<Eigen::Index> features;
  for (Eigen::Index i = 0; i < k; ++i) {
    features.push_back(i);
  }

  // Simple sanity check;
  std::ostringstream oss;
  print_small_eigen_directions(random, features, cast::to_size(k) - 4,
                               details::DEFAULT_EIGEN_VALUE_PRINT_THRESHOLD,
                               &oss);
  print_large_eigen_directions(random, features, cast::to_size(k) - 4,
                               details::DEFAULT_EIGEN_VALUE_PRINT_THRESHOLD,
                               &oss);
}

} // namespace albatross
