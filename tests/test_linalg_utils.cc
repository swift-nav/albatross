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
#include <albatross/utils/LinalgUtils>
#include <gtest/gtest.h>

#include <albatross/Indexing>
#include <albatross/src/utils/random_utils.hpp>

namespace albatross {

TEST(test_linalg_utils, test_print_eigen_values) {

  Eigen::Index k = 10;
  Eigen::MatrixXd random = Eigen::MatrixXd::Random(k, k);
  random = random * random.transpose();

  std::vector<int> features;
  for (Eigen::Index i = 0; i < k; ++i) {
    features.push_back(i);
  }

  // Simple sanity check;
  std::ostringstream oss;
  print_small_eigen_directions(random, features, k - 4,
                               details::DEFAULT_EIGEN_VALUE_PRINT_THRESHOLD,
                               &oss);
  print_large_eigen_directions(random, features, k - 4,
                               details::DEFAULT_EIGEN_VALUE_PRINT_THRESHOLD,
                               &oss);
}

TEST(test_linalg_utils, test_lyapunov_solver) {

  std::default_random_engine gen(2012);

  Eigen::Index k = 10;
  Eigen::MatrixXd C(k, k);
  gaussian_fill(C, gen);
  C = C.colPivHouseholderQr().matrixQ();
  C.array() *= 0.1;

  const auto X = random_covariance_matrix(k, gen);

  const auto Q = X - C * X * C.transpose();

  const auto actual = lyapunov_solve(C, Q);

  std::cout << actual << std::endl;
  std::cout << "=============" << std::endl;
  std::cout << Q << std::endl;
  std::cout << "=============" << std::endl;
  std::cout << (actual - Q).norm() << std::endl;
}

} // namespace albatross
