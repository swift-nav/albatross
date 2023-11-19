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
#include <albatross/GP>
#include <albatross/linalg/Utils>
#include <albatross/linalg/Block>
#include <gtest/gtest.h>

namespace albatross {

TEST(test_linalg_utils, test_block_arrow_ldlt) {
  const std::vector<double> u_0 = {-2., -1.};
  const std::vector<double> u_1 = {1., 2.};
  const std::vector<double> u_c = {0.};

  const std::vector<double> u_g = concatenate(std::vector<std::vector<double>>{u_0, u_1});
  const std::vector<double> u = concatenate(std::vector<std::vector<double>>{u_0, u_1, u_c});

  SquaredExponential<EuclideanDistance> cov;
  cov.squared_exponential_length_scale.value = 1.;
  cov.sigma_squared_exponential.value = 1.;
  Eigen::MatrixXd K_uu = Eigen::MatrixXd::Zero(u.size(), u.size());

  // Diagonal Blocks
  K_uu.block(0, 0, 2, 2) = cov(u_0);
  K_uu.block(2, 2, 2, 2) = cov(u_1);
  K_uu.block(4, 4, 1, 1) = cov(u_c);
  // Upper right
  K_uu.block(0, 4, 2, 1) = cov(u_0, u_c);
  K_uu.block(2, 4, 2, 1) = cov(u_1, u_c);
  // Lower Left
  K_uu.block(4, 0, 1, 2) = cov(u_c, u_0);
  K_uu.block(4, 2, 1, 2) = cov(u_c, u_1);

  std::cout << "K_uu" << std::endl;
  std::cout << K_uu << std::endl;

  const BlockDiagonal K_gg({cov(u_0), cov(u_1)});
  const Eigen::MatrixXd K_gc = cov(u_g, u_c);
  const Eigen::MatrixXd K_cc = cov(u_c);

  std::cout << "K_gg" << std::endl;
  std::cout << K_gg.toDense() << std::endl;

  std::cout << "K_gc" << std::endl;
  std::cout << K_gc << std::endl;

  std::cout << "K_cc" << std::endl;
  std::cout << K_cc << std::endl;
  const auto arrow_ldlt = block_symmetric_arrow_ldlt(K_gg, K_gc, K_cc);

  std::cout << "EXPECTED : " << std::endl;
  const Eigen::MatrixXd rhs = Eigen::MatrixXd::Random(K_uu.rows(), 2);
  const Eigen::SerializableLDLT expected_ldlt(K_uu);
  const Eigen::MatrixXd expected = expected_ldlt.sqrt_solve(rhs);

  std::cout << expected.transpose() * expected << std::endl;

  std::cout << "ACTUAL : " << std::endl;
  const Eigen::MatrixXd actual = arrow_ldlt.sqrt_solve(rhs);
  std::cout << actual.transpose() * actual << std::endl;

  EXPECT_LT((expected - actual).norm(), 1e-8);
}

} // namespace albatross
