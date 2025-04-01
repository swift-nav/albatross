/*
 * Copyright (C) 2025 Swift Navigation Inc.
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
#include <chrono>

#include <albatross/CGGP>

namespace albatross {

TEST(PartialCholesky, Construct) {
  Eigen::PartialCholesky<double> p;
  EXPECT_EQ(Eigen::Success, p.info());
}

static constexpr Eigen::Index cExampleSize = 4;

TEST(PartialCholesky, Compute) {
  Eigen::PartialCholesky<double> p;
  EXPECT_EQ(Eigen::Success, p.info());

  std::default_random_engine gen{22};
  const auto m = random_covariance_matrix(cExampleSize, gen);

  p.compute(m);
  std::cout << "A:\n" << m << std::endl;

  ASSERT_EQ(Eigen::Success, p.info());

  Eigen::MatrixXd L = p.matrixL();
  Eigen::PartialCholesky<double>::PermutationType P = p.permutationsP();

  Eigen::MatrixXd m_reconstructed{P * L * L.transpose() * P.transpose()};
  EXPECT_LT((m_reconstructed - m).norm(), 1.e-9) << m_reconstructed - m;
}

TEST(PartialCholesky, Solve) {
  Eigen::PartialCholesky<double> p;
  ASSERT_EQ(Eigen::Success, p.info());

  std::default_random_engine gen{22};
  const auto m = random_covariance_matrix(cExampleSize, gen);
  std::cout << m << std::endl;
  const Eigen::VectorXd b{Eigen::VectorXd::NullaryExpr(cExampleSize, [&gen]() {
    return std::normal_distribution<double>(0, 1)(gen);
  })};

  p.compute(m);

  const Eigen::VectorXd x_llt = m.ldlt().solve(b);
  std::cout << "x_llt:  " << x_llt.transpose() << std::endl;

  ASSERT_EQ(Eigen::Success, p.info());

  const Eigen::VectorXd x = p.solve(b);
  EXPECT_EQ(Eigen::Success, p.info());
  std::cout << "    x:  " << x.transpose() << std::endl;
  EXPECT_LT((x - x_llt).norm(), p.nugget());
}

}  // namespace albatross