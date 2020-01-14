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

#include <albatross/Core>
#include <albatross/Indexing>
#include <albatross/src/eigen/serializable_ldlt.hpp>

#include <gtest/gtest.h>

namespace albatross {

class SerializableLDLTTest : public ::testing::Test {
public:
  SerializableLDLTTest() : cov(), information() {
    const int n = 5;
    const auto part = Eigen::MatrixXd(Eigen::MatrixXd::Random(n, n));
    cov = part * part.transpose();
    information = Eigen::VectorXd::Ones(n);
  };

  Eigen::MatrixXd cov;
  Eigen::VectorXd information;
};

TEST_F(SerializableLDLTTest, test_solve) {
  auto ldlt = cov.ldlt();
  const auto serializable_ldlt = Eigen::SerializableLDLT(ldlt);
  EXPECT_EQ(serializable_ldlt.solve(information), ldlt.solve(information));
}

TEST_F(SerializableLDLTTest, test_inverse_diagonal) {
  auto ldlt = cov.ldlt();
  const auto serializable_ldlt = Eigen::SerializableLDLT(ldlt);
  const auto inverse = cov.inverse();
  const auto diag = serializable_ldlt.inverse_diagonal();
  EXPECT_LE(fabs((Eigen::VectorXd(inverse.diagonal()) - diag).norm()), 1e-8);
}

TEST_F(SerializableLDLTTest, test_log_det) {
  auto ldlt = cov.ldlt();
  const auto serializable_ldlt = Eigen::SerializableLDLT(ldlt);
  const double expected = log(cov.determinant());
  const double actual = serializable_ldlt.log_determinant();

  // The sqrt solves aren't unique, but we can check the outer product
  EXPECT_NEAR(expected, actual, 1e-8);
}

TEST_F(SerializableLDLTTest, test_sqrt_solve) {
  auto ldlt = cov.ldlt();
  const auto serializable_ldlt = Eigen::SerializableLDLT(ldlt);
  const Eigen::VectorXd expected = cov.llt().matrixL().solve(information);
  const Eigen::VectorXd actual = serializable_ldlt.sqrt_solve(information);

  // The sqrt solves aren't unique, but we can check the outer product
  EXPECT_NEAR(expected.transpose() * expected, actual.transpose() * actual,
              1e-4);

  // And can also check that the sqrt solve when applied twice produces the
  // inverse.
  Eigen::MatrixXd identity =
      Eigen::MatrixXd::Identity(information.size(), information.size());
  Eigen::MatrixXd inv = serializable_ldlt.sqrt_solve(identity);
  inv = inv.transpose() * inv;
  const Eigen::VectorXd actual_inverse = inv * information;

  EXPECT_LE((cov.ldlt().solve(information) - actual_inverse).norm(), 1e-8);
}

} // namespace albatross
