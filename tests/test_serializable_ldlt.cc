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
#include <albatross/src/eigen/serializable_ldlt.h>

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

} // namespace albatross
