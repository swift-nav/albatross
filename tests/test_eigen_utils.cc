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

#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>

#include "eigen_utils.h"

TEST(test_eigen_utils, test_dense_block_diag) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(2, 2);

  Eigen::MatrixXd B = Eigen::MatrixXd::Random(2, 3);

  std::vector<Eigen::MatrixXd> blocks = {A, B};
  auto C = albatross::block_diagonal(blocks);

  EXPECT_NEAR((C.block(0, 0, 2, 2) - A).norm(), 0., 1e-10);
  EXPECT_NEAR(C.block(0, 2, 2, 3).norm(), 0., 1e-10);
  EXPECT_NEAR((C.block(2, 2, 2, 3) - B).norm(), 0., 1e-10);
  EXPECT_NEAR(C.block(2, 0, 2, 2).norm(), 0., 1e-10);
}

TEST(test_eigen_utils, test_diag_block_diag) {
  Eigen::DiagonalMatrix<double, Eigen::Dynamic> A =
      Eigen::VectorXd::Random(2).asDiagonal();

  Eigen::DiagonalMatrix<double, Eigen::Dynamic> B =
      Eigen::VectorXd::Random(3).asDiagonal();

  std::vector<decltype(A)> blocks = {A, B};
  auto C = albatross::block_diagonal(blocks);

  EXPECT_NEAR((A.diagonal() - C.diagonal().segment(0, A.rows())).norm(), 0.,
              1E-10);
  EXPECT_NEAR((B.diagonal() - C.diagonal().segment(A.rows(), B.rows())).norm(),
              0., 1E-10);
}
