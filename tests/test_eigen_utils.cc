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

#include <vector>

#include <albatross/Core>
#include <albatross/Indexing>

#include <albatross/src/utils/eigen_utils.hpp>

#include <gtest/gtest.h>
#include <iostream>

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

TEST(test_eigen_utils, test_vertical_stack_matrix) {
  Eigen::MatrixXd A(2, 3);
  A << 1, 2, 3, 4, 5, 6;

  Eigen::MatrixXd B(3, 3);
  B << 7, 8, 9, 10, 11, 12, 13, 14, 15;

  std::vector<Eigen::MatrixXd> input = {A, B};
  const auto C = albatross::vertical_stack(input);
  Eigen::MatrixXd expected(5, 3);
  expected << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15;
  EXPECT_EQ(expected, C);
}

TEST(test_eigen_utils, test_vertical_stack_vector) {
  Eigen::VectorXd A(6);
  A << 1, 2, 3, 4, 5, 6;

  Eigen::VectorXd B(9);
  B << 7, 8, 9, 10, 11, 12, 13, 14, 15;

  std::vector<Eigen::VectorXd> input = {A, B};
  const auto C = albatross::vertical_stack(input);
  Eigen::VectorXd expected(15);
  expected << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15;
  EXPECT_EQ(expected, C);
}

TEST(test_eigen_utils, test_truncated_solve_sanity) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(3, 3);
  A = A.transpose() * A;

  Eigen::VectorXd soln(3);
  soln << 1., 2., 3.;
  Eigen::VectorXd rhs = A * soln;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> evd(A);

  auto actual = albatross::truncated_psd_solve(evd, rhs);

  for (Eigen::Index i = 0; i < soln.size(); ++i) {
    EXPECT_LT(fabs(soln[i] - actual[i]), 1e-6);
  }
}

TEST(test_eigen_utils, test_truncated_solve_low_rank) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(2, 3);
  A = A.transpose() * A;

  // Here we project the arbitrary solution vector
  // to remove anything in the nullspace of A since
  // we can never expect to be able to recover
  // those components.
  Eigen::VectorXd soln(3);
  soln << 1., 2., 3.;
  Eigen::VectorXd k = A.fullPivLu().kernel().col(0);
  k.normalize();
  double scale = soln.transpose() * k;
  soln = soln - scale * k;

  Eigen::VectorXd rhs = A * soln;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> evd(A);

  auto actual = albatross::truncated_psd_solve(evd, rhs);

  for (Eigen::Index i = 0; i < soln.size(); ++i) {
    EXPECT_LT(fabs(soln[i] - actual[i]), 1e-6);
  }
}
