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
#include <albatross/Indexing>

#include <gtest/gtest.h>

namespace albatross {

TEST(test_indexing, test_vector_subset) {

  std::vector<std::size_t> idx;
  std::vector<int> expected;

  std::vector<int> x = {3, 6, 4, 7, 9};

  idx = {1, 2};
  expected = {6, 4};
  EXPECT_EQ(subset(x, idx), expected);

  idx = {2, 1};
  expected = {4, 6};
  EXPECT_EQ(subset(x, idx), expected);

  idx = {3, 3};
  expected = {7, 7};
  EXPECT_EQ(subset(x, idx), expected);

  idx = {2};
  expected = {4};
  EXPECT_EQ(subset(x, idx), expected);

  idx = {0, 1, 2, 3, 4};
  expected = x;
  EXPECT_EQ(subset(x, idx), expected);

  idx = {};
  expected = {};
  EXPECT_EQ(subset(x, idx), expected);
}

TEST(test_indexing, test_vector_set_subset) {
  std::vector<std::size_t> idx;
  Eigen::VectorXd expected(5);
  Eigen::VectorXd from;

  Eigen::VectorXd x(5);
  x << 3., 6., 4., 7., 9.;
  Eigen::VectorXd to;

  idx = {1, 2};
  from.resize(2);
  from << -1, -2;
  expected << 3., -1, -2, 7, 9;
  to = Eigen::VectorXd(x);
  set_subset(from, idx, &to);
  EXPECT_EQ(to, expected);

  idx = {2, 1};
  from.resize(2);
  from << -1, -2;
  expected << 3., -2, -1, 7, 9;
  to = Eigen::VectorXd(x);
  set_subset(from, idx, &to);
  EXPECT_EQ(to, expected);

  idx = {3, 3};
  from.resize(2);
  from << -1, -2;
  expected << 3., 6., 4., -2., 9.;
  to = Eigen::VectorXd(x);
  set_subset(from, idx, &to);
  EXPECT_EQ(to, expected);

  idx = {2};
  from.resize(1);
  from << -1;
  expected << 3., 6., -1., 7, 9;
  to = Eigen::VectorXd(x);
  set_subset(from, idx, &to);
  EXPECT_EQ(to, expected);

  idx = {0, 1, 2, 3, 4};
  from.resize(5);
  from << -1, -2., -3., -4., -5;
  expected << -1., -2., -3., -4, -5;
  to = Eigen::VectorXd(x);
  set_subset(from, idx, &to);
  EXPECT_EQ(to, expected);

  idx = {};
  from.resize(0);
  to = Eigen::VectorXd(x);
  expected = Eigen::VectorXd(x);
  set_subset(from, idx, &to);
  EXPECT_EQ(to, expected);
}

TEST(test_indexing, test_diagonal_set_subset) {
  std::vector<std::size_t> idx;
  DiagonalMatrixXd expected;
  Eigen::VectorXd expected_diag(5);
  DiagonalMatrixXd from;
  Eigen::VectorXd from_diag;

  Eigen::VectorXd diag(5);
  diag << 3., 6., 4., 7., 9.;
  DiagonalMatrixXd x = diag.asDiagonal();
  DiagonalMatrixXd to;

  idx = {1, 2};
  from_diag.resize(2);
  from_diag << -1, -2;
  from = from_diag.asDiagonal();
  expected_diag << 3., -1, -2, 7, 9;
  expected = expected_diag.asDiagonal();
  to = DiagonalMatrixXd(x);
  set_subset(from, idx, &to);
  EXPECT_TRUE(to == expected);

  idx = {2, 1};
  from_diag.resize(2);
  from_diag << -1, -2;
  from = from_diag.asDiagonal();
  expected_diag << 3., -2, -1, 7, 9;
  expected = expected_diag.asDiagonal();
  to = DiagonalMatrixXd(x);
  set_subset(from, idx, &to);
  EXPECT_TRUE(to == expected);

  idx = {3, 3};
  from_diag.resize(2);
  from_diag << -1, -2;
  from = from_diag.asDiagonal();
  expected_diag << 3., 6., 4., -2, 9;
  expected = expected_diag.asDiagonal();
  to = DiagonalMatrixXd(x);
  set_subset(from, idx, &to);
  EXPECT_TRUE(to == expected);

  idx = {2};
  from_diag.resize(1);
  from_diag << -1;
  from = from_diag.asDiagonal();
  expected_diag << 3., 6., -1., 7., 9.;
  expected = expected_diag.asDiagonal();
  to = DiagonalMatrixXd(x);
  set_subset(from, idx, &to);
  EXPECT_TRUE(to == expected);

  idx = {0, 1, 2, 3, 4};
  from_diag.resize(5);
  from_diag << -1, -2., -3., -4., -5;
  from = from_diag.asDiagonal();
  expected_diag << -1, -2., -3., -4., -5.;
  expected = expected_diag.asDiagonal();
  to = DiagonalMatrixXd(x);
  set_subset(from, idx, &to);
  EXPECT_TRUE(to == expected);

  idx = {};
  from_diag.resize(0);
  from = from_diag.asDiagonal();
  expected = DiagonalMatrixXd(x);
  to = DiagonalMatrixXd(x);
  set_subset(from, idx, &to);
  EXPECT_TRUE(to == expected);
}

TEST(test_indexing, test_eigen_vector_subset) {
  Eigen::VectorXd x(5);
  x << 3., 6., 4., 7., 9.;
  std::vector<std::size_t> idx;

  Eigen::VectorXd expected;
  idx = {1, 2};
  expected.resize(2);
  expected << 6., 4.;

  EXPECT_EQ(subset(x, idx), expected);

  idx = {2, 1};
  expected.resize(2);
  expected << 4., 6.;
  EXPECT_EQ(subset(x, idx), expected);

  idx = {3, 3};
  expected.resize(2);
  expected << 7., 7.;
  EXPECT_EQ(subset(x, idx), expected);

  idx = {2};
  expected.resize(1);
  expected << 4.;
  EXPECT_EQ(subset(x, idx), expected);

  idx = {0, 1, 2, 3, 4};
  expected = x;
  EXPECT_EQ(subset(x, idx), expected);

  idx = {};
  expected.resize(0);
  EXPECT_EQ(subset(x, idx), expected);
}

TEST(test_indexing, test_matrix_subset_col) {

  Eigen::MatrixXd expected;
  Eigen::MatrixXd x(4, 4);
  x.row(0) << 1., 2., 3., 4.;
  x.row(1) << 5., 6., 7., 8.;
  x.row(2) << 9., 10., 11., 12.;
  x.row(3) << 13., 14., 15., 16.;
  std::vector<std::size_t> idx;

  idx = {1, 2};
  expected.resize(4, 2);
  expected.col(0) = x.col(1);
  expected.col(1) = x.col(2);
  EXPECT_EQ(subset_cols(x, idx), expected);

  idx = {2, 1};
  expected.resize(4, 2);
  expected.col(0) = x.col(2);
  expected.col(1) = x.col(1);
  EXPECT_EQ(subset_cols(x, idx), expected);

  idx = {3, 3};
  expected.resize(4, 2);
  expected.col(0) = x.col(3);
  expected.col(1) = x.col(3);
  EXPECT_EQ(subset_cols(x, idx), expected);

  idx = {2};
  expected.resize(4, 1);
  expected.col(0) = x.col(2);
  EXPECT_EQ(subset_cols(x, idx), expected);

  idx = {0, 1, 2, 3};
  expected = x;
  EXPECT_EQ(subset_cols(x, idx), expected);

  idx = {};
  expected.resize(4, 0);
  EXPECT_EQ(subset_cols(x, idx), expected);
}

TEST(test_indexing, test_matrix_subset_row) {

  Eigen::MatrixXd expected;
  Eigen::MatrixXd x(4, 4);
  x.row(0) << 1., 2., 3., 4.;
  x.row(1) << 5., 6., 7., 8.;
  x.row(2) << 9., 10., 11., 12.;
  x.row(3) << 13., 14., 15., 16.;
  std::vector<std::size_t> idx;

  idx = {1, 2};
  expected.resize(2, 4);
  expected.row(0) = x.row(1);
  expected.row(1) = x.row(2);
  EXPECT_EQ(subset_rows(x, idx), expected);

  idx = {2, 1};
  expected.resize(2, 4);
  expected.row(0) = x.row(2);
  expected.row(1) = x.row(1);
  EXPECT_EQ(subset_rows(x, idx), expected);

  idx = {3, 3};
  expected.resize(2, 4);
  expected.row(0) = x.row(3);
  expected.row(1) = x.row(3);
  EXPECT_EQ(subset_rows(x, idx), expected);

  idx = {2};
  expected.resize(1, 4);
  expected.row(0) = x.row(2);
  EXPECT_EQ(subset_rows(x, idx), expected);

  idx = {0, 1, 2, 3};
  expected = x;
  EXPECT_EQ(subset_rows(x, idx), expected);

  idx = {};
  expected.resize(0, 4);
  EXPECT_EQ(subset_rows(x, idx), expected);
}

TEST(test_indexing, test_matrix_symmetric_subset) {

  Eigen::MatrixXd expected;
  Eigen::MatrixXd x(4, 4);
  x.row(0) << 1., 2., 3., 4.;
  x.row(1) << 5., 6., 7., 8.;
  x.row(2) << 9., 10., 11., 12.;
  x.row(3) << 13., 14., 15., 16.;
  std::vector<std::size_t> idx;

  idx = {1, 2};
  expected = subset_cols(subset_rows(x, idx), idx);
  EXPECT_EQ(symmetric_subset(x, idx), expected);

  idx = {2, 1};
  expected = subset_cols(subset_rows(x, idx), idx);
  EXPECT_EQ(symmetric_subset(x, idx), expected);

  idx = {3, 3};
  expected = subset_cols(subset_rows(x, idx), idx);
  EXPECT_EQ(symmetric_subset(x, idx), expected);

  idx = {2};
  expected = subset_cols(subset_rows(x, idx), idx);
  EXPECT_EQ(symmetric_subset(x, idx), expected);

  idx = {0, 1, 2, 3};
  expected = subset_cols(subset_rows(x, idx), idx);
  EXPECT_EQ(symmetric_subset(x, idx), expected);

  idx = {};
  expected = subset_cols(subset_rows(x, idx), idx);
  EXPECT_EQ(symmetric_subset(x, idx), expected);
}

} // namespace albatross
