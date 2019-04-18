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

#include <Eigen/Cholesky>

#include <albatross/src/utils/block_utils.h>

namespace albatross {

struct BlockExample {
  BlockDiagonal block;
  Eigen::MatrixXd dense;
};

BlockExample block_example() {
  Eigen::Index block_size = 3;
  Eigen::Index block_count = 2;
  Eigen::Index n = block_size * block_count;

  BlockDiagonal block_diag;
  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(n, n);
  for (Eigen::Index i = 0; i < block_count; ++i) {
    Eigen::MatrixXd m = Eigen::MatrixXd::Random(block_size, block_size);
    m = m.transpose() * m;
    block_diag.blocks.push_back(m);
    dense.block(i * block_size, i * block_size, block_size, block_size) = m;
  }
  return {block_diag, dense};
}

TEST(test_block_utils, test_to_dense) {

  auto example = block_example();
  auto block_diag = example.block;
  auto dense = example.dense;

  const auto block_result = block_diag.toDense();
  EXPECT_LE((block_result - dense).norm(), 1e-6);
}

TEST(test_block_utils, test_diagonal) {

  auto example = block_example();
  auto block_diag = example.block;
  auto dense = example.dense;

  const auto block_result = block_diag.diagonal();
  EXPECT_LE((block_result - dense.diagonal()).norm(), 1e-6);
}

TEST(test_block_utils, test_dot_product) {

  auto example = block_example();
  auto block_diag = example.block;
  auto dense = example.dense;

  Eigen::MatrixXd rhs = Eigen::MatrixXd::Random(dense.cols(), 3);

  const Eigen::MatrixXd block_result = block_diag * rhs;
  const Eigen::MatrixXd dense_result = dense * rhs;

  EXPECT_LE((block_result - dense_result).norm(), 1e-6);
}

TEST(test_block_utils, test_solve) {

  auto example = block_example();
  auto block_diag = example.block;
  auto dense = example.dense;

  Eigen::VectorXd rhs = Eigen::VectorXd::Random(dense.cols());
  const auto block_result = block_diag.llt().solve(rhs);

  const Eigen::VectorXd dense_result = dense.llt().solve(rhs);

  EXPECT_LE((block_result - dense_result).norm(), 1e-6);
}

TEST(test_block_utils, test_matrix_l) {

  auto example = block_example();
  auto block_diag = example.block;
  auto dense = example.dense;

  Eigen::MatrixXd rhs = Eigen::MatrixXd::Random(dense.cols(), 3);
  const auto block_llt = block_diag.llt();
  BlockDiagonal block_l_val = block_llt.matrixL();
  const auto block_result = block_l_val * rhs;
  Eigen::MatrixXd block_l = block_l_val.toDense();

  const auto dense_llt = dense.llt();
  Eigen::MatrixXd dense_l = dense_llt.matrixL();
  const Eigen::MatrixXd dense_result = dense_l * rhs;

  EXPECT_LE((block_l - dense_l).norm(), 1e-6);
  EXPECT_LE((block_result - dense_result).norm(), 1e-6);
}

} // namespace albatross
