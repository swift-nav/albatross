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

#include <albatross/GP>
#include <albatross/utils/RandomUtils>

#include "test_utils.h"

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

TEST(test_block_utils, test_block_ldlt) {
  const auto example = block_example();
  const BlockDiagonalLDLT ldlt = example.block.ldlt();

  const Eigen::MatrixXd expect_identity = ldlt.solve(example.dense);
  const Eigen::MatrixXd identity =
      Eigen::MatrixXd::Identity(example.dense.rows(), example.dense.cols());
  EXPECT_LT((expect_identity - identity).norm(), 1e-8);

  EXPECT_NEAR(example.dense.determinant(),
              exp(example.block.ldlt().log_determinant()), 1e-8);
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
  const auto block_result = block_diag.ldlt().solve(rhs);

  const Eigen::VectorXd dense_result = dense.ldlt().solve(rhs);

  EXPECT_LE((block_result - dense_result).norm(), 1e-6);
}

TEST(test_block_utils, test_sqrt_methods) {

  auto example = block_example();
  auto block_diag = example.block;
  auto dense = example.dense;

  Eigen::MatrixXd rhs = Eigen::MatrixXd::Random(dense.cols(), 3);
  const auto block_ldlt = block_diag.ldlt();

  const Eigen::MatrixXd block_sqrt = block_ldlt.sqrt_solve(rhs);
  const Eigen::MatrixXd block_result = block_sqrt.transpose() * block_sqrt;

  const Eigen::SerializableLDLT dense_ldlt = dense.ldlt();
  const Eigen::MatrixXd dense_sqrt = dense_ldlt.sqrt_solve(rhs);
  const Eigen::MatrixXd dense_result = dense_sqrt.transpose() * dense_sqrt;

  EXPECT_LE((block_result - dense_result).norm(), 1e-6);
}

TEST(test_block_utils, test_block_symmetric) {

  std::default_random_engine gen(2012);
  const auto X = random_covariance_matrix(5, gen);

  const Eigen::MatrixXd rhs = Eigen::MatrixXd::Random(X.cols(), 3);
  const Eigen::MatrixXd expected = X.ldlt().solve(rhs);

  const Eigen::MatrixXd A = X.topLeftCorner(3, 3);
  const Eigen::MatrixXd B = X.topRightCorner(3, 2);
  const Eigen::MatrixXd C = X.bottomRightCorner(2, 2);

  // Test when constructing from the actual blocks.
  const auto block = build_block_symmetric(A.ldlt(), B, C);
  const Eigen::MatrixXd actual = block.solve(rhs);
  EXPECT_TRUE(actual.isApprox(expected));

  // And again using a pre computed S
  const Eigen::MatrixXd S = C - B.transpose() * A.ldlt().solve(B);
  const Eigen::SerializableLDLT S_ldlt = S.ldlt();
  const auto block_direct = build_block_symmetric(A.ldlt(), B, S_ldlt);
  const Eigen::MatrixXd actual_direct = block_direct.solve(rhs);
  EXPECT_TRUE(actual_direct.isApprox(expected));
}

} // namespace albatross
