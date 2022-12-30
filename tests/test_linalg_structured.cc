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
#include <albatross/linalg/Block>

#include <gtest/gtest.h>
#include <iostream>

namespace albatross {

struct StructuredBlockDiagonal {

  Eigen::MatrixXd dense() const {
    Eigen::MatrixXd output(4, 4);
    output.row(0) << 1, 0, 0.9, 0;
    output.row(1) << 0, 1, 0, 0.7;
    output.row(2) << 0.9, 0, 1, 0;
    output.row(3) << 0, 0.7, 0, 1;
    return output;
  }

  Structured<BlockDiagonal> structured() const {
    Eigen::MatrixXd block_0(2, 2);
    block_0 << 1, 0.7, 0.7, 1;
    Eigen::MatrixXd block_1(2, 2);
    block_1 << 1, 0.9, 0.9, 1;
    BlockDiagonal block_diagonal;
    block_diagonal.blocks.push_back(block_0);
    block_diagonal.blocks.push_back(block_1);
    Eigen::VectorXi indices(4);
    indices << 1, 3, 2, 0;
    Eigen::PermutationMatrix<Eigen::Dynamic> P(indices);
    return Structured<BlockDiagonal>{block_diagonal, P, P.transpose()};
  }
};

template <typename TestCase> class StructuredTest : public ::testing::Test {
public:
  TestCase test_case;
};

typedef ::testing::Types<StructuredBlockDiagonal> StructuredTestCases;
TYPED_TEST_SUITE(StructuredTest, StructuredTestCases);

TYPED_TEST(StructuredTest, test_to_dense_matches_test_case) {
  const Eigen::MatrixXd dense = this->test_case.dense();
  const auto structured = this->test_case.structured();
  const Eigen::MatrixXd actual = to_dense(structured);
  EXPECT_EQ(actual, dense);
}

TYPED_TEST(StructuredTest, test_to_dense_components) {
  const Eigen::MatrixXd dense = this->test_case.dense();
  const auto structured = this->test_case.structured();
  Eigen::MatrixXd expected =
      structured.P_rows * to_dense(structured.matrix) * structured.P_cols;

  EXPECT_EQ(expected, dense);
}

TYPED_TEST(StructuredTest, test_reverse) {
  const Eigen::MatrixXd dense = this->test_case.dense();
  const auto structured = this->test_case.structured();
  // Pr * A * Pc = X
  // A = Pr.T * X * Pc.T
  Eigen::MatrixXd reverse =
      structured.P_rows.transpose() * dense * structured.P_cols.transpose();

  EXPECT_EQ(reverse, to_dense(structured.matrix));
}
} // namespace albatross