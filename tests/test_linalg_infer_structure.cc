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
#include <albatross/linalg/Block>

#include <gtest/gtest.h>
#include <iostream>

namespace albatross {

struct FullyIndependent {

  Eigen::MatrixXd matrix() const { return Eigen::MatrixXd::Identity(5, 5); }

  std::vector<std::set<Eigen::Index>> expected() const {
    return std::vector<std::set<Eigen::Index>>{{0}, {1}, {2}, {3}, {4}};
  }
};

struct FullyConnected {

  Eigen::MatrixXd matrix() const {
    Eigen::MatrixXd x = Eigen::MatrixXd::Identity(5, 5);
    // Add an offset diagonal so everything is connected
    for (Eigen::Index i = 1; i < x.rows(); ++i) {
      x(i, i - 1) = 1;
      x(i - 1, i) = 1;
    }
    return x;
  }

  std::vector<std::set<Eigen::Index>> expected() const {
    return std::vector<std::set<Eigen::Index>>{{0, 1, 2, 3, 4}};
  }
};

struct TwoBlocks {

  Eigen::MatrixXd matrix() const {
    Eigen::MatrixXd x = Eigen::MatrixXd::Identity(6, 6);
    // Add an offset diagonal so everything is connected
    for (Eigen::Index i = 1; i < x.rows(); ++i) {
      // disconnect a row from the previous ones
      if (i != 3) {
        x(i, i - 1) = 1;
        x(i - 1, i) = 1;
      }
    }
    return x;
  }

  std::vector<std::set<Eigen::Index>> expected() const {
    return std::vector<std::set<Eigen::Index>>{{0, 1, 2}, {3, 4, 5}};
  }
};

struct ThreeBlocksEmptyRow {

  Eigen::MatrixXd matrix() const {
    auto x = TwoBlocks().matrix();
    // completely zero out row/col 3
    x.row(3).fill(0.);
    x.col(3).fill(0.);

    return x;
  }

  std::vector<std::set<Eigen::Index>> expected() const {
    return std::vector<std::set<Eigen::Index>>{{0, 1, 2}, {3}, {4, 5}};
  }
};

struct ThreeBlocksEmptyRowPermuted {

  Eigen::MatrixXd matrix() const {
    auto x = ThreeBlocksEmptyRow().matrix();
    assert(x.rows() == 6);
    Eigen::VectorXi indices(6);
    indices << 2, 4, 3, 1, 0, 5;
    Eigen::PermutationMatrix<Eigen::Dynamic> P(indices);
    x = P * x * P.transpose();
    return x;
  }

  std::vector<std::set<Eigen::Index>> expected() const {
    return std::vector<std::set<Eigen::Index>>{{0, 5}, {1}, {2, 3, 4}};
  }
};

struct EmptyMatrix {
  Eigen::MatrixXd matrix() const { return Eigen::MatrixXd::Zero(0, 0); }

  std::vector<std::set<Eigen::Index>> expected() const { return {}; }
};

struct SizeOne {
  Eigen::MatrixXd matrix() const { return Eigen::MatrixXd::Identity(1, 1); }

  std::vector<std::set<Eigen::Index>> expected() const { return {{0}}; }
};

template <typename TestCase> class InferStructureTest : public ::testing::Test {
public:
  TestCase test_case;
};

typedef ::testing::Types<FullyIndependent, FullyConnected, TwoBlocks,
                         ThreeBlocksEmptyRow, ThreeBlocksEmptyRowPermuted,
                         EmptyMatrix, SizeOne>
    InferStructureTestCases;
TYPED_TEST_SUITE(InferStructureTest, InferStructureTestCases);

TYPED_TEST(InferStructureTest, test_expected) {
  const auto actual = linalg::infer_diagonal_blocks(this->test_case.matrix());
  EXPECT_EQ(this->test_case.expected(), actual);
}

TYPED_TEST(InferStructureTest, test_independent) {
  const auto x = this->test_case.matrix();
  const auto blocks = linalg::infer_diagonal_blocks(x);

  // Make sure any values outside of blocks are actually 0
  for (const auto &block : blocks) {
    for (Eigen::Index i : block) {
      for (Eigen::Index j = 0; j < x.cols(); ++j) {
        if (block.find(j) == block.end()) {
          EXPECT_EQ(x(i, j), 0);
        }
      }
    }
  }
}

TYPED_TEST(InferStructureTest, test_to_permutation_matrix) {
  const auto x = this->test_case.matrix();
  const auto blocks = linalg::infer_diagonal_blocks(x);
  const auto P = linalg::to_permutation_matrix(blocks);
  const Eigen::MatrixXd block_diag = P * x * P.transpose();

  // If you permute the input matrix you should end up with
  // a block diagonal matrix, so if you then infer the blocks
  // you should get monotonically increasing indices.
  const auto block_diag_blocks = linalg::infer_diagonal_blocks(block_diag);
  Eigen::Index expected = 0;
  for (const auto &block : block_diag_blocks) {
    for (const auto &ind : block) {
      EXPECT_EQ(ind, expected);
      ++expected;
    }
  }
}

TYPED_TEST(InferStructureTest, test_to_structured_matrix) {
  const auto x = this->test_case.matrix();
  const auto structured_x = linalg::infer_block_diagonal_matrix(x);
  EXPECT_EQ(x, structured_x.to_dense());
}

} // namespace albatross