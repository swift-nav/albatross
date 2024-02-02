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
#include <albatross/linalg/Utils>
#include <gtest/gtest.h>

namespace albatross {

struct PermutationTestData {
  Eigen::MatrixXd A;
  Eigen::VectorXi P;
};

PermutationTestData test_data() {
  Eigen::VectorXi P(4);
  P << 2, 3, 0, 1;
  return PermutationTestData{Eigen::MatrixXd::Random(P.size(), P.size()), P};
}

PermutationTestData identity_test_data() {
  Eigen::VectorXi P(4);
  P << 0, 1, 2, 3;
  return PermutationTestData{Eigen::MatrixXd::Random(P.size(), P.size()), P};
}

TEST(test_permutation_utils, test_identity_from_left) {
  const auto test = identity_test_data();
  const auto actual = permute::from_left(test.P, test.A);
  EXPECT_EQ(test.A, actual);
}

TEST(test_permutation_utils, test_identity_transpose_from_left) {
  const auto test = identity_test_data();
  const auto actual = permute::transpose_from_left(test.P, test.A);
  EXPECT_EQ(test.A, actual);
}

TEST(test_permutation_utils, test_identity_from_right) {
  const auto test = identity_test_data();
  const auto actual = permute::from_right(test.A, test.P);
  EXPECT_EQ(test.A, actual);
}

TEST(test_permutation_utils, test_identity_transpose_from_right) {
  const auto test = identity_test_data();
  const auto actual = permute::transpose_from_right(test.A, test.P);
  EXPECT_EQ(test.A, actual);
}

TEST(test_permutation_utils, test_from_left) {
  const auto test = test_data();
  const auto actual = permute::from_left(test.P, test.A);
  for (Eigen::Index i = 0; i < test.A.rows(); ++i) {
    const Eigen::Index pi = test.P[i];
    EXPECT_EQ(test.A.row(pi), actual.row(i));
  }
}

TEST(test_permutation_utils, test_transpose_from_left) {
  const auto test = test_data();
  const auto actual = permute::transpose_from_left(test.P, test.A);
  for (Eigen::Index i = 0; i < test.A.rows(); ++i) {
    const Eigen::Index pi = test.P[i];
    EXPECT_EQ(test.A.row(i), actual.row(pi));
  }
}

TEST(test_permutation_utils, test_from_right) {
  const auto test = test_data();
  const auto actual = permute::from_right(test.A, test.P);
  for (Eigen::Index i = 0; i < test.A.cols(); ++i) {
    const Eigen::Index pi = test.P[i];
    EXPECT_EQ(test.A.col(pi), actual.col(i));
  }
}

TEST(test_permutation_utils, test_tranpose_from_right) {
  const auto test = test_data();
  const auto actual = permute::transpose_from_right(test.A, test.P);
  for (Eigen::Index i = 0; i < test.A.cols(); ++i) {
    const Eigen::Index pi = test.P[i];
    EXPECT_EQ(test.A.col(i), actual.col(pi));
  }
}

TEST(test_permutation_utils, test_roundtrip_from_left) {
  const auto test = test_data();
  // Make sure we're not roundtripping a trivial operation
  EXPECT_NE(permute::transpose_from_left(test.P, test.A), test.A);

  const auto actual =
      permute::from_left(test.P, permute::transpose_from_left(test.P, test.A));
  EXPECT_EQ(test.A, actual);
  const auto reverse =
      permute::transpose_from_left(test.P, permute::from_left(test.P, test.A));
  EXPECT_EQ(test.A, reverse);
}

TEST(test_permutation_utils, test_roundtrip_from_right) {
  const auto test = test_data();

  // Make sure we're not roundtripping a trivial operation
  EXPECT_NE(permute::transpose_from_right(test.A, test.P), test.A);

  const auto actual = permute::from_right(
      permute::transpose_from_right(test.A, test.P), test.P);
  EXPECT_EQ(test.A, actual);

  const auto reverse = permute::transpose_from_right(
      permute::from_right(test.A, test.P), test.P);
  EXPECT_EQ(test.A, reverse);
}

// TEST(test_linalg_utils, test_print_eigen_values) {
// }

} // namespace albatross
