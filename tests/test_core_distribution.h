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

#include <gtest/gtest.h>
#include <albatross/Distribution>

namespace albatross {

void expect_subset_equal(const Eigen::VectorXd &original,
                         const Eigen::VectorXd &actual,
                         const std::vector<Eigen::Index> indices) {
  for (std::size_t i = 0; i < indices.size(); i++) {
    EXPECT_EQ(actual[cast::to_index(i)], original[indices[i]]);
  }
};

void expect_subset_equal(const Eigen::MatrixXd &original,
                         const Eigen::MatrixXd &actual,
                         const std::vector<Eigen::Index> indices) {
  for (std::size_t i = 0; i < indices.size(); i++) {
    for (std::size_t j = 0; j < indices.size(); j++) {
      EXPECT_EQ(actual(cast::to_index(i), cast::to_index(j)),
                original(indices[i], indices[j]));
    }
  }
}

template <typename Scalar, int Size>
void expect_subset_equal(const Eigen::DiagonalMatrix<Scalar, Size> &original,
                         const Eigen::DiagonalMatrix<Scalar, Size> &actual,
                         const std::vector<Eigen::Index> indices) {
  expect_subset_equal(original.diagonal(), actual.diagonal(), indices);
}

template <typename X>
struct DistributionTestCase {
  using RepresentationType = X;
  virtual RepresentationType create() const {
    RepresentationType obj;
    return obj;
  }
};

template <typename Distribution>
struct DistributionTest : public ::testing::Test {
  typedef typename Distribution::RepresentationType Representation;
};

TYPED_TEST_SUITE_P(DistributionTest);
}  // namespace albatross
