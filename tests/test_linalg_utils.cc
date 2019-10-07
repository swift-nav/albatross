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
#include <albatross/utils/LinalgUtils>
#include <gtest/gtest.h>

namespace albatross {

TEST(test_linalg_utils, test_print_eigen_values) {

  Eigen::Index k = 10;
  Eigen::MatrixXd random = Eigen::MatrixXd::Random(k, k);
  random = random * random.transpose();

  std::vector<int> features;
  for (Eigen::Index i = 0; i < k; ++i) {
    features.push_back(i);
  }

  std::ostringstream oss;
  print_small_eigen_directions(random, features, k - 4, &oss);

  print_large_eigen_directions(random, features, k - 4, &oss);
}

} // namespace albatross
