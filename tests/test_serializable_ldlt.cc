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

#include "eigen/serializable_ldlt.h"

namespace albatross {

/*
 * This test makes sure that we can make predictions of what
 * the attenuation of a signal would be at some unobserved location.
 */
TEST(test_scaling_functions, test_predicts) {
  auto part = Eigen::MatrixXd::Random(n, n);
  auto cov = part * part.transpose();
  auto ldlt = cov.ldlt();
  auto information = Eigen::VectorXd::Ones(n);

}

} // namespace albatross
