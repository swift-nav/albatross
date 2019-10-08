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

#include <albatross/Common>
#include <albatross/Stats>
#include <gtest/gtest.h>

namespace albatross {

TEST(test_stats, test_chi_squared) {

  EXPECT_LT(fabs(chi_squared_cdf(9.260, 23) - 0.005), 1e-5);
  EXPECT_LT(fabs(chi_squared_cdf(38.932, 21) - 0.99), 1e-5);
  EXPECT_LT(fabs(chi_squared_cdf(96.578, 80) - 0.9), 1e-5);
  EXPECT_LT(fabs(chi_squared_cdf(70.065, 100) - 0.01), 1e-5);

  EXPECT_EQ(chi_squared_cdf(0., 0.), 1.);
  EXPECT_LT(chi_squared_cdf(0., 1), 1e-6);
  EXPECT_LT(chi_squared_cdf(0., 2), 1e-6);
  EXPECT_LT(chi_squared_cdf(0., 10), 1e-6);
  EXPECT_LT(chi_squared_cdf(0., 100.), 1e-6);
  EXPECT_LT(chi_squared_cdf(0., INFINITY), 1e-6);

  EXPECT_LT(fabs(chi_squared_cdf(1.e-4, 0.) - 1.), 1e-4);
  EXPECT_LT(fabs(chi_squared_cdf(1., 0.) - 1.), 1e-4);
  EXPECT_LT(fabs(chi_squared_cdf(1000, 100) - 1.), 1e-4);
  EXPECT_LT(fabs(chi_squared_cdf(10000, 100) - 1.), 1e-4);
  EXPECT_LT(fabs(chi_squared_cdf(100000, 100) - 1.), 1e-4);
  EXPECT_LT(fabs(chi_squared_cdf(INFINITY, 1) - 1.), 1e-4);

  EXPECT_TRUE(std::isnan(chi_squared_cdf(-1e-6, 0)));
  EXPECT_TRUE(std::isnan(chi_squared_cdf(-1e-6, 1)));
  EXPECT_TRUE(std::isnan(chi_squared_cdf(-1e-6, 100)));

  EXPECT_TRUE(std::isnan(chi_squared_cdf(NAN, 0)));
  EXPECT_TRUE(std::isnan(chi_squared_cdf(NAN, 1)));
  EXPECT_TRUE(std::isnan(chi_squared_cdf(NAN, NAN)));
}

} // namespace albatross
