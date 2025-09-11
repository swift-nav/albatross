/*
 * Copyright (C) 2022 Swift Navigation Inc.
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

#include <gtest/gtest.h>

namespace albatross {

bool add_one(int *x) {
  (*x) += 1;
  return true;
}

TEST(test_error_handling, test_assert_evaluates) {
  int x = 0;
  ALBATROSS_ASSERT(add_one(&x));
  EXPECT_EQ(x, 1);
}

}  // namespace albatross
