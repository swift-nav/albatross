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
#include "assert.h"
#include "random_utils.h"
#include <gtest/gtest.h>

namespace albatross {

TEST(test_random_utils, randint_without_replacement) {

  int iterations = 10;
  int k = 6;

  std::default_random_engine gen;

  for (int i = 0; i < iterations; i++) {
    for (int n = 0; n <= k; n++) {
      const auto inds = randint_without_replacement(n, i, i + k, gen);
      for (const auto &j : inds) {
        EXPECT_LE(j, i + k);
        EXPECT_GE(j, i);
      }
    }
  }
}

TEST(test_random_utils, randint_without_replacement_full_set) {
  std::default_random_engine gen;
  const auto inds = randint_without_replacement(10, 0, 9, gen);
}

} // namespace albatross
