/*
 * Copyright (C) 2023 Swift Navigation Inc.
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
#include <albatross/Common>

namespace albatross {

TEST(test_vector_utils, test_vector_contains) {
  std::vector<int> test_vector = {1, 3, 5, 7, 11};

  // map_contain should return true for all keys.
  for (const auto &v : test_vector) {
    EXPECT_TRUE(vector_contains(test_vector, v));
  }
  // But should return false for ones not included.
  EXPECT_FALSE(vector_contains(test_vector, -1));
  EXPECT_FALSE(vector_contains(test_vector, 12));
}

TEST(test_vector_utils, test_vector_contains_empty) {
  std::vector<int> empty_vector = {};
  // But should return false for ones not included.
  EXPECT_FALSE(vector_contains(empty_vector, -1));
  EXPECT_FALSE(vector_contains(empty_vector, 12));
}

TEST(test_vector_utils, test_all) {
  EXPECT_TRUE(all({true}));
  EXPECT_TRUE(all({true, true}));
  EXPECT_TRUE(all({}));
  EXPECT_FALSE(all({false}));
  EXPECT_FALSE(all({false, false}));
  EXPECT_FALSE(all({false, true, false}));
  EXPECT_FALSE(all({false, true, true}));
  EXPECT_FALSE(all({true, true, false}));
}

TEST(test_vector_utils, test_any) {
  EXPECT_TRUE(any({true}));
  EXPECT_TRUE(any({true, true}));
  EXPECT_FALSE(any({}));
  EXPECT_FALSE(any({false}));
  EXPECT_FALSE(any({false, false}));
  EXPECT_TRUE(any({false, true, false}));
  EXPECT_TRUE(any({false, true, true}));
  EXPECT_TRUE(any({true, true, false}));
}

}  // namespace albatross
