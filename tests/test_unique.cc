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

#include <albatross/Indexing>
#include <gtest/gtest.h>

namespace albatross {

TEST(test_unique, test_empty_vector) {
  std::vector<int> empty = {};
  EXPECT_EQ(unique_values(empty).size(), 0);
}

} // namespace albatross
