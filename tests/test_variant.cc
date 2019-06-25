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
#include <gtest/gtest.h>

namespace albatross {

TEST(test_variant, test_details) {

  const auto one =
      cereal::mapbox_variant_detail::variant_size<variant<int>>::value;
  EXPECT_EQ(one, 1);

  const auto two =
      cereal::mapbox_variant_detail::variant_size<variant<int, double>>::value;
  EXPECT_EQ(two, 2);

  struct X {};
  const auto three = cereal::mapbox_variant_detail::variant_size<
      variant<int, double, X>>::value;
  EXPECT_EQ(three, 3);
}

} // namespace albatross
