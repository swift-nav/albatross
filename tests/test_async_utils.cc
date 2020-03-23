/*
 * Copyright (C) 2020 Swift Navigation Inc.
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

#include <albatross/Indexing>
#include <albatross/utils/AsyncUtils>
#include <albatross/utils/RandomUtils>

#include <chrono>
#include <mutex>
#include <numeric>

namespace albatross {

TEST(test_async_utils, test_async_apply_with_capture) {
  std::vector<int> xs = {0, 1, 2, 3, 4, 5};

  std::mutex mu;
  int sum = 0.;

  std::vector<int> order_processed;

  auto add_to_sum = [&](const int x) {
    std::this_thread::sleep_for(std::chrono::milliseconds(abs(x - 2)));
    mu.lock();
    sum += x;
    order_processed.push_back(x);
    mu.unlock();
  };

  async_apply(xs, add_to_sum);

  EXPECT_EQ(sum, std::accumulate(xs.begin(), xs.end(), 0));
  // Make sure the async apply was indeed processed out of order.
  EXPECT_NE(order_processed, xs);
}

TEST(test_async_utils, test_async_apply_with_return) {
  std::vector<int> xs = {0, 1, 2, 3, 4, 5};
  std::vector<int> order_processed;

  std::mutex mu;
  auto compute_power = [&](const int x) {
    std::this_thread::sleep_for(std::chrono::milliseconds(abs(x - 2)));
    mu.lock();
    order_processed.push_back(x);
    mu.unlock();
    return x * x;
  };

  std::vector<int> expected;
  for (const auto &x : xs) {
    expected.push_back(x * x);
  }

  const auto actual = async_apply(xs, compute_power);

  EXPECT_EQ(expected, actual);
  // Make sure the async apply was indeed processed out of order.
  EXPECT_NE(order_processed, xs);
}

TEST(test_async_utils, test_async_is_faster) {

  auto slow_process = [&](const int i) {
    const auto start = std::chrono::system_clock::now();
    std::chrono::seconds delay(1);
    while (std::chrono::system_clock::now() - start < delay) {
    };
    return i;
  };

  std::vector<int> inds;
  for (std::size_t i = 0; i < 4; ++i) {
    inds.push_back(i);
  }

  const auto start = std::chrono::system_clock::now();
  const auto actual = async_apply(inds, slow_process);
  const auto end = std::chrono::system_clock::now();

  EXPECT_LT(end - start, std::chrono::seconds(inds.size() - 1));

  const auto start_direct = std::chrono::system_clock::now();
  apply(inds, slow_process);
  const auto end_direct = std::chrono::system_clock::now();

  EXPECT_GT(end_direct - start_direct, std::chrono::seconds(inds.size() - 1));
}

} // namespace albatross
