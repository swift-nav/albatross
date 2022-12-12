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

#include <ThreadPool.h>
#include <gtest/gtest.h>

#include "test_utils.h"

constexpr std::size_t kNumCalls = 10000;

TEST(test_thread_pool, test_threaded_enqueue_dequeue) {
  ThreadPool pool(60);

  std::atomic<std::size_t> shared{0};

  const auto do_it = [&shared](std::size_t value) {
    const auto first = shared.fetch_add(value);
    std::this_thread::sleep_for(std::chrono::nanoseconds(first));
    shared += value;
    return value;
  };

  std::vector<std::future<std::size_t>> results{};

  for (std::size_t i = 0; i < kNumCalls; ++i) {
    results.emplace_back(pool.enqueue(do_it, i));
  }

  const std::size_t total =
      std::accumulate(results.begin(), results.end(), std::size_t{0},
                      [](const auto a, auto &b) { return a + b.get(); });

  // Should have had each value added to it twice.
  EXPECT_EQ(shared.load(), (kNumCalls - 1) * kNumCalls);
  EXPECT_EQ(total, (kNumCalls - 1) * kNumCalls / 2);
}
