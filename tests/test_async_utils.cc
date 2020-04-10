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

    std::lock_guard<std::mutex> lock(mu);
    sum += x;
    order_processed.push_back(x);
  };

  async_apply(xs, add_to_sum);

  EXPECT_EQ(sum, std::accumulate(xs.begin(), xs.end(), 0));
  // Make sure the async apply was indeed processed out of order.
  EXPECT_NE(order_processed, xs);
}

TEST(test_async_utils, test_async_apply_with_capture_map) {
  std::map<std::string, int> xs = {{"0", 0}, {"1", 1}, {"2", 2},
                                   {"3", 3}, {"4", 4}, {"5", 5}};

  std::mutex mu;
  int sum = 0.;

  std::vector<int> order_processed;

  auto add_to_sum = [&](const int x) {
    std::this_thread::sleep_for(std::chrono::milliseconds(abs(x - 2)));
    std::lock_guard<std::mutex> lock(mu);
    sum += x;
    order_processed.push_back(x);
  };

  async_apply_map(xs, add_to_sum);

  EXPECT_EQ(sum, 15);
  // Make sure the async apply was indeed processed out of order.
  std::vector<int> map_order = {0, 1, 2, 3, 4, 5};
  EXPECT_NE(order_processed, map_order);
}

TEST(test_async_utils, test_async_apply_with_capture_map_key) {
  std::map<std::string, int> xs = {{"0", 0}, {"1", 1}, {"2", 2},
                                   {"3", 3}, {"4", 4}, {"5", 5}};

  std::mutex mu;
  int sum = 0.;

  std::vector<int> order_processed;

  auto add_to_sum = [&](const std::string &key, const int x) {
    std::this_thread::sleep_for(std::chrono::milliseconds(abs(x - 2)));
    std::lock_guard<std::mutex> lock(mu);
    sum += x;
    order_processed.push_back(x);
    std::cout << key << std::endl;
  };

  async_apply_map(xs, add_to_sum);

  EXPECT_EQ(sum, 15);
  // Make sure the async apply was indeed processed out of order.
  std::vector<int> map_order = {0, 1, 2, 3, 4, 5};
  EXPECT_NE(order_processed, map_order);
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
  const auto expected = apply(inds, slow_process);
  const auto end_direct = std::chrono::system_clock::now();

  EXPECT_EQ(actual, expected);

  EXPECT_GT(end_direct - start_direct, std::chrono::seconds(inds.size() - 1));
}

TEST(test_async_utils, test_async_is_faster_maps) {
  auto slow_square = [&](const int i) {
    const auto start = std::chrono::system_clock::now();
    std::chrono::seconds delay(1);
    while (std::chrono::system_clock::now() - start < delay) {
    };
    return i * i;
  };

  std::map<std::string, int> xs = {{"0", 0}, {"1", 1}, {"2", 2}, {"3", 3}};

  const auto start = std::chrono::system_clock::now();
  const auto actual = async_apply_map(xs, slow_square);
  const auto end = std::chrono::system_clock::now();

  EXPECT_LT(end - start, std::chrono::seconds(xs.size() - 1));
  const double time =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
          .count();
  std::cout << time << std::endl;

  const auto start_direct = std::chrono::system_clock::now();
  const auto expected = apply(xs, slow_square);
  const auto end_direct = std::chrono::system_clock::now();

  for (const auto &x : expected) {
    EXPECT_EQ(actual.at(x.first), x.second);
  }
  EXPECT_GT(end_direct - start_direct, std::chrono::seconds(xs.size() - 1));
  const double time_direct =
      std::chrono::duration_cast<std::chrono::duration<double>>(end_direct -
                                                                start_direct)
          .count();
  std::cout << time_direct << std::endl;
}

TEST(test_async_utils, test_async_is_faster_map_key) {
  auto slow_square = [&](const double key, const int i) {
    const auto start = std::chrono::system_clock::now();
    std::chrono::seconds delay(1);
    while (std::chrono::system_clock::now() - start < delay) {
    };
    return key * i;
  };

  std::map<double, int> xs = {{0., 0}, {1., 1}, {2., 2}, {3., 3}};

  const auto start = std::chrono::system_clock::now();
  const auto actual = async_apply_map(xs, slow_square);
  const auto end = std::chrono::system_clock::now();

  EXPECT_LT(end - start, std::chrono::seconds(xs.size() - 1));
  const double time =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
          .count();
  std::cout << time << std::endl;

  const auto start_direct = std::chrono::system_clock::now();
  const auto expected = apply(xs, slow_square);
  const auto end_direct = std::chrono::system_clock::now();

  for (const auto &x : expected) {
    EXPECT_EQ(actual.at(x.first), x.second);
  }
  EXPECT_GT(end_direct - start_direct, std::chrono::seconds(xs.size() - 1));
  const double time_direct =
      std::chrono::duration_cast<std::chrono::duration<double>>(end_direct -
                                                                start_direct)
          .count();
  std::cout << time_direct << std::endl;
}

}  // namespace albatross
