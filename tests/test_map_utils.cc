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

#include <albatross/Common>
#include <charconv>
#include <gtest/gtest.h>

namespace albatross {

TEST(test_map_utils, test_map_contains) {
  std::map<int, int> test_map = {{1, 2}, {2, 3}, {3, 4}, {6, 7}};

  // map_contain should return true for all keys.
  for (const auto &pair : test_map) {
    EXPECT_TRUE(map_contains(test_map, pair.first));
  }
  // But should return false for ones not included.
  EXPECT_FALSE(map_contains(test_map, -1));

  // Removing a value from the map should cause a False.
  test_map.erase(3);
  EXPECT_FALSE(map_contains(test_map, 3));
}

namespace {

int from_string_view(std::string_view s) {
  int si{};
  auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), si);
  ALBATROSS_ASSERT(ec == std::errc{} && ptr == s.data() + s.size() &&
                   "Invalid number string");
  return si;
}

struct CompareString {
  using is_transparent = void;
  inline bool operator()(std::string_view s, int i) const {
    return from_string_view(s) < i;
  }

  inline bool operator()(int i, std::string_view s) const {
    return i < from_string_view(s);
  }

  inline bool operator()(int i, int j) const { return i < j; }
};

} // namespace

TEST(test_map_utils, test_map_contains_heterogeneous) {
  std::map<int, int, CompareString> test_map = {{1, 2}, {2, 3}, {3, 4}, {6, 7}};

  // map_contain should return true for all keys.
  for (const auto &pair : test_map) {
    EXPECT_TRUE(map_contains(test_map, pair.first));
  }
  using std::literals::string_view_literals::operator""sv;
  // But should return false for ones not included.
  EXPECT_FALSE(map_contains(test_map, "-1"sv));

  // Removing a value from the map should cause a False.
  test_map.erase(3);
  EXPECT_FALSE(map_contains(test_map, "3"));
}

TEST(test_map_utils, map_at_or) {
  std::map<int, int> test_map = {{1, 2}, {2, 3}, {3, 4}, {6, 7}};

  // If a key already exists, the corresponding value should be
  // returned.
  for (const auto &pair : test_map) {
    EXPECT_EQ(map_at_or(test_map, pair.first, -1), pair.second);
  }
  // For a missing key it should return the default int.
  EXPECT_EQ(map_at_or(test_map, -1), 0);
  // If a third argument is present it should use that
  EXPECT_EQ(map_at_or(test_map, -1, 2), 2);
  // For a missing key it should not have modified the original.
  EXPECT_FALSE(map_contains(test_map, -1));
}

bool is_in_vector(const std::vector<int> &vector, int key) {
  for (const int &v : vector) {
    if (key == v) {
      return true;
    }
  }
  return false;
}

TEST(test_map_utils, map_keys) {
  const std::map<int, int> test_map = {{1, 2}, {2, 3}, {3, 4}, {6, 7}};
  const std::vector<int> keys = map_keys(test_map);

  // Any key in the map should be in the vector of keys.
  for (const auto &pair : test_map) {
    EXPECT_TRUE(is_in_vector(keys, pair.first));
  }
  // A key not in the map should not be in the vector
  EXPECT_FALSE(is_in_vector(keys, -1));
  // The length of the map and vector should be the same
  EXPECT_EQ(static_cast<int>(keys.size()), static_cast<int>(test_map.size()));
}

TEST(test_map_utils, map_keys_heterogeneous) {
  const std::map<int, int, CompareString> test_map = {
      {1, 2}, {2, 3}, {3, 4}, {6, 7}};
  const std::vector<int> keys = map_keys(test_map);

  // Any key in the map should be in the vector of keys.
  for (const auto &pair : test_map) {
    EXPECT_TRUE(is_in_vector(keys, pair.first));
  }
  // A key not in the map should not be in the vector
  EXPECT_FALSE(is_in_vector(keys, -1));
  // The length of the map and vector should be the same
  EXPECT_EQ(static_cast<int>(keys.size()), static_cast<int>(test_map.size()));
}

TEST(test_map_utils, map_join) {
  const std::map<int, int> test_map_1 = {{1, 2}, {2, 3}, {3, 4}, {6, 7}};
  const std::map<int, int> test_map_2 = {{1, 2}, {7, 8}, {8, 9}};
  const std::map<int, int> joined_1 = map_join(test_map_1, test_map_2);
  // They have one shared key so the resulting join should be the size of the
  // sum of the two minus one
  EXPECT_EQ(static_cast<int>(joined_1.size()),
            static_cast<int>(test_map_1.size() + test_map_2.size()) - 1);
  // When joining maps, any duplicate keys will be forced to the value from
  // the second argument.
  for (const auto &pair : joined_1) {
    if (map_contains(test_map_2, pair.first)) {
      EXPECT_EQ(test_map_2.at(pair.first), pair.second);
    } else {
      EXPECT_EQ(test_map_1.at(pair.first), pair.second);
    }
  }

  // Reversing the arguments reverses that ^^ behavior.
  const std::map<int, int> joined_2 = map_join(test_map_2, test_map_1);
  // They have one shared key so the resulting join should be the size of the
  // sum of the two minus one
  EXPECT_EQ(static_cast<int>(joined_2.size()),
            static_cast<int>(test_map_1.size() + test_map_2.size()) - 1);
  for (const auto &pair : joined_2) {
    if (map_contains(test_map_1, pair.first)) {
      EXPECT_EQ(test_map_1.at(pair.first), pair.second);
    } else {
      EXPECT_EQ(test_map_2.at(pair.first), pair.second);
    }
  }
}
} // namespace albatross
