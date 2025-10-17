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

  inline int project(int i) const { return i; }

  inline int project(std::string_view s) const { return from_string_view(s); }

  template <typename T, typename U> inline bool operator()(T t, U u) const {
    return project(t) < project(u);
  }
};

struct CompareBackwards {
  using is_transparent = void;

  inline int project(int i) const { return i; }

  inline int project(std::string_view s) const { return from_string_view(s); }

  template <typename T, typename U> inline bool operator()(T t, U u) const {
    return project(t) > project(u);
  }
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
  EXPECT_EQ(keys.size(), test_map.size());
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
  EXPECT_EQ(keys.size(), test_map.size());
}

TEST(test_map_utils, map_join) {
  const std::map<int, int> test_map_1 = {{1, 2}, {2, 3}, {3, 4}, {6, 7}};
  const std::map<int, int> test_map_2 = {{1, 2}, {7, 8}, {8, 9}};
  const std::map<int, int> joined_1 = map_join(test_map_1, test_map_2);
  // They have one shared key so the resulting join should be the size of the
  // sum of the two minus one
  EXPECT_EQ(joined_1.size(),
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
  EXPECT_EQ(joined_2.size(),
            static_cast<int>(test_map_1.size() + test_map_2.size()) - 1);
  for (const auto &pair : joined_2) {
    if (map_contains(test_map_1, pair.first)) {
      EXPECT_EQ(test_map_1.at(pair.first), pair.second);
    } else {
      EXPECT_EQ(test_map_2.at(pair.first), pair.second);
    }
  }
}

TEST(test_map_utils, map_difference) {
  const std::map<int, int> test_map_1 = {{1, 10}, {2, 20}, {3, 30}, {4, 40}};
  const std::map<int, int> test_map_2 = {{2, 200}, {4, 400}, {5, 500}};

  // Difference should contain keys in map_1 that are not in map_2
  const std::map<int, int> diff = map_difference(test_map_1, test_map_2);
  EXPECT_EQ(diff.size(), 2);
  EXPECT_TRUE(map_contains(diff, 1));
  EXPECT_TRUE(map_contains(diff, 3));
  EXPECT_FALSE(map_contains(diff, 2));
  EXPECT_FALSE(map_contains(diff, 4));
  // Values from map_1 should be preserved
  for (const auto &[key, value] : diff) {
    EXPECT_TRUE(map_contains(test_map_1, key));
    EXPECT_EQ(value, test_map_1.at(key));
  }
}

TEST(test_map_utils, map_difference_empty) {
  const std::map<int, int> test_map = {{1, 10}, {2, 20}};
  const std::map<int, int> empty_map;

  // Empty map difference with non-empty should return empty
  const std::map<int, int> diff_1 = map_difference(empty_map, test_map);
  EXPECT_EQ(diff_1.size(), 0);

  // Non-empty difference with empty should return original
  const std::map<int, int> diff_2 = map_difference(test_map, empty_map);
  EXPECT_EQ(diff_2.size(), test_map.size());
  for (const auto &[key, value] : test_map) {
    EXPECT_TRUE(map_contains(diff_2, key));
    EXPECT_EQ(diff_2.at(key), value);
  }
}

TEST(test_map_utils, map_difference_disjoint) {
  const std::map<int, int> test_map_1 = {{1, 10}, {2, 20}};
  const std::map<int, int> test_map_2 = {{3, 30}, {4, 40}};

  // Disjoint maps should return first map unchanged
  const std::map<int, int> diff = map_difference(test_map_1, test_map_2);
  EXPECT_EQ(diff.size(), test_map_1.size());
  for (const auto &[key, value] : test_map_1) {
    EXPECT_TRUE(map_contains(diff, key));
    EXPECT_EQ(diff.at(key), value);
  }
}

TEST(test_map_utils, map_difference_identical) {
  const std::map<int, int> test_map = {{1, 10}, {2, 20}, {3, 30}};

  // Identical maps should return empty
  const std::map<int, int> diff = map_difference(test_map, test_map);
  EXPECT_EQ(diff.size(), 0);
}

TEST(test_map_utils, map_difference_heterogeneous) {
  const std::map<int, int, CompareString> test_map_1 = {
      {1, 10}, {2, 20}, {3, 30}, {4, 40}};
  const std::map<std::string, int> test_map_2 = {
      {"2", 200}, {"4", 400}, {"5", 500}};

  using std::literals::string_view_literals::operator""sv;

  // Difference using heterogeneous lookup with different key types
  // CompareString should compare int keys from test_map_1 to string keys from
  // test_map_2
  const auto diff = map_difference(test_map_1, test_map_2);
  EXPECT_EQ(diff.size(), 2);
  // Check with string_view to ensure `diff` has inherited the
  // comparator from `test_map_1`
  EXPECT_TRUE(map_contains(diff, "1"sv));
  EXPECT_TRUE(map_contains(diff, "3"sv));
  EXPECT_FALSE(map_contains(diff, "2"sv));
  EXPECT_FALSE(map_contains(diff, "4"sv));
  // Values from map_1 should be preserved
  for (const auto &[key, value] : diff) {
    EXPECT_TRUE(map_contains(test_map_1, key));
    EXPECT_EQ(value, test_map_1.at(key));
  }
}

TEST(test_map_utils, map_difference_incompatible_comparators) {
  const std::map<int, int, CompareString> test_map_1 = {
      {1, 10}, {2, 20}, {3, 30}, {4, 40}};
  // CompareBackwards orders in reverse, forcing copy and sort path
  const std::map<int, int, CompareBackwards> test_map_2 = {
      {2, 200}, {4, 400}, {5, 500}};

  using std::literals::string_view_literals::operator""sv;

  // Difference with incompatible comparators should still work correctly
  // This tests the "copy and sort" path
  const auto diff = map_difference(test_map_1, test_map_2);
  EXPECT_EQ(diff.size(), 2);
  EXPECT_TRUE(map_contains(diff, "1"sv));
  EXPECT_TRUE(map_contains(diff, "3"sv));
  EXPECT_FALSE(map_contains(diff, "2"sv));
  EXPECT_FALSE(map_contains(diff, "4"sv));
  // Values from map_1 should be preserved
  for (const auto &[key, value] : diff) {
    EXPECT_TRUE(map_contains(test_map_1, key));
    EXPECT_EQ(value, test_map_1.at(key));
  }
}

TEST(test_map_utils, map_difference_keys) {
  const std::map<int, int> test_map_1 = {{1, 10}, {2, 20}, {3, 30}, {4, 40}};
  const std::map<int, int> test_map_2 = {{2, 200}, {4, 400}, {5, 500}};

  // Difference keys should contain keys in map_1 that are not in map_2
  const std::vector<int> diff_keys =
      map_difference_keys(test_map_1, test_map_2);
  EXPECT_EQ(diff_keys.size(), 2);
  EXPECT_TRUE(is_in_vector(diff_keys, 1));
  EXPECT_TRUE(is_in_vector(diff_keys, 3));
  EXPECT_FALSE(is_in_vector(diff_keys, 2));
  EXPECT_FALSE(is_in_vector(diff_keys, 4));
  // Result should be sorted
  EXPECT_LT(diff_keys[0], diff_keys[1]);
}

TEST(test_map_utils, map_difference_keys_empty) {
  const std::map<int, int> test_map = {{1, 10}, {2, 20}};
  const std::map<int, int> empty_map;

  // Empty map difference should return empty vector
  const std::vector<int> diff_keys_1 = map_difference_keys(empty_map, test_map);
  EXPECT_EQ(diff_keys_1.size(), 0);

  // Difference with empty map should return all keys
  const std::vector<int> diff_keys_2 = map_difference_keys(test_map, empty_map);
  EXPECT_EQ(diff_keys_2.size(), test_map.size());
  for (const auto &[key, _] : test_map) {
    EXPECT_TRUE(is_in_vector(diff_keys_2, key));
  }
}

TEST(test_map_utils, map_difference_keys_disjoint) {
  const std::map<int, int> test_map_1 = {{1, 10}, {2, 20}};
  const std::map<int, int> test_map_2 = {{3, 30}, {4, 40}};

  // Disjoint maps should return all keys from first map
  const std::vector<int> diff_keys =
      map_difference_keys(test_map_1, test_map_2);
  EXPECT_EQ(diff_keys.size(), test_map_1.size());
  for (const auto &[key, _] : test_map_1) {
    EXPECT_TRUE(is_in_vector(diff_keys, key));
  }
}

TEST(test_map_utils, map_difference_keys_heterogeneous) {
  const std::map<int, int, CompareString> test_map_1 = {
      {1, 10}, {2, 20}, {3, 30}, {4, 40}};
  const std::map<std::string, int> test_map_2 = {
      {"2", 200}, {"4", 400}, {"5", 500}};

  // Difference keys using heterogeneous lookup with different key types
  // CompareString should compare int keys from test_map_1 to string keys from
  // test_map_2
  const std::vector<int> diff_keys =
      map_difference_keys(test_map_1, test_map_2);
  EXPECT_EQ(diff_keys.size(), 2);
  EXPECT_TRUE(is_in_vector(diff_keys, 1));
  EXPECT_TRUE(is_in_vector(diff_keys, 3));
  EXPECT_FALSE(is_in_vector(diff_keys, 2));
  EXPECT_FALSE(is_in_vector(diff_keys, 4));
  // Result should be sorted
  EXPECT_LT(diff_keys[0], diff_keys[1]);
}

TEST(test_map_utils, map_difference_keys_incompatible_comparators) {
  const std::map<int, int, CompareString> test_map_1 = {
      {1, 10}, {2, 20}, {3, 30}, {4, 40}};
  // CompareBackwards orders in reverse, forcing copy and sort path
  const std::map<int, int, CompareBackwards> test_map_2 = {
      {2, 200}, {4, 400}, {5, 500}};

  // Difference keys with incompatible comparators should still work correctly
  // This tests the "copy and sort" path
  const std::vector<int> diff_keys =
      map_difference_keys(test_map_1, test_map_2);
  EXPECT_EQ(diff_keys.size(), 2);
  EXPECT_TRUE(is_in_vector(diff_keys, 1));
  EXPECT_TRUE(is_in_vector(diff_keys, 3));
  EXPECT_FALSE(is_in_vector(diff_keys, 2));
  EXPECT_FALSE(is_in_vector(diff_keys, 4));
  // Result should be sorted
  EXPECT_LT(diff_keys[0], diff_keys[1]);
}

TEST(test_map_utils, map_intersect_keys) {
  const std::map<int, int> test_map_1 = {{1, 10}, {2, 20}, {3, 30}, {4, 40}};
  const std::map<int, int> test_map_2 = {{2, 200}, {4, 400}, {5, 500}};

  // Intersection should contain keys present in both maps
  const std::vector<int> intersect_keys =
      map_intersect_keys(test_map_1, test_map_2);
  EXPECT_EQ(intersect_keys.size(), 2);
  EXPECT_TRUE(is_in_vector(intersect_keys, 2));
  EXPECT_TRUE(is_in_vector(intersect_keys, 4));
  EXPECT_FALSE(is_in_vector(intersect_keys, 1));
  EXPECT_FALSE(is_in_vector(intersect_keys, 3));
  EXPECT_FALSE(is_in_vector(intersect_keys, 5));
  // Result should be sorted
  EXPECT_LT(intersect_keys[0], intersect_keys[1]);
}

TEST(test_map_utils, map_intersect_keys_empty) {
  const std::map<int, int> test_map = {{1, 10}, {2, 20}};
  const std::map<int, int> empty_map;

  // Intersection with empty map should return empty vector
  const std::vector<int> intersect_keys_1 =
      map_intersect_keys(test_map, empty_map);
  EXPECT_EQ(intersect_keys_1.size(), 0);

  const std::vector<int> intersect_keys_2 =
      map_intersect_keys(empty_map, test_map);
  EXPECT_EQ(intersect_keys_2.size(), 0);
}

TEST(test_map_utils, map_intersect_keys_disjoint) {
  const std::map<int, int> test_map_1 = {{1, 10}, {2, 20}};
  const std::map<int, int> test_map_2 = {{3, 30}, {4, 40}};

  // Disjoint maps should return empty vector
  const std::vector<int> intersect_keys =
      map_intersect_keys(test_map_1, test_map_2);
  EXPECT_EQ(intersect_keys.size(), 0);
}

TEST(test_map_utils, map_intersect_keys_identical) {
  const std::map<int, int> test_map = {{1, 10}, {2, 20}, {3, 30}};

  // Identical maps should return all keys
  const std::vector<int> intersect_keys =
      map_intersect_keys(test_map, test_map);
  EXPECT_EQ(intersect_keys.size(), test_map.size());
  for (const auto &[key, _] : test_map) {
    EXPECT_TRUE(is_in_vector(intersect_keys, key));
  }
}

TEST(test_map_utils, map_intersect_keys_heterogeneous) {
  const std::map<int, int, CompareString> test_map_1 = {
      {1, 10}, {2, 20}, {3, 30}, {4, 40}};
  const std::map<std::string, int> test_map_2 = {
      {"2", 200}, {"4", 400}, {"5", 500}};

  // Intersection using heterogeneous lookup with different key types
  // CompareString should compare int keys from test_map_1 to string keys from
  // test_map_2
  const std::vector<int> intersect_keys =
      map_intersect_keys(test_map_1, test_map_2);
  EXPECT_EQ(intersect_keys.size(), 2);
  EXPECT_TRUE(is_in_vector(intersect_keys, 2));
  EXPECT_TRUE(is_in_vector(intersect_keys, 4));
  EXPECT_FALSE(is_in_vector(intersect_keys, 1));
  EXPECT_FALSE(is_in_vector(intersect_keys, 3));
  EXPECT_FALSE(is_in_vector(intersect_keys, 5));
  // Result should be sorted
  EXPECT_LT(intersect_keys[0], intersect_keys[1]);
}

TEST(test_map_utils, map_intersect_keys_incompatible_comparators) {
  const std::map<int, int, CompareString> test_map_1 = {
      {1, 10}, {2, 20}, {3, 30}, {4, 40}};
  // CompareBackwards orders in reverse, forcing copy and sort path
  const std::map<int, int, CompareBackwards> test_map_2 = {
      {2, 200}, {4, 400}, {5, 500}};

  // Intersection with incompatible comparators should still work correctly
  // This tests the "copy and sort" path
  const std::vector<int> intersect_keys =
      map_intersect_keys(test_map_1, test_map_2);
  EXPECT_EQ(intersect_keys.size(), 2);
  EXPECT_TRUE(is_in_vector(intersect_keys, 2));
  EXPECT_TRUE(is_in_vector(intersect_keys, 4));
  EXPECT_FALSE(is_in_vector(intersect_keys, 1));
  EXPECT_FALSE(is_in_vector(intersect_keys, 3));
  EXPECT_FALSE(is_in_vector(intersect_keys, 5));
  // Result should be sorted
  EXPECT_LT(intersect_keys[0], intersect_keys[1]);
}

TEST(test_map_utils, map_intersect_subset) {
  std::map<int, int> test_map_1 = {{1, 10}, {2, 20}, {3, 30}, {4, 40}};
  const std::map<int, int> test_map_2 = {{2, 200}, {4, 400}, {5, 500}};

  // map_intersect_subset should modify test_map_1 in place
  map_intersect_subset(test_map_1, test_map_2);
  EXPECT_EQ(test_map_1.size(), 2);
  EXPECT_TRUE(map_contains(test_map_1, 2));
  EXPECT_TRUE(map_contains(test_map_1, 4));
  EXPECT_FALSE(map_contains(test_map_1, 1));
  EXPECT_FALSE(map_contains(test_map_1, 3));
  // Values from original map_1 should be preserved
  EXPECT_EQ(test_map_1.at(2), 20);
  EXPECT_EQ(test_map_1.at(4), 40);
}

TEST(test_map_utils, map_intersect_subset_empty) {
  std::map<int, int> test_map = {{1, 10}, {2, 20}, {3, 30}};
  const std::map<int, int> empty_map;

  // Intersecting with empty map should clear the first map
  map_intersect_subset(test_map, empty_map);
  EXPECT_EQ(test_map.size(), 0);
}

TEST(test_map_utils, map_intersect_subset_disjoint) {
  std::map<int, int> test_map_1 = {{1, 10}, {2, 20}};
  const std::map<int, int> test_map_2 = {{3, 30}, {4, 40}};

  // Disjoint maps should clear the first map
  map_intersect_subset(test_map_1, test_map_2);
  EXPECT_EQ(test_map_1.size(), 0);
}

TEST(test_map_utils, map_intersect_subset_identical) {
  std::map<int, int> test_map_1 = {{1, 10}, {2, 20}, {3, 30}};
  const std::map<int, int> test_map_1_original(test_map_1);
  const std::map<int, int> test_map_2 = {{1, 100}, {2, 200}, {3, 300}};

  // Identical keys should preserve all entries with original values
  map_intersect_subset(test_map_1, test_map_2);
  EXPECT_EQ(test_map_1, test_map_1_original);
}

TEST(test_map_utils, map_intersect_subset_heterogeneous) {
  std::map<int, int, CompareString> test_map_1 = {
      {1, 10}, {2, 20}, {3, 30}, {4, 40}};
  const std::map<std::string, int> test_map_2 = {
      {"2", 200}, {"4", 400}, {"5", 500}};

  using std::literals::string_view_literals::operator""sv;

  // map_intersect_subset with heterogeneous lookup with different key types
  // CompareString should compare int keys from test_map_1 to string keys from
  // test_map_2
  map_intersect_subset(test_map_1, test_map_2);
  EXPECT_EQ(test_map_1.size(), 2);
  // Can check with string_view for heterogeneous lookup
  EXPECT_TRUE(map_contains(test_map_1, "2"sv));
  EXPECT_TRUE(map_contains(test_map_1, "4"sv));
  EXPECT_FALSE(map_contains(test_map_1, "1"sv));
  EXPECT_FALSE(map_contains(test_map_1, "3"sv));
  // Values from original map_1 should be preserved
  EXPECT_EQ(map_at_or(test_map_1, "2"sv), 20);
  EXPECT_EQ(map_at_or(test_map_1, "4"sv), 40);
}

TEST(test_map_utils, map_intersect) {
  const std::map<int, int> test_map_1 = {{1, 10}, {2, 20}, {3, 30}, {4, 40}};
  const std::map<int, int> test_map_2 = {{2, 200}, {4, 400}, {5, 500}};

  // Default match function should create pairs of values
  const auto intersection = map_intersect(test_map_1, test_map_2);
  EXPECT_EQ(intersection.size(), 2);
  EXPECT_TRUE(map_contains(intersection, 2));
  EXPECT_TRUE(map_contains(intersection, 4));
  EXPECT_FALSE(map_contains(intersection, 1));
  EXPECT_FALSE(map_contains(intersection, 3));
  EXPECT_FALSE(map_contains(intersection, 5));
  // Values should be pairs from both maps
  for (const auto &[key, pair] : intersection) {
    const auto &[v1, v2] = pair;
    EXPECT_TRUE(map_contains(test_map_1, key));
    EXPECT_EQ(test_map_1.at(key), v1);
    EXPECT_TRUE(map_contains(test_map_2, key));
    EXPECT_EQ(test_map_2.at(key), v2);
  }
}

TEST(test_map_utils, map_intersect_empty) {
  const std::map<int, int> test_map = {{1, 10}, {2, 20}};
  const std::map<int, int> empty_map;

  // Intersection with empty map should return empty map
  const auto intersection_1 = map_intersect(test_map, empty_map);
  EXPECT_EQ(intersection_1.size(), 0);

  const auto intersection_2 = map_intersect(empty_map, test_map);
  EXPECT_EQ(intersection_2.size(), 0);
}

TEST(test_map_utils, map_intersect_disjoint) {
  const std::map<int, int> test_map_1 = {{1, 10}, {2, 20}};
  const std::map<int, int> test_map_2 = {{3, 30}, {4, 40}};

  // Disjoint maps should return empty map
  const auto intersection = map_intersect(test_map_1, test_map_2);
  EXPECT_EQ(intersection.size(), 0);
}

TEST(test_map_utils, map_intersect_identical) {
  const std::map<int, int> test_map_1 = {{1, 10}, {2, 20}, {3, 30}};
  const std::map<int, int> test_map_2 = {{1, 100}, {2, 200}, {3, 300}};

  // Identical keys should return all keys with paired values
  const auto intersection = map_intersect(test_map_1, test_map_2);
  EXPECT_EQ(intersection.size(), test_map_1.size());
  for (const auto &[key, value] : test_map_1) {
    EXPECT_TRUE(map_contains(intersection, key));
    const auto &[v1, v2] = intersection.at(key);
    EXPECT_EQ(v1, value);
    EXPECT_EQ(v2, test_map_2.at(key));
  }
}

TEST(test_map_utils, map_intersect_custom_match_values_only) {
  const std::map<int, int> test_map_1 = {{1, 10}, {2, 20}, {3, 30}, {4, 40}};
  const std::map<int, int> test_map_2 = {{2, 200}, {4, 400}, {5, 500}};

  const auto intersection =
      map_intersect(test_map_1, test_map_2, std::plus<int>{});

  EXPECT_EQ(intersection.size(), 2);
  EXPECT_TRUE(map_contains(intersection, 2));
  EXPECT_TRUE(map_contains(intersection, 4));
  // Values should be sums
  for (const auto &[key, value] : intersection) {
    EXPECT_EQ(test_map_1.at(key) + test_map_2.at(key), value);
  }
}

TEST(test_map_utils, map_intersect_custom_match_with_key) {
  const std::map<int, int> test_map_1 = {{1, 10}, {2, 20}, {3, 30}, {4, 40}};
  const std::map<int, int> test_map_2 = {{2, 200}, {4, 400}, {5, 500}};

  // Custom match function f(key, v1, v2) that multiplies key by sum of values
  const auto key_times_sum = [](int key, int v1, int v2) {
    return key * (v1 + v2);
  };
  const auto intersection =
      map_intersect(test_map_1, test_map_2, key_times_sum);

  EXPECT_EQ(intersection.size(), 2);
  EXPECT_TRUE(map_contains(intersection, 2));
  EXPECT_TRUE(map_contains(intersection, 4));
  // Values should be key * (v1 + v2)
  for (const auto &[key, value] : intersection) {
    EXPECT_EQ(key * (test_map_1.at(key) + test_map_2.at(key)), value);
  }
}

TEST(test_map_utils, map_intersect_heterogeneous) {
  const std::map<int, int, CompareString> test_map_1 = {
      {1, 10}, {2, 20}, {3, 30}, {4, 40}};
  const std::map<std::string, int> test_map_2 = {
      {"2", 200}, {"4", 400}, {"5", 500}};

  using std::literals::string_view_literals::operator""sv;

  // Intersection with heterogeneous lookup with different key types
  // CompareString should compare int keys from test_map_1 to string keys from
  // test_map_2
  const auto intersection = map_intersect(test_map_1, test_map_2);
  EXPECT_EQ(intersection.size(), 2);
  // Can check with string_view for heterogeneous lookup on result
  EXPECT_TRUE(map_contains(intersection, "2"sv));
  EXPECT_TRUE(map_contains(intersection, "4"sv));
  EXPECT_FALSE(map_contains(intersection, "1"sv));
  EXPECT_FALSE(map_contains(intersection, "3"sv));
  // Values should be pairs from both maps
  for (const auto &[key, pair] : intersection) {
    const auto &[v1, v2] = pair;
    const std::string key_string = std::to_string(key);
    EXPECT_TRUE(map_contains(test_map_1, key_string));
    EXPECT_EQ(test_map_1.at(key), v1);
    EXPECT_TRUE(map_contains(test_map_2, key_string));
    EXPECT_EQ(test_map_2.at(key_string), v2);
  }
}

TEST(test_map_utils, map_intersect_heterogeneous_custom_match_values_only) {
  const std::map<int, int, CompareString> test_map_1 = {
      {1, 10}, {2, 20}, {3, 30}, {4, 40}};
  const std::map<std::string, int> test_map_2 = {
      {"2", 200}, {"4", 400}, {"5", 500}};

  using std::literals::string_view_literals::operator""sv;

  const auto intersection =
      map_intersect(test_map_1, test_map_2, std::plus<>{});

  EXPECT_EQ(intersection.size(), 2);
  EXPECT_TRUE(map_contains(intersection, "2"sv));
  EXPECT_TRUE(map_contains(intersection, "4"sv));
  // Values should be sums
  for (const auto &[key, value] : intersection) {
    const std::string key_string = std::to_string(key);
    EXPECT_EQ(test_map_1.at(key) + test_map_2.at(key_string), value);
  }
}

TEST(test_map_utils, map_intersect_heterogeneous_custom_match_with_key) {
  const std::map<int, int, CompareString> test_map_1 = {
      {1, 10}, {2, 20}, {3, 30}, {4, 40}};
  const std::map<std::string, int> test_map_2 = {
      {"2", 200}, {"4", 400}, {"5", 500}};

  using std::literals::string_view_literals::operator""sv;

  // Custom match function f(key, v1, v2) with heterogeneous lookup
  const auto key_times_sum = [](int key, int v1, int v2) {
    return key * (v1 + v2);
  };
  const auto intersection =
      map_intersect(test_map_1, test_map_2, key_times_sum);

  EXPECT_EQ(intersection.size(), 2);
  EXPECT_TRUE(map_contains(intersection, "2"sv));
  EXPECT_TRUE(map_contains(intersection, "4"sv));
  // Values should be key * (v1 + v2)
  for (const auto &[key, value] : intersection) {
    const std::string key_string = std::to_string(key);
    EXPECT_EQ(key * (test_map_1.at(key) + test_map_2.at(key_string)), value);
  }
}

TEST(test_map_utils, map_intersect_incompatible_comparators) {
  const std::map<int, int, CompareString> test_map_1 = {
      {1, 10}, {2, 20}, {3, 30}, {4, 40}};
  // CompareBackwards orders in reverse, forcing copy and sort path
  const std::map<int, int, CompareBackwards> test_map_2 = {
      {2, 200}, {4, 400}, {5, 500}};

  using std::literals::string_view_literals::operator""sv;

  // Intersection with incompatible comparators should still work correctly
  // This tests the "copy and sort" path with default match function
  const auto intersection = map_intersect(test_map_1, test_map_2);
  EXPECT_EQ(intersection.size(), 2);
  EXPECT_TRUE(map_contains(intersection, "2"sv));
  EXPECT_TRUE(map_contains(intersection, "4"sv));
  EXPECT_FALSE(map_contains(intersection, "1"sv));
  EXPECT_FALSE(map_contains(intersection, "3"sv));
  // Values should be pairs from both maps
  for (const auto &[key, pair] : intersection) {
    const auto &[v1, v2] = pair;
    EXPECT_TRUE(map_contains(test_map_1, key));
    EXPECT_EQ(test_map_1.at(key), v1);
    EXPECT_TRUE(map_contains(test_map_2, key));
    EXPECT_EQ(test_map_2.at(key), v2);
  }
}

TEST(test_map_utils, map_intersect_return_left) {
  const std::map<int, int> test_map_1 = {{1, 10}, {2, 20}, {3, 30}, {4, 40}};
  const std::map<int, int> test_map_2 = {{2, 200}, {4, 400}, {5, 500}};

  // ReturnLeft should return values from first map only
  const auto intersection = map_intersect(test_map_1, test_map_2, ReturnLeft{});
  EXPECT_EQ(intersection.size(), 2);
  EXPECT_TRUE(map_contains(intersection, 2));
  EXPECT_TRUE(map_contains(intersection, 4));
  // Values should be from test_map_1
  for (const auto &[key, value] : intersection) {
    EXPECT_EQ(value, test_map_1.at(key));
  }
}

TEST(test_map_utils, map_intersect_return_right) {
  const std::map<int, int> test_map_1 = {{1, 10}, {2, 20}, {3, 30}, {4, 40}};
  const std::map<int, int> test_map_2 = {{2, 200}, {4, 400}, {5, 500}};

  // ReturnRight should return values from second map only
  const auto intersection =
      map_intersect(test_map_1, test_map_2, ReturnRight{});
  EXPECT_EQ(intersection.size(), 2);
  EXPECT_TRUE(map_contains(intersection, 2));
  EXPECT_TRUE(map_contains(intersection, 4));
  // Values should be from test_map_2
  for (const auto &[key, value] : intersection) {
    EXPECT_EQ(value, test_map_2.at(key));
  }
}

TEST(test_map_utils, map_intersect_return_left_heterogeneous) {
  const std::map<int, int, CompareString> test_map_1 = {
      {1, 10}, {2, 20}, {3, 30}, {4, 40}};
  const std::map<std::string, int> test_map_2 = {
      {"2", 200}, {"4", 400}, {"5", 500}};

  using std::literals::string_view_literals::operator""sv;

  // ReturnLeft with heterogeneous lookup - left-biased intersection
  // CompareString should compare int keys from test_map_1 to string keys from
  // test_map_2
  const auto intersection = map_intersect(test_map_1, test_map_2, ReturnLeft{});
  EXPECT_EQ(intersection.size(), 2);
  // Result should inherit comparator from test_map_1
  EXPECT_TRUE(map_contains(intersection, "2"sv));
  EXPECT_TRUE(map_contains(intersection, "4"sv));
  EXPECT_FALSE(map_contains(intersection, "1"sv));
  EXPECT_FALSE(map_contains(intersection, "3"sv));
  // Values should be from test_map_1 only
  for (const auto &[key, value] : intersection) {
    EXPECT_EQ(value, test_map_1.at(key));
  }
}

TEST(test_map_utils, map_intersect_return_right_heterogeneous) {
  const std::map<int, int, CompareString> test_map_1 = {
      {1, 10}, {2, 20}, {3, 30}, {4, 40}};
  const std::map<std::string, int> test_map_2 = {
      {"2", 200}, {"4", 400}, {"5", 500}};

  using std::literals::string_view_literals::operator""sv;

  // ReturnRight with heterogeneous lookup - right-biased intersection
  // CompareString should compare int keys from test_map_1 to string keys from
  // test_map_2
  const auto intersection =
      map_intersect(test_map_1, test_map_2, ReturnRight{});
  EXPECT_EQ(intersection.size(), 2);
  // Result should inherit comparator from test_map_1
  EXPECT_TRUE(map_contains(intersection, "2"sv));
  EXPECT_TRUE(map_contains(intersection, "4"sv));
  EXPECT_FALSE(map_contains(intersection, "1"sv));
  EXPECT_FALSE(map_contains(intersection, "3"sv));
  // Values should be from test_map_2 only
  for (const auto &[key, value] : intersection) {
    EXPECT_EQ(value, test_map_2.at(std::to_string(key)));
  }
}

TEST(test_map_utils, map_subset_vector) {
  const std::map<int, int> test_map = {{1, 10}, {2, 20}, {3, 30}, {4, 40}};
  const std::vector<int> keys = {2, 4};

  // Subset with vector of keys should return only those associations
  const auto subset = map_subset(test_map, keys);
  EXPECT_EQ(subset.size(), 2);
  EXPECT_TRUE(map_contains(subset, 2));
  EXPECT_TRUE(map_contains(subset, 4));
  EXPECT_FALSE(map_contains(subset, 1));
  EXPECT_FALSE(map_contains(subset, 3));
  // Values should be preserved from original map
  for (const auto &[key, value] : subset) {
    EXPECT_EQ(test_map.at(key), value);
  }
}

TEST(test_map_utils, map_subset_set) {
  const std::map<int, int> test_map = {{1, 10}, {2, 20}, {3, 30}, {4, 40}};
  const std::set<int> keys = {2, 4};

  // Subset with set of keys should return only those associations
  const auto subset = map_subset(test_map, keys);
  EXPECT_EQ(subset.size(), 2);
  EXPECT_TRUE(map_contains(subset, 2));
  EXPECT_TRUE(map_contains(subset, 4));
  EXPECT_FALSE(map_contains(subset, 1));
  EXPECT_FALSE(map_contains(subset, 3));
  // Values should be preserved from original map
  for (const auto &[key, value] : subset) {
    EXPECT_EQ(test_map.at(key), value);
  }
}

TEST(test_map_utils, map_subset_empty_keys) {
  const std::map<int, int> test_map = {{1, 10}, {2, 20}, {3, 30}};
  const std::vector<int> empty_keys;

  // Empty key sequence should return empty map
  const auto subset = map_subset(test_map, empty_keys);
  EXPECT_EQ(subset.size(), 0);
}

TEST(test_map_utils, map_subset_empty_map) {
  const std::map<int, int> empty_map;
  const std::vector<int> keys = {1, 2, 3};

  // Empty map should return empty subset
  const auto subset = map_subset(empty_map, keys);
  EXPECT_EQ(subset.size(), 0);
}

TEST(test_map_utils, map_subset_disjoint) {
  const std::map<int, int> test_map = {{1, 10}, {2, 20}};
  const std::vector<int> keys = {3, 4, 5};

  // Disjoint keys should return empty map
  const auto subset = map_subset(test_map, keys);
  EXPECT_EQ(subset.size(), 0);
}

TEST(test_map_utils, map_subset_superset) {
  const std::map<int, int> test_map = {{1, 10}, {2, 20}, {3, 30}};
  const std::vector<int> keys = {1, 2, 3, 4, 5};

  // Keys superset should return entire map
  const auto subset = map_subset(test_map, keys);
  EXPECT_EQ(subset.size(), test_map.size());
  for (const auto &[key, value] : test_map) {
    EXPECT_TRUE(map_contains(subset, key));
    EXPECT_EQ(subset.at(key), value);
  }
}

TEST(test_map_utils, map_subset_unsorted_vector) {
  const std::map<int, int> test_map = {{1, 10}, {2, 20}, {3, 30}, {4, 40}};
  // Vector with unsorted keys - should still work
  const std::vector<int> keys = {4, 1, 3};

  const auto subset = map_subset(test_map, keys);
  EXPECT_EQ(subset.size(), 3);
  EXPECT_TRUE(map_contains(subset, 1));
  EXPECT_TRUE(map_contains(subset, 3));
  EXPECT_TRUE(map_contains(subset, 4));
  EXPECT_FALSE(map_contains(subset, 2));
  // Values should be preserved from original map
  for (const auto &[key, value] : subset) {
    EXPECT_EQ(test_map.at(key), value);
  }
}

TEST(test_map_utils, map_subset_heterogeneous_vector) {
  const std::map<int, int, CompareString> test_map = {
      {1, 10}, {2, 20}, {3, 30}, {4, 40}};
  // Use vector of strings to select subset via heterogeneous lookup
  const std::vector<std::string> keys = {"2", "4"};

  using std::literals::string_view_literals::operator""sv;

  // CompareString should compare int keys from test_map to string elements
  const auto subset = map_subset(test_map, keys);
  EXPECT_EQ(subset.size(), 2);
  // Result should inherit comparator from test_map
  EXPECT_TRUE(map_contains(subset, "2"sv));
  EXPECT_TRUE(map_contains(subset, "4"sv));
  EXPECT_FALSE(map_contains(subset, "1"sv));
  EXPECT_FALSE(map_contains(subset, "3"sv));
  // Values should be preserved from original map
  for (const auto &[key, value] : subset) {
    EXPECT_EQ(test_map.at(key), value);
  }
}

TEST(test_map_utils, map_subset_heterogeneous_set) {
  const std::map<int, int, CompareString> test_map = {
      {1, 10}, {2, 20}, {3, 30}, {4, 40}};
  // Use set of strings to select subset via heterogeneous lookup
  const std::set<std::string> keys = {"2", "4"};

  using std::literals::string_view_literals::operator""sv;

  // CompareString should compare int keys from test_map to string elements
  const auto subset = map_subset(test_map, keys);
  EXPECT_EQ(subset.size(), 2);
  // Result should inherit comparator from test_map
  EXPECT_TRUE(map_contains(subset, "2"sv));
  EXPECT_TRUE(map_contains(subset, "4"sv));
  EXPECT_FALSE(map_contains(subset, "1"sv));
  EXPECT_FALSE(map_contains(subset, "3"sv));
  // Values should be preserved from original map
  for (const auto &[key, value] : subset) {
    EXPECT_EQ(test_map.at(key), value);
  }
}

TEST(test_map_utils, map_subset_heterogeneous_empty_keys) {
  const std::map<int, int, CompareString> test_map = {
      {1, 10}, {2, 20}, {3, 30}};
  const std::vector<std::string> empty_keys;

  // Empty key sequence should return empty map
  const auto subset = map_subset(test_map, empty_keys);
  EXPECT_EQ(subset.size(), 0);
}

TEST(test_map_utils, map_subset_heterogeneous_disjoint) {
  const std::map<int, int, CompareString> test_map = {{1, 10}, {2, 20}};
  const std::set<std::string> keys = {"3", "4", "5"};

  // Disjoint keys should return empty map
  const auto subset = map_subset(test_map, keys);
  EXPECT_EQ(subset.size(), 0);
}

} // namespace albatross
