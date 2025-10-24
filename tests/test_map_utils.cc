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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace albatross {

// Bring commonly used matchers into scope
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

namespace {

// Helper to convert string_view to int
int from_string_view(std::string_view s) {
  int si{};
  auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), si);
  ALBATROSS_ASSERT(ec == std::errc{} && ptr == s.data() + s.size() &&
                   "Invalid number string");
  return si;
}

// Transparent comparator for heterogeneous lookup
struct CompareString {
  using is_transparent = void;

  inline int project(int i) const { return i; }

  inline int project(std::string_view s) const { return from_string_view(s); }

  template <typename T, typename U> inline bool operator()(T t, U u) const {
    return project(t) < project(u);
  }
};

struct CompareLongInt {
  using is_transparent = void;
  template <typename T, typename U>
  constexpr bool operator()(T a, U b) const noexcept {
    return static_cast<long>(a) < static_cast<long>(b);
  }
};

// Configuration for homogeneous map tests (same key types)
struct HomogeneousMapConfig {
  using Map1 = std::map<int, int>;
  using Map2 = std::map<int, int>;
  using Key1 = int;
  using Key2 = int;

  static constexpr bool is_heterogeneous = false;

  static Map1 make_map_1() { return {{1, 10}, {2, 20}, {3, 30}, {4, 40}}; }
  static Map2 make_map_2() { return {{2, 200}, {4, 400}, {5, 500}}; }

  // Convert Key1 to Key2 (identity for homogeneous)
  static Key2 convert_key(Key1 k) { return k; }
};

// Configuration for heterogeneous map tests (different key types, same
// comparator)
struct HeterogeneousMapConfig {
  using Map1 = std::map<int, int, CompareString>;
  using Map2 = std::map<std::string, int, CompareString>;
  using Key1 = int;
  using Key2 = std::string;

  static constexpr bool is_heterogeneous = true;

  static Map1 make_map_1() { return {{1, 10}, {2, 20}, {3, 30}, {4, 40}}; }
  static Map2 make_map_2() { return {{"2", 200}, {"4", 400}, {"5", 500}}; }

  // Convert Key1 (int) to Key2 (string)
  static Key2 convert_key(Key1 k) { return std::to_string(k); }
};

} // namespace

// Test fixture with common helpers (can be extended in future phases)
class MapUtilsTestBase : public ::testing::Test {
protected:
  // Future: add common test data, helper methods, etc.
};

// Typed test fixture for testing operations with different map configurations
template <typename ConfigType>
class MapOperationsTest : public ::testing::Test {
protected:
  using Map1 = typename ConfigType::Map1;
  using Map2 = typename ConfigType::Map2;
  using Key1 = typename ConfigType::Key1;
  using Key2 = typename ConfigType::Key2;

  ConfigType config;
};

using MapConfigTypes =
    ::testing::Types<HomogeneousMapConfig, HeterogeneousMapConfig>;
TYPED_TEST_SUITE_P(MapOperationsTest);

// Typed test for map_difference (covers both homogeneous and heterogeneous)
TYPED_TEST_P(MapOperationsTest, Difference) {
  auto map1 = TestFixture::config.make_map_1();
  auto map2 = TestFixture::config.make_map_2();

  auto diff = map_difference(map1, map2);

  EXPECT_EQ(diff.size(), 2);

  // All values in `diff` should come from `map1`
  for (const auto &[key, value] : diff) {
    SCOPED_TRACE(::testing::Message() << "Checking key=" << key);
    EXPECT_TRUE(map_contains(map1, key));
    EXPECT_EQ(value, map1.at(key));
  }

  // Additional heterogeneous-specific checks
  if constexpr (TypeParam::is_heterogeneous) {
    using std::literals::string_view_literals::operator""sv;
    // Verify transparent comparison works on result
    EXPECT_TRUE(map_contains(diff, "1"sv));
    EXPECT_TRUE(map_contains(diff, "3"sv));
    EXPECT_FALSE(map_contains(diff, "2"sv));
    EXPECT_FALSE(map_contains(diff, "4"sv));
  } else {
    EXPECT_TRUE(map_contains(diff, 1));
    EXPECT_TRUE(map_contains(diff, 3));
    EXPECT_FALSE(map_contains(diff, 2));
    EXPECT_FALSE(map_contains(diff, 4));
  }
}

// Typed test for map_difference_keys
TYPED_TEST_P(MapOperationsTest, DifferenceKeys) {
  auto map1 = TestFixture::config.make_map_1();
  auto map2 = TestFixture::config.make_map_2();

  auto diff_keys = map_difference_keys(map1, map2);

  EXPECT_THAT(diff_keys, ElementsAre(1, 3));
}

// Typed test for map_intersect_keys
TYPED_TEST_P(MapOperationsTest, IntersectKeys) {
  auto map1 = TestFixture::config.make_map_1();
  auto map2 = TestFixture::config.make_map_2();

  auto intersect_keys = map_intersect_keys(map1, map2);

  EXPECT_THAT(intersect_keys, ElementsAre(2, 4));
}

// Typed test for map_intersect with default MakePair
TYPED_TEST_P(MapOperationsTest, IntersectWithMakePair) {
  auto map1 = TestFixture::config.make_map_1();
  auto map2 = TestFixture::config.make_map_2();

  auto intersection = map_intersect(map1, map2);

  EXPECT_THAT(map_keys(intersection), ElementsAre(2, 4));

  // Values should be pairs from both maps
  for (const auto &[key, pair] : intersection) {
    SCOPED_TRACE(::testing::Message() << "Checking key=" << key);
    const auto &[v1, v2] = pair;
    auto key2 = TestFixture::config.convert_key(key);
    EXPECT_EQ(v1, map1.at(key));
    EXPECT_EQ(v2, map2.at(key2));
  }

  // Additional heterogeneous-specific checks
  if constexpr (TypeParam::is_heterogeneous) {
    using std::literals::string_view_literals::operator""sv;
    EXPECT_TRUE(map_contains(intersection, "2"sv));
    EXPECT_TRUE(map_contains(intersection, "4"sv));
    EXPECT_FALSE(map_contains(intersection, "1"sv));
    EXPECT_FALSE(map_contains(intersection, "3"sv));
  }
}

// Typed test for map_intersect with custom merge (std::plus)
TYPED_TEST_P(MapOperationsTest, IntersectWithCustomMerge) {
  auto map1 = TestFixture::config.make_map_1();
  auto map2 = TestFixture::config.make_map_2();

  auto intersection = map_intersect(map1, map2, std::plus<>{});

  EXPECT_THAT(map_keys(intersection), ElementsAre(2, 4));

  // Values should be sums
  for (const auto &[key, value] : intersection) {
    SCOPED_TRACE(::testing::Message() << "Checking key=" << key);
    auto key2 = TestFixture::config.convert_key(key);
    EXPECT_EQ(value, map1.at(key) + map2.at(key2));
  }

  // Additional heterogeneous-specific checks
  if constexpr (TypeParam::is_heterogeneous) {
    using std::literals::string_view_literals::operator""sv;
    EXPECT_TRUE(map_contains(intersection, "2"sv));
    EXPECT_TRUE(map_contains(intersection, "4"sv));
  }
}

// Typed test for map_intersect with key-based custom merge
TYPED_TEST_P(MapOperationsTest, IntersectWithKeyBasedMerge) {
  auto map1 = TestFixture::config.make_map_1();
  auto map2 = TestFixture::config.make_map_2();

  const auto key_times_sum = [](int key, int v1, int v2) {
    return key * (v1 + v2);
  };
  auto intersection = map_intersect(map1, map2, key_times_sum);

  EXPECT_THAT(map_keys(intersection), ElementsAre(2, 4));

  // Values should be key * (v1 + v2)
  for (const auto &[key, value] : intersection) {
    SCOPED_TRACE(::testing::Message() << "Checking key=" << key);
    auto key2 = TestFixture::config.convert_key(key);
    EXPECT_EQ(value, key * (map1.at(key) + map2.at(key2)));
  }

  // Additional heterogeneous-specific checks
  if constexpr (TypeParam::is_heterogeneous) {
    using std::literals::string_view_literals::operator""sv;
    EXPECT_TRUE(map_contains(intersection, "2"sv));
    EXPECT_TRUE(map_contains(intersection, "4"sv));
  }
}

// Typed test for map_intersect with ReturnLeft
TYPED_TEST_P(MapOperationsTest, IntersectReturnLeft) {
  auto map1 = TestFixture::config.make_map_1();
  auto map2 = TestFixture::config.make_map_2();

  auto intersection = map_intersect(map1, map2, ReturnLeft{});

  EXPECT_EQ(intersection.size(), 2);

  // Values should be from map1 only
  for (const auto &[key, value] : intersection) {
    EXPECT_EQ(value, map1.at(key));
  }

  // Additional heterogeneous-specific checks
  if constexpr (TypeParam::is_heterogeneous) {
    using std::literals::string_view_literals::operator""sv;
    EXPECT_TRUE(map_contains(intersection, "2"sv));
    EXPECT_TRUE(map_contains(intersection, "4"sv));
    EXPECT_FALSE(map_contains(intersection, "1"sv));
    EXPECT_FALSE(map_contains(intersection, "3"sv));
  }
}

// Typed test for map_intersect with ReturnRight
TYPED_TEST_P(MapOperationsTest, IntersectReturnRight) {
  auto map1 = TestFixture::config.make_map_1();
  auto map2 = TestFixture::config.make_map_2();

  auto intersection = map_intersect(map1, map2, ReturnRight{});

  EXPECT_EQ(intersection.size(), 2);

  // Values should be from map2 only
  for (const auto &[key, value] : intersection) {
    auto key2 = TestFixture::config.convert_key(key);
    EXPECT_EQ(value, map2.at(key2));
  }

  // Additional heterogeneous-specific checks
  if constexpr (TypeParam::is_heterogeneous) {
    using std::literals::string_view_literals::operator""sv;
    EXPECT_TRUE(map_contains(intersection, "2"sv));
    EXPECT_TRUE(map_contains(intersection, "4"sv));
    EXPECT_FALSE(map_contains(intersection, "1"sv));
    EXPECT_FALSE(map_contains(intersection, "3"sv));
  }
}

// Register all typed tests
REGISTER_TYPED_TEST_SUITE_P(MapOperationsTest, Difference, DifferenceKeys,
                            IntersectKeys, IntersectWithMakePair,
                            IntersectWithCustomMerge,
                            IntersectWithKeyBasedMerge, IntersectReturnLeft,
                            IntersectReturnRight);

// Instantiate typed tests with both configurations
INSTANTIATE_TYPED_TEST_SUITE_P(MapUtils, MapOperationsTest, MapConfigTypes);

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

TEST(test_map_utils, map_keys) {
  const std::map<int, int> test_map = {{1, 2}, {2, 3}, {3, 4}, {6, 7}};
  const std::vector<int> keys = map_keys(test_map);

  // Keys should contain exactly the map keys
  EXPECT_THAT(keys, UnorderedElementsAre(1, 2, 3, 6));
}

TEST(test_map_utils, map_keys_heterogeneous) {
  const std::map<int, int, CompareString> test_map = {
      {1, 2}, {2, 3}, {3, 4}, {6, 7}};
  const std::vector<int> keys = map_keys(test_map);

  // Keys should contain exactly the map keys
  EXPECT_THAT(keys, UnorderedElementsAre(1, 2, 3, 6));
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

struct MapDifferenceTestCase {
  std::string name;
  std::map<int, int> input_a;
  std::map<int, int> input_b;
  std::map<int, int> expected;
};

class MapDifferenceTest
    : public ::testing::TestWithParam<MapDifferenceTestCase> {};

TEST_P(MapDifferenceTest, ProducesExpectedResult) {
  const auto &tc = GetParam();
  auto result = map_difference(tc.input_a, tc.input_b);
  EXPECT_EQ(result, tc.expected);
}

INSTANTIATE_TEST_SUITE_P(
    CornerCases, MapDifferenceTest,
    ::testing::Values(
        MapDifferenceTestCase{"EmptyA", {}, {{1, 10}, {2, 20}}, {}},
        MapDifferenceTestCase{
            "EmptyB", {{1, 10}, {2, 20}}, {}, {{1, 10}, {2, 20}}},
        MapDifferenceTestCase{"Disjoint",
                              {{1, 10}, {2, 20}},
                              {{3, 30}, {4, 40}},
                              {{1, 10}, {2, 20}}},
        MapDifferenceTestCase{"Identical",
                              {{1, 10}, {2, 20}, {3, 30}},
                              {{1, 10}, {2, 20}, {3, 30}},
                              {}}),
    [](const auto &info) { return info.param.name; });

TEST(test_map_utils, map_symmetric_difference) {
  const std::map<int, int> test_map_1 = {{1, 10}, {2, 20}, {3, 30}};
  const std::map<int, int> test_map_2 = {{2, 200}, {4, 400}, {5, 500}};

  // Symmetric difference should contain keys in either map but not both
  const auto sym_diff = map_symmetric_difference(test_map_1, test_map_2);

  EXPECT_THAT(map_keys(sym_diff), UnorderedElementsAre(1, 3, 4, 5));
  EXPECT_FALSE(map_contains(sym_diff, 2));

  // Values from map_1 for keys only in map_1
  EXPECT_EQ(sym_diff.at(1), 10);
  EXPECT_EQ(sym_diff.at(3), 30);
  // Values from map_2 for keys only in map_2
  EXPECT_EQ(sym_diff.at(4), 400);
  EXPECT_EQ(sym_diff.at(5), 500);
}

// Parameterized tests for map_symmetric_difference corner cases
struct MapSymmetricDifferenceTestCase {
  std::string name;
  std::map<int, int> input_a;
  std::map<int, int> input_b;
  std::map<int, int> expected;
};

class MapSymmetricDifferenceTest
    : public ::testing::TestWithParam<MapSymmetricDifferenceTestCase> {};

TEST_P(MapSymmetricDifferenceTest, ProducesExpectedResult) {
  const auto &tc = GetParam();
  auto result = map_symmetric_difference(tc.input_a, tc.input_b);
  EXPECT_EQ(result, tc.expected);
}

INSTANTIATE_TEST_SUITE_P(
    CornerCases, MapSymmetricDifferenceTest,
    ::testing::Values(
        MapSymmetricDifferenceTestCase{
            "EmptyA", {}, {{1, 10}, {2, 20}}, {{1, 10}, {2, 20}}},
        MapSymmetricDifferenceTestCase{
            "EmptyB", {{1, 10}, {2, 20}}, {}, {{1, 10}, {2, 20}}},
        MapSymmetricDifferenceTestCase{"Disjoint",
                                       {{1, 10}, {2, 20}},
                                       {{3, 30}, {4, 40}},
                                       {{1, 10}, {2, 20}, {3, 30}, {4, 40}}},
        MapSymmetricDifferenceTestCase{"Identical",
                                       {{1, 10}, {2, 20}, {3, 30}},
                                       {{1, 10}, {2, 20}, {3, 30}},
                                       {}}),
    [](const auto &info) { return info.param.name; });

TEST(test_map_utils, map_symmetric_difference_heterogeneous) {
  using std::literals::string_view_literals::operator""sv;

  const std::map<int, int, CompareString> test_map_1 = {
      {1, 10}, {2, 20}, {3, 30}};
  // Both maps use CompareString, same key and value types
  const std::map<int, int, CompareString> test_map_2 = {{2, 200}, {4, 400}};

  // Symmetric difference with same comparator
  const auto sym_diff = map_symmetric_difference(test_map_1, test_map_2);
  EXPECT_EQ(sym_diff.size(), 3);
  EXPECT_TRUE(map_contains(sym_diff, "1"sv));
  EXPECT_TRUE(map_contains(sym_diff, "3"sv));
  EXPECT_TRUE(map_contains(sym_diff, "4"sv));
  EXPECT_FALSE(map_contains(sym_diff, "2"sv));
  // Verify values
  EXPECT_EQ(map_at_or(sym_diff, "1"sv), 10);
  EXPECT_EQ(map_at_or(sym_diff, "3"sv), 30);
  EXPECT_EQ(map_at_or(sym_diff, "4"sv), 400);
}

TEST(test_map_utils, map_symmetric_difference_heterogeneous_key_conversion) {
  // Use long as Key2 type - implicitly convertible to int
  const std::map<int, int, CompareLongInt> test_map_1 = {{1, 10}, {3, 30}};
  // Map with long keys that will be converted to int
  const std::map<long, int, CompareLongInt> test_map_2 = {{2L, 200}, {4L, 400}};

  // Symmetric difference with key conversion (long → int)
  const auto sym_diff = map_symmetric_difference(test_map_1, test_map_2);

  // Should contain {1, 2, 3, 4} all with int keys
  EXPECT_EQ(sym_diff.size(), 4);
  EXPECT_TRUE(map_contains(sym_diff, 1));
  EXPECT_TRUE(map_contains(sym_diff, 2)); // Converted from long!
  EXPECT_TRUE(map_contains(sym_diff, 3));
  EXPECT_TRUE(map_contains(sym_diff, 4)); // Converted from long!

  // Verify values
  EXPECT_EQ(sym_diff.at(1), 10);  // From map_1
  EXPECT_EQ(sym_diff.at(2), 200); // From map_2 (converted key)
  EXPECT_EQ(sym_diff.at(3), 30);  // From map_1
  EXPECT_EQ(sym_diff.at(4), 400); // From map_2 (converted key)
}

struct MapDifferenceKeysTestCase {
  std::string name;
  std::map<int, int> input_a;
  std::map<int, int> input_b;
  std::vector<int> expected_keys;
};

class MapDifferenceKeysTest
    : public ::testing::TestWithParam<MapDifferenceKeysTestCase> {};

TEST_P(MapDifferenceKeysTest, ProducesExpectedKeys) {
  const auto &tc = GetParam();
  auto result = map_difference_keys(tc.input_a, tc.input_b);
  EXPECT_THAT(result, ElementsAreArray(tc.expected_keys));
}

INSTANTIATE_TEST_SUITE_P(
    CornerCases, MapDifferenceKeysTest,
    ::testing::Values(
        MapDifferenceKeysTestCase{"EmptyA", {}, {{1, 10}, {2, 20}}, {}},
        MapDifferenceKeysTestCase{"EmptyB", {{1, 10}, {2, 20}}, {}, {1, 2}},
        MapDifferenceKeysTestCase{
            "Disjoint", {{1, 10}, {2, 20}}, {{3, 30}, {4, 40}}, {1, 2}},
        MapDifferenceKeysTestCase{"Overlapping",
                                  {{1, 10}, {2, 20}, {3, 30}, {4, 40}},
                                  {{2, 200}, {4, 400}, {5, 500}},
                                  {1, 3}}),
    [](const auto &info) { return info.param.name; });

struct MapIntersectKeysTestCase {
  std::string name;
  std::map<int, int> input_a;
  std::map<int, int> input_b;
  std::vector<int> expected_keys;
};

class MapIntersectKeysTest
    : public ::testing::TestWithParam<MapIntersectKeysTestCase> {};

TEST_P(MapIntersectKeysTest, ProducesExpectedKeys) {
  const auto &tc = GetParam();
  auto result = map_intersect_keys(tc.input_a, tc.input_b);
  EXPECT_THAT(result, ElementsAreArray(tc.expected_keys));
}

INSTANTIATE_TEST_SUITE_P(
    CornerCases, MapIntersectKeysTest,
    ::testing::Values(
        MapIntersectKeysTestCase{"EmptyA", {}, {{1, 10}, {2, 20}}, {}},
        MapIntersectKeysTestCase{"EmptyB", {{1, 10}, {2, 20}}, {}, {}},
        MapIntersectKeysTestCase{
            "Disjoint", {{1, 10}, {2, 20}}, {{3, 30}, {4, 40}}, {}},
        MapIntersectKeysTestCase{"Identical",
                                 {{1, 10}, {2, 20}, {3, 30}},
                                 {{1, 10}, {2, 20}, {3, 30}},
                                 {1, 2, 3}},
        MapIntersectKeysTestCase{"Overlapping",
                                 {{1, 10}, {2, 20}, {3, 30}, {4, 40}},
                                 {{2, 200}, {4, 400}, {5, 500}},
                                 {2, 4}}),
    [](const auto &info) { return info.param.name; });

struct MapIntersectTestCase {
  std::string name;
  std::map<int, int> input_a;
  std::map<int, int> input_b;
  std::vector<int> expected_keys;
};

class MapIntersectTest : public ::testing::TestWithParam<MapIntersectTestCase> {
};

TEST_P(MapIntersectTest, ProducesExpectedKeysWithPairs) {
  const auto &tc = GetParam();
  auto result = map_intersect(tc.input_a, tc.input_b);

  EXPECT_THAT(map_keys(result), ElementsAreArray(tc.expected_keys));

  // Verify paired values for non-empty results
  for (const auto &[key, pair] : result) {
    SCOPED_TRACE(::testing::Message() << "Checking key=" << key);
    const auto &[v1, v2] = pair;
    EXPECT_EQ(v1, tc.input_a.at(key));
    EXPECT_EQ(v2, tc.input_b.at(key));
  }
}

INSTANTIATE_TEST_SUITE_P(
    CornerCases, MapIntersectTest,
    ::testing::Values(
        MapIntersectTestCase{"EmptyA", {}, {{1, 10}, {2, 20}}, {}},
        MapIntersectTestCase{"EmptyB", {{1, 10}, {2, 20}}, {}, {}},
        MapIntersectTestCase{
            "Disjoint", {{1, 10}, {2, 20}}, {{3, 30}, {4, 40}}, {}},
        MapIntersectTestCase{"Identical",
                             {{1, 10}, {2, 20}, {3, 30}},
                             {{1, 100}, {2, 200}, {3, 300}},
                             {1, 2, 3}},
        MapIntersectTestCase{"Overlapping",
                             {{1, 10}, {2, 20}, {3, 30}, {4, 40}},
                             {{2, 200}, {4, 400}, {5, 500}},
                             {2, 4}}),
    [](const auto &info) { return info.param.name; });

struct MapSubsetSortedTestCase {
  std::string name;
  std::map<int, int> input_map;
  std::vector<int> keys;
  std::map<int, int> expected;
};

class MapSubsetSortedTest
    : public ::testing::TestWithParam<MapSubsetSortedTestCase> {};

TEST_P(MapSubsetSortedTest, ProducesExpectedSubset) {
  const auto &tc = GetParam();
  auto result = map_subset_sorted(tc.input_map, tc.keys);
  EXPECT_EQ(result, tc.expected);
}

INSTANTIATE_TEST_SUITE_P(
    CornerCases, MapSubsetSortedTest,
    ::testing::Values(
        MapSubsetSortedTestCase{
            "EmptyKeys", {{1, 10}, {2, 20}, {3, 30}}, {}, {}},
        MapSubsetSortedTestCase{"EmptyMap", {}, {1, 2, 3}, {}},
        MapSubsetSortedTestCase{"Disjoint", {{1, 10}, {2, 20}}, {3, 4, 5}, {}},
        MapSubsetSortedTestCase{"Superset",
                                {{1, 10}, {2, 20}, {3, 30}},
                                {1, 2, 3, 4, 5},
                                {{1, 10}, {2, 20}, {3, 30}}},
        MapSubsetSortedTestCase{"Subset",
                                {{1, 10}, {2, 20}, {3, 30}, {4, 40}},
                                {2, 4},
                                {{2, 20}, {4, 40}}},
        MapSubsetSortedTestCase{"Sorted",
                                {{1, 10}, {2, 20}, {3, 30}, {4, 40}},
                                {1, 3, 4},
                                {{1, 10}, {3, 30}, {4, 40}}}),
    [](const auto &info) { return info.param.name; });

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

TEST(test_map_utils, map_subset_sorted_heterogeneous_vector) {
  const std::map<int, int, CompareString> test_map = {
      {1, 10}, {2, 20}, {3, 30}, {4, 40}};
  // Use vector of strings - must be sorted according to CompareString
  const std::vector<std::string> keys = {"2", "4"};

  using std::literals::string_view_literals::operator""sv;

  // CompareString can compare int keys from test_map to string elements
  const auto subset = map_subset_sorted(test_map, keys);
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
  // Use set of strings with same comparator
  const std::set<std::string, CompareString> keys = {"2", "4"};

  using std::literals::string_view_literals::operator""sv;

  // CompareString can compare int keys from test_map to string elements
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

TEST(test_map_utils, map_subset_sorted_heterogeneous_empty_keys) {
  const std::map<int, int, CompareString> test_map = {
      {1, 10}, {2, 20}, {3, 30}};
  const std::vector<std::string> empty_keys;

  // Empty key sequence should return empty map
  const auto subset = map_subset_sorted(test_map, empty_keys);
  EXPECT_EQ(subset.size(), 0);
}

TEST(test_map_utils, map_subset_heterogeneous_disjoint) {
  const std::map<int, int, CompareString> test_map = {{1, 10}, {2, 20}};
  const std::set<std::string, CompareString> keys = {"3", "4", "5"};

  // Disjoint keys should return empty map
  const auto subset = map_subset(test_map, keys);
  EXPECT_EQ(subset.size(), 0);
}

// Death tests for assertion failures
TEST(MapUtilsDeathTest, map_subset_sorted_unsorted_input) {
  const std::map<int, int> test_map = {{1, 10}, {2, 20}, {3, 30}};
  std::vector<int> unsorted_keys = {2, 1, 3}; // Not sorted!

  EXPECT_DEATH(map_subset_sorted(test_map, unsorted_keys), "promised.*sorted");
}

TEST(MapUtilsDeathTest, map_subset_sorted_unsorted_heterogeneous) {
  const std::map<int, int, CompareString> test_map = {
      {1, 10}, {2, 20}, {3, 30}};
  // Keys not sorted according to CompareString
  std::vector<std::string> unsorted_keys = {"3", "1", "2"};

  EXPECT_DEATH(map_subset_sorted(test_map, unsorted_keys), "promised.*sorted");
}

TEST(test_map_utils, map_union) {
  const std::map<int, int> test_map_1 = {{1, 10}, {2, 20}, {3, 30}};
  const std::map<int, int> test_map_2 = {{2, 200}, {4, 400}, {5, 500}};

  // Default merge (ReturnLeft) should keep values from first map on overlap
  const auto result = map_union(test_map_1, test_map_2);
  EXPECT_EQ(result.size(), 5);
  EXPECT_TRUE(map_contains(result, 1));
  EXPECT_TRUE(map_contains(result, 2));
  EXPECT_TRUE(map_contains(result, 3));
  EXPECT_TRUE(map_contains(result, 4));
  EXPECT_TRUE(map_contains(result, 5));
  // All keys from first map should have their original values
  for (const auto &[key, value] : test_map_1) {
    EXPECT_EQ(result.at(key), value);
  }
  // Unique keys from second map should have their values
  EXPECT_EQ(result.at(4), 400);
  EXPECT_EQ(result.at(5), 500);
}

// Parameterized tests for map_union corner cases (with default ReturnLeft)
struct MapUnionTestCase {
  std::string name;
  std::map<int, int> input_a;
  std::map<int, int> input_b;
  std::map<int, int> expected;
};

class MapUnionTest : public ::testing::TestWithParam<MapUnionTestCase> {};

TEST_P(MapUnionTest, ProducesExpectedResult) {
  const auto &tc = GetParam();
  auto result = map_union(tc.input_a, tc.input_b);
  EXPECT_EQ(result, tc.expected);
}

INSTANTIATE_TEST_SUITE_P(
    CornerCases, MapUnionTest,
    ::testing::Values(
        MapUnionTestCase{"EmptyA", {}, {{1, 10}, {2, 20}}, {{1, 10}, {2, 20}}},
        MapUnionTestCase{"EmptyB", {{1, 10}, {2, 20}}, {}, {{1, 10}, {2, 20}}},
        MapUnionTestCase{"Disjoint",
                         {{1, 10}, {2, 20}},
                         {{3, 30}, {4, 40}},
                         {{1, 10}, {2, 20}, {3, 30}, {4, 40}}},
        MapUnionTestCase{"Overlapping_ReturnLeft",
                         {{1, 10}, {2, 20}, {3, 30}},
                         {{2, 200}, {4, 400}, {5, 500}},
                         {{1, 10}, {2, 20}, {3, 30}, {4, 400}, {5, 500}}}),
    [](const auto &info) { return info.param.name; });

TEST(test_map_utils, map_union_return_right) {
  const std::map<int, int> test_map_1 = {{1, 10}, {2, 20}, {3, 30}};
  const std::map<int, int> test_map_2 = {{2, 200}, {4, 400}, {5, 500}};

  // ReturnRight should keep values from second map on overlap
  const auto result = map_union(test_map_1, test_map_2, ReturnRight{});
  EXPECT_EQ(result.size(), 5);
  // All keys from second map should have their original values
  for (const auto &[key, value] : test_map_2) {
    EXPECT_EQ(result.at(key), value);
  }
  // Unique keys from first map should have their values
  EXPECT_EQ(result.at(1), 10);
  EXPECT_EQ(result.at(3), 30);
}

TEST(test_map_utils, map_union_custom_merge) {
  const std::map<int, int> test_map_1 = {{1, 10}, {2, 20}, {3, 30}};
  const std::map<int, int> test_map_2 = {{2, 200}, {4, 400}, {5, 500}};

  // Custom merge function that adds values on overlap
  const auto result = map_union(test_map_1, test_map_2, std::plus<>{});

  EXPECT_EQ(result.size(), 5);
  // Verify all keys and values
  for (const auto &[key, value] : result) {
    SCOPED_TRACE(::testing::Message() << "Checking key=" << key);
    const bool in_1 = map_contains(test_map_1, key);
    const bool in_2 = map_contains(test_map_2, key);
    if (in_1 && in_2) {
      // Overlapping key should have sum
      EXPECT_EQ(value, test_map_1.at(key) + test_map_2.at(key));
    } else if (in_1) {
      EXPECT_EQ(value, test_map_1.at(key));
    } else {
      EXPECT_EQ(value, test_map_2.at(key));
    }
  }
}

TEST(test_map_utils, map_union_custom_merge_with_key) {
  const std::map<int, int> test_map_1 = {{1, 10}, {2, 20}, {3, 30}};
  const std::map<int, int> test_map_2 = {{2, 200}, {4, 400}, {5, 500}};

  // Custom merge function f(key, v1, v2) that multiplies key by sum
  const auto key_times_sum = [](int key, int v1, int v2) {
    return key * (v1 + v2);
  };
  const auto result = map_union(test_map_1, test_map_2, key_times_sum);

  EXPECT_EQ(result.size(), 5);
  // Overlapping key should have key * (v1 + v2)
  EXPECT_EQ(result.at(2), 2 * (20 + 200)); // 440
  // Unique keys should have original values
  EXPECT_EQ(result.at(1), 10);
  EXPECT_EQ(result.at(3), 30);
  EXPECT_EQ(result.at(4), 400);
  EXPECT_EQ(result.at(5), 500);
}

TEST(test_map_utils, map_union_heterogeneous_key_conversion) {
  // Test union with key conversion (long → int)
  const std::map<int, int, CompareLongInt> test_map_1 = {{1, 10}, {3, 30}};
  const std::map<long, int, CompareLongInt> test_map_2 = {{2L, 200}, {3L, 300}};

  // Union with key conversion, default ReturnLeft
  const auto result = map_union(test_map_1, test_map_2);

  EXPECT_EQ(result.size(), 3);
  EXPECT_TRUE(map_contains(result, 1));
  EXPECT_TRUE(map_contains(result, 2)); // Converted from long
  EXPECT_TRUE(map_contains(result, 3));

  // Key 1 only in map_1
  EXPECT_EQ(result.at(1), 10);
  // Key 2 only in map_2 (converted)
  EXPECT_EQ(result.at(2), 200);
  // Key 3 in both - ReturnLeft should win
  EXPECT_EQ(result.at(3), 30); // From map_1, not map_2's 300
}

TEST(test_map_utils, map_union_all_keys_overlap) {
  // Test when all keys overlap (tests merge on every key)
  const std::map<int, int> test_map_1 = {{1, 10}, {2, 20}, {3, 30}};
  const std::map<int, int> test_map_2 = {{1, 100}, {2, 200}, {3, 300}};

  const auto result = map_union(test_map_1, test_map_2);

  EXPECT_EQ(result.size(), 3);
  // ReturnLeft: all values from map_1
  EXPECT_EQ(result.at(1), 10);
  EXPECT_EQ(result.at(2), 20);
  EXPECT_EQ(result.at(3), 30);
}

TEST(test_map_utils, map_subset_sorted_duplicate_keys) {
  // Test subset_sorted with duplicate keys in input vector
  const std::map<int, int> test_map = {{1, 10}, {2, 20}, {3, 30}, {4, 40}};
  std::vector<int> keys_with_duplicates = {1, 1, 2, 2, 4}; // Duplicates present

  // Should handle duplicates gracefully (process each, emplace_hint
  // ignores dups)
  const auto subset = map_subset_sorted(test_map, keys_with_duplicates);

  EXPECT_EQ(subset.size(), 3); // Only unique keys: 1, 2, 4
  EXPECT_TRUE(map_contains(subset, 1));
  EXPECT_TRUE(map_contains(subset, 2));
  EXPECT_TRUE(map_contains(subset, 4));
  EXPECT_FALSE(map_contains(subset, 3));

  // Values should be preserved
  EXPECT_EQ(subset.at(1), 10);
  EXPECT_EQ(subset.at(2), 20);
  EXPECT_EQ(subset.at(4), 40);
}

TEST(test_map_utils, map_subset_multiset) {
  // Test map_subset with std::multiset (allows duplicate keys)
  const std::map<int, int> test_map = {{1, 10}, {2, 20}, {3, 30}, {4, 40}};
  std::multiset<int> keys = {1, 1, 2, 4}; // multiset allows duplicates

  // Should work - multiset has key_compare
  const auto subset = map_subset(test_map, keys);

  EXPECT_EQ(subset.size(), 3);
  EXPECT_TRUE(map_contains(subset, 1));
  EXPECT_TRUE(map_contains(subset, 2));
  EXPECT_TRUE(map_contains(subset, 4));

  // Values preserved
  for (const auto &[key, value] : subset) {
    EXPECT_EQ(test_map.at(key), value);
  }
}

// Compile-time tests to verify type constraints are enforced
//
// Note: These tests verify that certain invalid type combinations are
// rejected at compile time via SFINAE. We use helper traits instead
// of std::is_invocable because the latter attempts to instantiate the
// template, which can give confusing error messages.
TEST(MapUtilsCompileTimeTest, TypeConstraintsDocumented) {
  // Test that key constraints are enforced
  using MapLess = std::map<int, int, std::less<int>>;
  using MapGreater = std::map<int, int, std::greater<int>>;

  // has_same_key_compare should detect different comparators
  static_assert(!has_same_key_compare_v<MapLess, MapGreater>,
                "Different comparators should be detected");

  static_assert(has_same_key_compare_v<MapLess, MapLess>,
                "Same comparators should be detected");

  // Test value type constraints
  using IntIntMap = std::map<int, int>;
  using IntStringMap = std::map<int, std::string>;

  static_assert(!std::is_same_v<typename IntIntMap::mapped_type,
                                typename IntStringMap::mapped_type>,
                "Different value types should be detected");

  // Test key conversion constraints
  static_assert(std::is_convertible_v<long, int>,
                "long should be convertible to int");

  static_assert(!std::is_convertible_v<std::string, int>,
                "string should not be convertible to int");

  // Test comparator detection for containers
  static_assert(has_same_key_compare_v<IntIntMap, std::set<int>>,
                "Map and set with same comparator should match");

  static_assert(
      !has_same_key_compare_v<IntIntMap, std::set<int, std::greater<int>>>,
      "Map and set with different comparators should not match");
}

} // namespace albatross
