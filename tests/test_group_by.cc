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

#include <albatross/Indexing>
#include <gtest/gtest.h>
#include <string.h>

namespace albatross {

bool above_three(const int &x) { return x > 3; }

struct AboveThree {
  bool operator()(const int &x) const { return above_three(x); }
};

int mod_three(const int &x) { return x % 3; }

struct IntMod3 {
  int operator()(const int &x) const { return mod_three(x); }
};

struct StringMod3 {
  std::string operator()(const int &x) const {
    return std::to_string(IntMod3()(x));
  }
};

bool overloaded_above_three(const int &x) { return above_three(x); }

bool overloaded_above_three(const double &x) { return x > 3.; }

struct CustomGroupKey {

  bool operator<(const CustomGroupKey &other) const {
    return value < other.value;
  }

  bool operator==(const CustomGroupKey &other) const {
    return value == other.value;
  }

  double value;
};

CustomGroupKey custom_nearest_even_number(const int &x) {
  CustomGroupKey key;
  key.value = x - (x % 2);
  return key;
}

struct CustomNearestEvenNumber {
  CustomGroupKey operator()(const int &x) const {
    return custom_nearest_even_number(x);
  }
};

RegressionDataset<int> test_integer_dataset() {
  std::vector<int> features = {3, 7, 1, 2, 5, 8};
  Eigen::VectorXd targets =
      Eigen::VectorXd::Random(cast::to_index(features.size()));
  return RegressionDataset<int>(features, targets);
}

/*
 * Here we define all the test cases.
 */

struct BoolClassMethodGrouper {

  auto get_parent() const { return test_integer_dataset(); }

  auto get_grouper() const { return AboveThree(); }
};

struct BoolFunctionGrouper {

  auto get_parent() const { return test_integer_dataset(); }

  auto get_grouper() const { return above_three; }
};

struct BoolOverloadedFunctionGrouper {

  auto get_parent() const { return test_integer_dataset(); }

  auto get_grouper() const {
    return select_overload<int>(overloaded_above_three);
  }
};

struct BoolFunctionPointerGrouper {

  auto get_parent() const { return test_integer_dataset(); }

  auto get_grouper() const { return &above_three; }
};

struct BoolClassMethodVectorGrouper {

  auto get_parent() const { return test_integer_dataset().features; }

  auto get_grouper() const { return AboveThree(); }
};

struct BoolLambdaGrouper {

  auto get_parent() const { return test_integer_dataset(); }

  auto get_grouper() const {
    const auto get_group = [](const int &x) { return AboveThree()(x); };
    return get_group;
  }
};

struct IntClassMethodGrouper {

  auto get_parent() const { return test_integer_dataset(); }

  auto get_grouper() const { return IntMod3(); }
};

struct StringClassMethodGrouper {

  auto get_parent() const { return test_integer_dataset(); }

  auto get_grouper() const { return IntMod3(); }
};

struct LeaveOneOutTest {
  auto get_parent() const { return test_integer_dataset(); }

  auto get_grouper() const { return LeaveOneOutGrouper(); }
};

struct CustomClassMethodGrouper {
  auto get_parent() const { return test_integer_dataset(); }

  auto get_grouper() const { return CustomNearestEvenNumber(); }
};

struct CustomFunctionGrouper {
  auto get_parent() const { return test_integer_dataset(); }

  auto get_grouper() const { return custom_nearest_even_number; }
};

template <typename CaseType> class GroupByTester : public ::testing::Test {
public:
  CaseType test_case;
};

typedef ::testing::Types<BoolClassMethodGrouper, BoolLambdaGrouper,
                         BoolFunctionGrouper, BoolOverloadedFunctionGrouper,
                         BoolFunctionPointerGrouper, IntClassMethodGrouper,
                         StringClassMethodGrouper, BoolClassMethodVectorGrouper,
                         LeaveOneOutTest, CustomClassMethodGrouper,
                         CustomFunctionGrouper>
    GrouperTestCases;

TYPED_TEST_SUITE_P(GroupByTester);

template <typename GrouperFunction, typename ValueType,
          typename GroupKey = typename details::grouper_result<GrouperFunction,
                                                               ValueType>::type,
          typename std::enable_if<
              !std::is_same<GrouperFunction, LeaveOneOutGrouper>::value,
              int>::type = 0>
void expect_group_key_matches_expected(const GrouperFunction &grouper,
                                       const ValueType &value,
                                       const GroupKey &expected) {
  EXPECT_EQ(grouper(value), expected);
}

template <typename GrouperFunction, typename ValueType,
          typename GroupKey = typename details::grouper_result<GrouperFunction,
                                                               ValueType>::type,
          typename std::enable_if<
              std::is_same<GrouperFunction, LeaveOneOutGrouper>::value,
              int>::type = 0>
void expect_group_key_matches_expected(
    const GrouperFunction &grouper ALBATROSS_UNUSED,
    const ValueType &value ALBATROSS_UNUSED,
    const GroupKey &expected ALBATROSS_UNUSED) {}

TYPED_TEST_P(GroupByTester, test_groupby_groups) {
  auto parent = this->test_case.get_parent();
  const auto grouped = group_by(parent, this->test_case.get_grouper());

  // Can split into groups, and those groups align with the grouper
  for (const auto &pair : grouped.groups()) {
    // we only need this to get at the underlying iterable
    const auto value_grouped =
        group_by(pair.second, this->test_case.get_grouper());
    for (const auto &f : value_grouped._get_iterable()) {
      expect_group_key_matches_expected(this->test_case.get_grouper(), f,
                                        pair.first);
    }
  }
};

template <typename FeatureType>
void expect_same_but_maybe_out_of_order(
    const RegressionDataset<FeatureType> &x,
    const RegressionDataset<FeatureType> &y) {
  EXPECT_EQ(vector_set_difference(x.features, y.features).size(), 0);
  EXPECT_LT(fabs(x.targets.mean.sum() - y.targets.mean.sum()), 1e-10);
}

template <typename FeatureType>
void expect_same_but_maybe_out_of_order(const std::vector<FeatureType> &x,
                                        const std::vector<FeatureType> &y) {
  EXPECT_EQ(vector_set_difference(x, y).size(), 0);
}

// This function is separate so that it can be annotated to disable
// sanitizers, which doesn't work with the `TYPED_TEST_P` or other
// googletest macros.
template <typename Grouper>
static auto const_groups_key(Grouper *grouper)
#ifdef __clang__
  // https://github.com/swift-nav/albatross/issues/372
  __attribute__((no_sanitize("address")))
  __attribute__((no_sanitize("thread")))
#endif // __clang__
{
  const auto const_groups = grouper->groups();
  assert(const_groups.size() > 0);
  const auto first_key = const_groups.keys()[0];
  const_groups.at(first_key);
  return first_key;
}

TYPED_TEST_P(GroupByTester, test_groupby_access_methods)
{
  auto parent = this->test_case.get_parent();
  const auto grouper = group_by(parent, this->test_case.get_grouper());

  const auto first_key = const_groups_key(&grouper);

  auto groups = grouper.groups();
  groups.at(first_key);
  groups[first_key];
}

TYPED_TEST_P(GroupByTester, test_groupby_combine) {
  auto parent = this->test_case.get_parent();
  const auto grouped = group_by(parent, this->test_case.get_grouper());

  const auto combined = grouped.groups().combine();

  EXPECT_EQ(combined.size(), parent.size());

  expect_same_but_maybe_out_of_order(combined, parent);
}

TYPED_TEST_P(GroupByTester, test_groupby_counts) {
  auto parent = this->test_case.get_parent();
  const auto grouped = group_by(parent, this->test_case.get_grouper());

  const auto counts = grouped.counts();

  for (const auto &pair : grouped.indexers()) {
    EXPECT_EQ(counts.at(pair.first), pair.second.size());
  }
}

TYPED_TEST_P(GroupByTester, test_groupby_modify_combine) {
  auto parent = this->test_case.get_parent();
  const auto grouped = group_by(parent, this->test_case.get_grouper());
  auto groups = grouped.groups();

  const auto first_key = map_keys(groups)[0];
  const auto first_group = groups[first_key];

  const std::size_t num_removed = first_group.size();
  std::vector<std::size_t> single_ind = {};
  groups[first_key] = albatross::subset(first_group, single_ind);

  const auto combined = groups.combine();

  EXPECT_EQ(combined.size(), parent.size() - num_removed);
  EXPECT_EQ(combine(groups).size(), combined.size());
}

TYPED_TEST_P(GroupByTester, test_groupby_apply_combine) {
  auto parent = this->test_case.get_parent();
  const auto grouped = group_by(parent, this->test_case.get_grouper());

  const auto only_keep_one = [](const auto &, const auto &one_group) {
    std::vector<std::size_t> single_ind = {0};
    return subset(one_group, single_ind);
  };

  const auto combined = grouped.apply(only_keep_one).combine();

  // Same number of final combined elements as there are groups.
  EXPECT_EQ(grouped.size(), combined.size());
}

TYPED_TEST_P(GroupByTester, test_groupby_apply_void) {
  auto parent = this->test_case.get_parent();
  const auto grouped = group_by(parent, this->test_case.get_grouper());

  std::size_t count = 0;

  const auto increment_count = [&](const auto &, const auto &) { ++count; };

  grouped.apply(increment_count);

  EXPECT_EQ(grouped.size(), count);
}

TYPED_TEST_P(GroupByTester, test_groupby_apply_value_only) {
  auto parent = this->test_case.get_parent();
  const auto grouped = group_by(parent, this->test_case.get_grouper());

  std::size_t count = 0;

  const auto increment_count = [&](const auto &) { ++count; };

  grouped.apply(increment_count);

  EXPECT_EQ(grouped.size(), count);
}

TYPED_TEST_P(GroupByTester, test_groupby_index_apply) {
  auto parent = this->test_case.get_parent();
  const auto grouped = group_by(parent, this->test_case.get_grouper());

  std::size_t count = 0;

  const auto increment_count = [&](const auto &, const GroupIndices &) {
    ++count;
  };

  grouped.index_apply(increment_count);

  EXPECT_EQ(grouped.size(), count);
}

TYPED_TEST_P(GroupByTester, test_groupby_filter) {
  auto parent = this->test_case.get_parent();
  const auto grouped = group_by(parent, this->test_case.get_grouper());

  const auto keys = grouped.keys();
  assert(keys.size() > 1);

  const auto remove_first = [&keys](const auto &key, const auto &) {
    return !(key == keys[0]);
  };

  const auto filtered = grouped.filter(remove_first);

  EXPECT_EQ(filtered.size(), grouped.size() - 1);

  // Combine, then regroup and make sure the combined object no longer
  // contains the removed group.
  EXPECT_EQ(group_by(filtered.combine(), this->test_case.get_grouper()).size(),
            filtered.size());
}

REGISTER_TYPED_TEST_SUITE_P(GroupByTester, test_groupby_access_methods,
                            test_groupby_groups, test_groupby_counts,
                            test_groupby_combine, test_groupby_modify_combine,
                            test_groupby_apply_combine, test_groupby_apply_void,
                            test_groupby_filter, test_groupby_apply_value_only,
                            test_groupby_index_apply);

INSTANTIATE_TYPED_TEST_SUITE_P(test_groupby, GroupByTester, GrouperTestCases);

/*
 * Test Filtering
 */

std::vector<double> fibonacci(std::size_t n) {
  assert(n > 2);
  std::vector<double> fib = {1., 2.};
  for (std::size_t i = 2; i < n; ++i) {
    fib.emplace_back(fib[i - 1] + fib[i - 2]);
  }
  return fib;
}

double mean(const std::vector<double> &xs) {
  double mean = 0;
  for (const auto &x : xs) {
    mean += x;
  }
  return mean / cast::to_double(xs.size());
}

long int number_of_digits(double x) { return lround(ceil(log10(x))); }

std::vector<double>
direct_remove_less_than_mean(const std::vector<double> &xs) {

  std::map<long, std::vector<double>> grouped;
  for (const auto &x : xs) {
    long digits = number_of_digits(x);
    if (grouped.find(digits) == grouped.end()) {
      grouped[digits] = {x};
    } else {
      grouped[digits].emplace_back(x);
    }
  }

  std::vector<double> output;
  for (const auto &group_pair : grouped) {
    const double group_mean = mean(group_pair.second);
    for (const auto &v : group_pair.second) {
      if (v > group_mean) {
        output.emplace_back(v);
      }
    }
  }

  return output;
}

std::vector<double>
split_apply_combine_less_than_mean(const std::vector<double> &xs) {

  const auto remove_less_than_mean = [](const std::vector<double> &group) {
    const double group_mean = mean(group);
    const auto is_greater_than_mean = [&group_mean](const double &x) {
      return x > group_mean;
    };
    return filter(group, is_greater_than_mean);
  };

  return group_by(xs, number_of_digits).apply(remove_less_than_mean).combine();
}

TEST(test_groupby, test_group_by_nested_filter) {

  const auto fib = fibonacci(20);

  const auto filtered = split_apply_combine_less_than_mean(fib);

  const auto direct = direct_remove_less_than_mean(fib);

  EXPECT_EQ(direct.size(), filtered.size());

  for (std::size_t i = 0; i < direct.size(); ++i) {
    EXPECT_EQ(direct[i], filtered[i]);
  }
}

TEST(test_groupby, test_group_by_combine_eigen) {

  albatross::Grouped<Eigen::Index, Eigen::VectorXd> grouped;
  grouped[3] = Eigen::VectorXd::Constant(3, 1, 3.);
  grouped[1] = Eigen::VectorXd::Constant(1, 1, 1.);
  grouped[0] = Eigen::VectorXd();
  grouped[5] = Eigen::VectorXd::Constant(5, 1, 5.);

  Eigen::VectorXd expected(9);
  expected << 1., 3., 3., 3., 5., 5., 5., 5., 5.;

  Eigen::VectorXd actual = grouped.combine();
  EXPECT_EQ(expected, actual);
}

TEST(test_groupby, test_group_by_first_group) {

  const auto fib = fibonacci(20);

  const auto grouped = group_by(fib, number_of_digits);
  const auto first_group = grouped.first_group();

  for (const auto &value : first_group.second) {
    EXPECT_EQ(number_of_digits(value), first_group.first);
  }

  const auto groups = grouped.groups();
  EXPECT_EQ(groups.first_group(), first_group);
  EXPECT_EQ(groups.first_value(), first_group.second);
}

TEST(test_groupby, test_group_by_get_group) {

  const auto fib = fibonacci(20);

  const auto group_2 = group_by(fib, number_of_digits).get_group(2);

  for (const auto &value : group_2) {
    EXPECT_EQ(number_of_digits(value), 2);
  }
}

TEST(test_groupby, test_group_by_erase) {

  const auto fib = fibonacci(20);

  const auto groups = group_by(fib, number_of_digits).groups();

  auto modified = group_by(fib, number_of_digits).groups();
  const long int to_remove = 2;
  modified.erase(to_remove);

  EXPECT_TRUE(map_contains(groups, to_remove));
  EXPECT_FALSE(map_contains(modified, to_remove));

  for (const auto &pair : groups) {
    if (pair.first != to_remove) {
      EXPECT_EQ(pair.second, modified.at(pair.first));
    }
  }
}

template <typename T> inline double test_sum(const std::vector<T> &ts) {
  double output = 0.;
  for (const auto &t : ts) {
    output += static_cast<double>(t);
  }
  return output;
};

template <typename T> inline double test_mean(const std::vector<T> &ts) {
  return test_sum(ts) / cast::to_double(ts.size());
}

TEST(test_groupby, test_group_by_min_max_value) {

  const auto fib = fibonacci(20);

  const auto sums = group_by(fib, number_of_digits).apply(test_sum<double>);

  double actual_max = -INFINITY;
  double actual_min = INFINITY;
  for (const auto &pair : sums) {
    if (pair.second < actual_min) {
      actual_min = pair.second;
    }
    if (pair.second > actual_max) {
      actual_max = pair.second;
    }
  }

  EXPECT_EQ(actual_min, sums.min_value());
  EXPECT_EQ(actual_max, sums.max_value());
}

TEST(test_groupby, test_group_by_sum_mean) {

  const auto fib = fibonacci(20);

  const auto means = group_by(fib, number_of_digits).apply(test_mean<double>);

  const double expected_mean = test_mean(map_values(means));
  EXPECT_EQ(expected_mean, means.mean());

  const double expected_sum = test_sum(map_values(means));
  EXPECT_EQ(expected_sum, means.sum());
}

TEST(test_groupby, test_group_by_min_max) {

  Grouped<std::string, int> example;
  example["one"] = 1;
  example["two"] = 2;
  example["negative"] = -5;
  example["large"] = 7;

  EXPECT_EQ(example.max_value(), 7);
  EXPECT_EQ(example.max_key(), "large");
  EXPECT_EQ(example.min_value(), -5);
  EXPECT_EQ(example.min_key(), "negative");
}

TEST(test_groupby, test_group_by_any_all) {

  const auto fib = fibonacci(20);

  const auto grouped = group_by(fib, number_of_digits);
  const auto counts = grouped.counts();
  const auto min_count = counts.min_value();
  const auto max_count = counts.max_value();

  auto greater_or_equal_to_min_count = [&](const auto &xs) {
    return xs.size() >= min_count;
  };

  auto greater_than_min_count = [&](const auto &xs) {
    return xs.size() > min_count;
  };

  auto greater_than_max_count = [&](const auto &xs) {
    return xs.size() > max_count;
  };

  EXPECT_TRUE(grouped.apply(greater_or_equal_to_min_count).all());
  EXPECT_TRUE(grouped.apply(greater_or_equal_to_min_count).any());

  EXPECT_FALSE(grouped.apply(greater_than_min_count).all());
  EXPECT_TRUE(grouped.apply(greater_than_min_count).any());

  EXPECT_FALSE(grouped.apply(greater_than_max_count).all());
  EXPECT_FALSE(grouped.apply(greater_than_max_count).any());
}

TEST(test_groupby, test_group_by_with_vector) {

  const auto fib = fibonacci(20);

  std::vector<std::string> strings;
  for (const auto &x : fib) {
    strings.push_back(std::to_string(x));
  }

  const auto grouped_with_strings =
      group_by(fib, number_of_digits).with(strings);
  EXPECT_GT(grouped_with_strings.size(), 0);
  for (const auto &group : grouped_with_strings) {
    const std::vector<double> &vector_of_doubles = group.second.first;
    const std::vector<std::string> &vector_of_strings = group.second.second;
    for (std::size_t i = 0; i < vector_of_doubles.size(); ++i) {
      EXPECT_EQ(std::to_string(vector_of_doubles[i]), vector_of_strings[i]);
    }
  }
}

TEST(test_groupby, test_group_by_with_map) {

  const auto fib = fibonacci(20);
  const auto grouped = group_by(fib, number_of_digits);

  auto doubles_to_strings = [](const std::vector<double> &xs) {
    std::vector<std::string> strings;
    for (const auto &x : xs) {
      strings.push_back(std::to_string(x));
    }
    return strings;
  };
  const auto as_strings = grouped.apply(doubles_to_strings);

  const auto with_strings = grouped.with(as_strings);

  EXPECT_GT(with_strings.size(), 0);
  for (const auto &group : with_strings) {
    const std::vector<double> &vector_of_doubles = group.second.first;
    const std::vector<std::string> &vector_of_strings = group.second.second;
    for (std::size_t i = 0; i < vector_of_doubles.size(); ++i) {
      EXPECT_EQ(std::to_string(vector_of_doubles[i]), vector_of_strings[i]);
    }
  }
}

} // namespace albatross
