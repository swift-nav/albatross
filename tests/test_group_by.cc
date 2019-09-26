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

#include <albatross/Dataset>
#include <albatross/src/core/groupby.hpp>

#include <gtest/gtest.h>

namespace albatross {

struct AboveThreshold {
  bool operator()(const int &x) const { return x > 3; }
};

struct IntMod2 {
  int operator()(const int &x) const { return x % 2; }
};

struct StringMode2 {
  std::string operator()(const int &x) const {
    return std::to_string(IntMod2()(x));
  }
};

RegressionDataset<int> test_integer_dataset() {
  std::vector<int> features = {3, 7, 1, 2, 5, 8};
  Eigen::VectorXd targets = Eigen::VectorXd::Random(6);
  return RegressionDataset<int>(features, targets);
}

struct BoolClassMethodGrouper {

  auto get_parent() const { return test_integer_dataset(); }

  auto get_grouper() const { return AboveThreshold(); }
};

struct BoolClassMethodVectorGrouper {

  auto get_parent() const { return test_integer_dataset().features; }

  auto get_grouper() const { return AboveThreshold(); }
};

struct BoolLambdaGrouper {

  auto get_parent() const { return test_integer_dataset(); }

  auto get_grouper() const {
    const auto get_group = [](const int &x) { return AboveThreshold()(x); };
    return get_group;
  }
};

struct IntClassMethodGrouper {

  auto get_parent() const { return test_integer_dataset(); }

  auto get_grouper() const { return IntMod2(); }
};

struct StringClassMethodGrouper {

  auto get_parent() const { return test_integer_dataset(); }

  auto get_grouper() const { return IntMod2(); }
};

template <typename CaseType> class GroupByTester : public ::testing::Test {
public:
  CaseType test_case;
};

typedef ::testing::Types<BoolClassMethodGrouper, BoolLambdaGrouper,
                         IntClassMethodGrouper, StringClassMethodGrouper,
                         BoolClassMethodVectorGrouper>
    GrouperTestCases;

TYPED_TEST_CASE_P(GroupByTester);

template <typename FeatureType>
auto get_iterable_elements(const RegressionDataset<FeatureType> &x) {
  return x.features;
}

template <typename FeatureType>
auto get_iterable_elements(const std::vector<FeatureType> &x) {
  return x;
}

TYPED_TEST_P(GroupByTester, test_groupby_groups) {
  auto parent = this->test_case.get_parent();
  const auto grouped = group_by(parent, this->test_case.get_grouper());

  // Can split into groups, and those groups align with the grouper
  for (const auto &pair : grouped.groups()) {
    for (const auto &f : get_iterable_elements(pair.second)) {
      EXPECT_EQ(this->test_case.get_grouper()(f), pair.first);
    }
  }
};

template <typename FeatureType>
void expect_same_but_maybe_out_of_order(
    const RegressionDataset<FeatureType> &x,
    const RegressionDataset<FeatureType> &y) {
  EXPECT_EQ(vector_set_difference(x.features, y.features).size(), 0);
  EXPECT_DOUBLE_EQ(x.targets.mean.sum(), y.targets.mean.sum());
}

template <typename FeatureType>
void expect_same_but_maybe_out_of_order(const std::vector<FeatureType> &x,
                                        const std::vector<FeatureType> &y) {
  EXPECT_EQ(vector_set_difference(x, y).size(), 0);
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

  ASSERT_GT(first_group.size(), 1);

  const std::size_t num_removed = first_group.size() - 1;
  std::vector<std::size_t> single_ind = {0};
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

REGISTER_TYPED_TEST_CASE_P(GroupByTester, test_groupby_groups,
                           test_groupby_counts, test_groupby_combine,
                           test_groupby_modify_combine,
                           test_groupby_apply_combine, test_groupby_apply_void);

INSTANTIATE_TYPED_TEST_CASE_P(test_groupby, GroupByTester, GrouperTestCases);

} // namespace albatross
