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

#ifndef TESTS_TEST_COVARIANCE_UTILS_H_
#define TESTS_TEST_COVARIANCE_UTILS_H_

#include <gtest/gtest.h>

namespace albatross {

struct V {};
struct W {};

struct X {
  bool operator==(const X &) const { return false; }
};

struct Y {};
struct Z {};

class HasXX : public CovarianceFunction<HasXX> {
public:
  double _call_impl(const X &, const X &) const { return 1.; };
};

class HasXY : public CovarianceFunction<HasXY> {
public:
  double _call_impl(const X &, const Y &) const { return 1.; };
};

class HasNone : public CovarianceFunction<HasNone> {};

class HasMultiple : public CovarianceFunction<HasMultiple> {
public:
  double _call_impl(const X &, const X &) const { return 1.; };

  double _call_impl(const X &, const Y &) const { return 3.; };

  double _call_impl(const Y &, const Y &) const { return 5.; };

  double _call_impl(const W &, const W &) const { return 7.; };

  double _call_impl(const V &, const V &) const { return 11.; };

  // These are all invalid:
  double _call_impl(const Z &, const X &) { return 1.; };

  double _call_impl(Z &, const Y &) const { return 1.; };

  int _call_impl(const Z &, const Z &) const { return 1.; };

  std::string name_ = "has_multiple";
};

class HasMultipleMean : public MeanFunction<HasMultiple> {
public:
  double _call_impl(const X &) const { return 1.; };

  double _call_impl(const Y &) const { return 3.; };

  double _call_impl(const W &) const { return 7.; };

  // These are all invalid:
  double _call_impl(const Z &) { return 1.; };

  double _call_impl(V &) const { return 11.; };
};

class HasPublicCallImpl {
public:
  double _call_impl(const X &, const Y &) const { return 1.; };
};

class HasProtectedCallImpl {
protected:
  double _call_impl(const X &, const Y &) const { return 1.; };
};

class HasPrivateCallImpl {
  double _call_impl(const X &, const Y &) const { return 1.; };
};

class HasNoCallImpl {};

/*
 * The following utilities are helpful for verifying covariance
 * functions work as expected.
 */

inline bool has_non_zero(const Eigen::MatrixXd &matrix) {
  double max_element = matrix.array().abs().maxCoeff();
  return max_element > 0.;
};

template <typename FeatureType, typename GrouperFunction,
          typename CovarianceFunction>
inline void expect_zero_covariance_across_groups(
    const RegressionDataset<FeatureType> &dataset, GrouperFunction grouper,
    const CovarianceFunction &cov_func, bool strict = true) {
  const auto grouped = dataset.group_by(grouper);

  const auto are_correlated = grouped.apply(
      [&](const auto &d) { return has_non_zero(cov_func(d.features)); });

  bool any = false;
  bool all = true;

  for (const auto &one_group : are_correlated) {
    any = any || one_group.second;
    all = all && one_group.second;
  }

  EXPECT_TRUE(any);
  EXPECT_TRUE(all || !strict);

  const auto expect_zero_across_groups =
      [&](const auto &, const albatross::GroupIndices &indices) {
        const auto in_group = albatross::subset(dataset.features, indices);
        const auto complement =
            albatross::indices_complement(indices, dataset.features.size());
        const auto not_in_group =
            albatross::subset(dataset.features, complement);
        EXPECT_EQ(cov_func(in_group, not_in_group).array().abs().maxCoeff(),
                  0.);
      };

  grouped.index_apply(expect_zero_across_groups);
}

} // namespace albatross

#endif /* TESTS_TEST_COVARIANCE_UTILS_H_ */
