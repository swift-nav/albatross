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
  return (matrix.array() != 0.0).any();
};

inline bool has_zero(const Eigen::MatrixXd &matrix) {
  return (matrix.array() == 0.0).any();
};

/*
 * Tests that the covariance between all features in different groups
 * is exactly zero. For example, if you're adding a covariance term for a
 * bias which is specific to a certain type of measurement, then
 * you probably want to make sure that your implementation is actually
 * producing zeros across measurement types.
 */
template <typename FeatureType, typename GrouperFunction,
          typename CovarianceFunction>
inline void
expect_zero_covariance_across_groups(const CovarianceFunction &cov_func,
                                     const std::vector<FeatureType> &features,
                                     GrouperFunction grouper,
                                     bool strict = true) {
  const auto grouped = group_by(features, grouper);
  // make sure we're actually testing something.
  ASSERT_TRUE(grouped.size() > 1);

  const auto are_correlated =
      grouped.apply([&](const auto &g) { return has_non_zero(cov_func(g)); });

  bool any = false;
  bool all = true;

  for (const auto &one_group : are_correlated) {
    any = any || one_group.second;
    all = all && one_group.second;
  }

  // first we make sure that within group(s) we see some non-zero covariance
  // otherwise the following checks aren't informative.
  EXPECT_TRUE(any);
  EXPECT_TRUE(all || !strict);

  const auto expect_zero_across_groups =
      [&](const auto &, const albatross::GroupIndices &indices) {
        const auto in_group = albatross::subset(features, indices);
        const auto complement =
            albatross::indices_complement(indices, features.size());
        const auto not_in_group = albatross::subset(features, complement);
        EXPECT_EQ(cov_func(in_group, not_in_group).array().abs().maxCoeff(),
                  0.);
      };

  grouped.index_apply(expect_zero_across_groups);
}

/*
 * Make sure that a covariance function is always non zero for any
 * features in the same group, this can be used to try to catch bugs
 * in which an equality operator is actually more strict than you'd
 * expect. For example, if you are building a covariance function
 * which should add a bias which is unique to some property of the
 * measurement you can use this to make sure it applies that bias
 * to all measurements with the same property.
 */
template <typename FeatureType, typename GrouperFunction,
          typename CovarianceFunction>
inline void expect_non_zero_covariance_within_groups(
    const CovarianceFunction &cov_func,
    const std::vector<FeatureType> &features, GrouperFunction grouper) {
  const auto grouped = group_by(features, grouper);

  const bool any_zeros_within_groups =
      grouped.apply([&](const auto &g) { return has_zero(cov_func(g)); }).any();

  EXPECT_FALSE(any_zeros_within_groups);
}

template <typename CovarianceFunction, typename FeatureType,
          typename InducingFeatureType>
inline Eigen::MatrixXd conditional_dependence(
    const CovarianceFunction &covariance,
    const std::vector<FeatureType> &a_features,
    const std::vector<FeatureType> &b_features,
    const std::vector<InducingFeatureType> &inducing_points) {
  // Here we compute the amount of the covariance between a and b which is not
  // captured by the inducing points.  Conditional independence is a fundamental
  // assumption in sparse gaussian processes, this provides a check for it.

  // Conditional independence is defined as:
  //
  //   P[a|u,b] = P[a|u]
  //   P[b|u,a] = P[b|u]
  //
  // Which means that A and B are independent of each other when you know the
  // inducing points.  In otherwords, the inducing points capture
  // any information shared between A and B.
  //
  // To start we have the priors of A given B:
  const Eigen::MatrixXd cov_aa = covariance(a_features);
  const Eigen::MatrixXd cov_bb = covariance(b_features);
  const Eigen::MatrixXd cov_ab = covariance(a_features, b_features);

  // Then we can compute the relations to U (inducing points)
  const Eigen::MatrixXd cov_au = covariance(a_features, inducing_points);
  const Eigen::MatrixXd cov_ub = covariance(inducing_points, b_features);
  const Eigen::MatrixXd cov_uu = covariance(inducing_points);

  // COV[a,b|u] = COV[a,b] - COV[a,u] COV[u,u]^-1 COV[u,b]
  const Eigen::MatrixXd cross_explained = cov_au * cov_uu.ldlt().solve(cov_ub);
  const Eigen::MatrixXd remaining_cross_covariance = cov_ab - cross_explained;

  // Rescale to represent a correlation instead of covariance.
  const Eigen::VectorXd a_sds = cov_aa.diagonal().array().sqrt();
  const Eigen::VectorXd b_sds = cov_bb.diagonal().array().sqrt();
  const Eigen::MatrixXd remaining_cross_correlation =
      a_sds.asDiagonal().inverse() * remaining_cross_covariance *
      b_sds.asDiagonal().inverse();
  return remaining_cross_correlation;
}

/*
 * This test is particularly geared towards sparse Gaussian processes,
 * it tests to make sure that a provided set of inducing points actually
 * captures most of the covariance between different groups. In otherwords
 * it tests that the unexplained covariance (see sparse_gp.hpp for notation):
 *
 *     A = K_ff - Q_ff
 *
 * can be approximated by a block diagonal, with blocks defined by groups.
 */
template <typename CovarianceFunction, typename FeatureType,
          typename InducingFeatureType, typename Grouper>
inline void expect_inducing_points_capture_cross_correlation(
    const CovarianceFunction &covariance,
    const std::vector<FeatureType> &features,
    const std::vector<InducingFeatureType> &inducing_points, Grouper grouper) {
  const auto groups = albatross::group_by(features, grouper).groups();

  auto between_group_cross_correlation_captured =
      [&](const auto &key, const auto &group_features) {
        for (const auto &pair : groups) {
          if (pair.first != key) {
            const Eigen::MatrixXd cond_dep = conditional_dependence(
                covariance, group_features, pair.second, inducing_points);
            const double acceptable_correlation = 0.005;
            EXPECT_LT(cond_dep.array().abs().maxCoeff(),
                      acceptable_correlation);
          }
        }
      };

  groups.apply(between_group_cross_correlation_captured);
}

// Sometimes we expect to be able to use two different feature types
// interchangebly for a given covariance function, here we check all
// possible conversions for a set of features.
template <typename CovarianceFunction, typename FeatureType, typename Convert>
inline void
expect_converted_feature_equivalence(const CovarianceFunction &cov,
                                     const std::vector<FeatureType> &features,
                                     Convert convert) {
  for (const auto &f_a : features) {
    const auto alt_a = convert(f_a);
    for (const auto &f_b : features) {
      const auto alt_b = convert(f_b);
      EXPECT_EQ(cov(f_a, f_b), cov(alt_a, alt_b));
      EXPECT_EQ(cov(f_a, f_b), cov(f_a, alt_b));
      EXPECT_EQ(cov(f_a, f_b), cov(alt_a, f_b));
    }
  }
}

} // namespace albatross

#endif /* TESTS_TEST_COVARIANCE_UTILS_H_ */
