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

#include <albatross/CovarianceFunctions>

#include <gtest/gtest.h>

namespace albatross {

template <typename X, typename CovarianceFunction> class IdentityCov {
public:
  IdentityCov(CovarianceFunction &covariance_function,
              const std::vector<X> &features)
      : features_(features), cov_(covariance_function) {}

  template <typename Y> void test(const std::vector<Y> &features) {
    Eigen::MatrixXd c = cov_(features, features_);
    std::cout << c << std::endl;
  }

  std::vector<X> features_;
  CovarianceFunction cov_;
};

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

  std::string name_ = "has_multiple";
};

TEST(test_covariance_function, test_operator_resolution) {

  EXPECT_TRUE(bool(has_call_operator<HasXY, X, Y>::value));
  EXPECT_TRUE(bool(has_call_operator<HasXY, Y, X>::value));
  EXPECT_FALSE(bool(has_call_operator<HasXY, X, X>::value));
  EXPECT_FALSE(bool(has_call_operator<HasXY, Y, Y>::value));
  EXPECT_FALSE(bool(has_call_operator<HasXY, Z, Z>::value));

  EXPECT_FALSE(bool(has_call_operator<HasNone, X, Y>::value));
  EXPECT_FALSE(bool(has_call_operator<HasNone, Y, X>::value));
  EXPECT_FALSE(bool(has_call_operator<HasNone, X, X>::value));
  EXPECT_FALSE(bool(has_call_operator<HasNone, Y, Y>::value));
  EXPECT_FALSE(bool(has_call_operator<HasNone, Z, Z>::value));

  EXPECT_TRUE(bool(has_call_operator<HasMultiple, X, Y>::value));
  EXPECT_TRUE(bool(has_call_operator<HasMultiple, Y, X>::value));
  EXPECT_TRUE(bool(has_call_operator<HasMultiple, X, X>::value));
  EXPECT_TRUE(bool(has_call_operator<HasMultiple, Y, Y>::value));
  EXPECT_FALSE(bool(has_call_operator<HasMultiple, Z, Z>::value));
}

TEST(test_covariance_function, test_vector_operator_inspection) {
  EXPECT_TRUE(
      bool(has_call_operator<HasXY, std::vector<X>, std::vector<Y>>::value));
  EXPECT_TRUE(
      bool(has_call_operator<HasXY, std::vector<Y>, std::vector<X>>::value));
  EXPECT_FALSE(
      bool(has_call_operator<HasXY, std::vector<X>, std::vector<X>>::value));
  EXPECT_FALSE(
      bool(has_call_operator<HasXY, std::vector<Y>, std::vector<Y>>::value));
  EXPECT_FALSE(
      bool(has_call_operator<HasXY, std::vector<Z>, std::vector<Z>>::value));

  EXPECT_TRUE(bool(
      has_call_operator<HasMultiple, std::vector<X>, std::vector<Y>>::value));
  EXPECT_TRUE(bool(
      has_call_operator<HasMultiple, std::vector<Y>, std::vector<X>>::value));
  EXPECT_TRUE(bool(
      has_call_operator<HasMultiple, std::vector<X>, std::vector<X>>::value));
  EXPECT_TRUE(bool(
      has_call_operator<HasMultiple, std::vector<Y>, std::vector<Y>>::value));
  EXPECT_FALSE(bool(
      has_call_operator<HasMultiple, std::vector<Z>, std::vector<Z>>::value));
}

TEST(test_covariance_function, test_covariance_matrix) {
  HasMultiple cov;

  std::vector<X> xs = {{}, {}, {}};
  std::vector<Y> ys = {{}, {}};

  EXPECT_EQ(cov(xs).size(), 9);
  EXPECT_EQ(cov(ys).size(), 4);
  EXPECT_EQ(cov(xs, ys).size(), 6);

  const std::vector<X> const_xs = {{}, {}, {}};
  const std::vector<Y> const_ys = {{}, {}};

  EXPECT_EQ(cov(const_xs).size(), 9);
  EXPECT_EQ(cov(const_ys).size(), 4);
  EXPECT_EQ(cov(const_xs, const_ys).size(), 6);

  EXPECT_EQ(cov(std::vector<X>({{}, {}, {}})).size(), 9);
  EXPECT_EQ(cov(std::vector<Y>({{}, {}})).size(), 4);
  EXPECT_EQ(cov(std::vector<X>({{}, {}, {}}), std::vector<Y>({{}, {}})).size(),
            6);
}

TEST(test_covariance_function, test_works_with_variants) {
  HasMultiple cov;

  X x;
  Y y;

  EXPECT_EQ(cov(x, x), 1.);
  EXPECT_EQ(cov(x, y), 3.);
  EXPECT_EQ(cov(y, x), 3.);
  EXPECT_EQ(cov(y, y), 5.);

  variant<X, Y> vx = x;
  variant<X, Y> vy = y;

  EXPECT_EQ(cov(vx, x), cov(x, x));
  EXPECT_EQ(cov(vx, vx), cov(x, x));
  EXPECT_EQ(cov(x, vx), cov(x, x));

  EXPECT_EQ(cov(vx, y), cov(x, y));
  EXPECT_EQ(cov(vx, vy), cov(x, y));
  EXPECT_EQ(cov(x, vy), cov(x, y));

  EXPECT_EQ(cov(vy, x), cov(x, y));
  EXPECT_EQ(cov(vy, vx), cov(x, y));
  EXPECT_EQ(cov(y, vx), cov(x, y));

  EXPECT_EQ(cov(vy, y), cov(y, y));
  EXPECT_EQ(cov(vy, vy), cov(y, y));
  EXPECT_EQ(cov(y, vy), cov(y, y));
}

TEST(test_covariance_function, test_variant_recurssion_bug) {
  // This tests a bug in which the variant forwarder would recurse down the tree
  // of covariance functions until it found one defined for all types involved
  // and in turn ignored some terms.
  HasMultiple has_multiple;
  HasXX has_xx;

  auto cov = has_xx + has_multiple;

  X x;
  Y y;

  EXPECT_EQ(cov(x, x), has_xx(x, x) + has_multiple(x, x));
  EXPECT_EQ(cov(x, y), has_multiple(x, y));
  EXPECT_EQ(cov(y, x), has_multiple(y, x));
  EXPECT_EQ(cov(y, y), has_multiple(y, y));

  variant<X, Y> vx = x;
  variant<X, Y> vy = y;

  EXPECT_EQ(cov(vx, x), cov(x, x));
  EXPECT_EQ(cov(vx, vx), cov(x, x));
  EXPECT_EQ(cov(x, vx), cov(x, x));

  EXPECT_EQ(cov(vx, y), cov(x, y));
  EXPECT_EQ(cov(vx, vy), cov(x, y));
  EXPECT_EQ(cov(x, vy), cov(x, y));

  EXPECT_EQ(cov(vy, x), cov(x, y));
  EXPECT_EQ(cov(vy, vx), cov(x, y));
  EXPECT_EQ(cov(y, vx), cov(x, y));

  EXPECT_EQ(cov(vy, y), cov(y, y));
  EXPECT_EQ(cov(vy, vy), cov(y, y));
  EXPECT_EQ(cov(y, vy), cov(y, y));
}

} // namespace albatross
