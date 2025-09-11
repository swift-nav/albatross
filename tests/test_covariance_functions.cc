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
#include <gtest/gtest.h>

#include <albatross/Core>
#include <albatross/CovarianceFunctions>

#include "test_covariance_utils.h"

namespace albatross {

std::vector<Eigen::Vector3d> points_on_a_line(const int n) {
  std::vector<Eigen::Vector3d> xs;
  for (Eigen::Index i = 0; i < n; i++) {
    Eigen::Vector3d x;
    for (Eigen::Index j = 0; j < 3; j++) {
      x[j] = 1000 * cast::to_double(i + j);
    }
    xs.push_back(x);
  }
  return xs;
}

TEST(test_covariance_functions, test_measurement_noise_wrapper) {
  SquaredExponential<EuclideanDistance> radial;
  IndependentNoise<double> noise;
  auto meas_noise = measurement_only(noise);
  auto sum = radial + meas_noise;
  auto prod = meas_noise * radial;
  auto prod_of_sum = noise * sum;

  std::vector<double> features = {0., 1., 2.};

  std::vector<Measurement<double>> measurements;
  for (const auto &f : features) {
    measurements.emplace_back(Measurement<double>(f));
  }

  const auto f = features[0];
  const auto m = measurements[0];

  // The measurement noise should only get applied to
  // features marked as measurements.
  EXPECT_EQ(meas_noise(f, f), 0.);
  EXPECT_EQ(meas_noise(f, m), 0.);
  EXPECT_EQ(meas_noise(m, f), 0.);
  EXPECT_GT(meas_noise(m, m), 0.);

  // The radial covariance function should behave the same
  // regardless of whether it's given features or measurements.
  EXPECT_GT(radial(f, f), 0.);
  EXPECT_EQ(radial(m, m), radial(f, f));
  EXPECT_EQ(radial(m, f), radial(f, f));
  EXPECT_EQ(radial(f, m), radial(f, f));

  // When you add covariance functions you should get
  // the sum of the individual calls.
  EXPECT_GT(sum(f, f), 0.);
  EXPECT_GT(sum(m, m), 0.);
  EXPECT_GT(sum(m, m), sum(f, f));
  EXPECT_EQ(sum(m, m), radial(m, m) + meas_noise(m, m));
  EXPECT_EQ(sum(m, f), radial(m, f) + meas_noise(m, f));
  EXPECT_EQ(sum(f, m), radial(f, m) + meas_noise(f, m));

  // When you multiply a measurement only covariance with a
  // fully defined one the measurement only property
  // propagates
  EXPECT_EQ(prod(f, f), 0.);
  EXPECT_GT(prod(m, m), 0.);
  EXPECT_EQ(prod(m, m), radial(m, m) * meas_noise(m, m));
  EXPECT_EQ(prod(m, f), 0.);
  EXPECT_EQ(prod(f, m), 0.);

  // Taking a product of a sum, the sum should drop the
  // measurement only behavior, so the product should then
  // still be non zero for non-measurement features.
  EXPECT_GT(prod_of_sum(f, f), 0.);
  EXPECT_GT(prod_of_sum(m, m), 0.);
  EXPECT_EQ(prod_of_sum(f, f), noise(f, f) * sum(f, f));
  EXPECT_EQ(prod_of_sum(m, m), noise(m, m) * sum(m, m));
  EXPECT_EQ(prod_of_sum(m, f), prod_of_sum(f, f));
  EXPECT_EQ(prod_of_sum(f, m), prod_of_sum(m, f));
}

TEST(test_covariance_functions, test_build_covariance) {
  using Feature = Eigen::Vector3d;
  IndependentNoise<Feature> noise;
  SquaredExponential<EuclideanDistance> sqr_exp;
  Exponential<RadialDistance> radial_exp;

  auto product = radial_exp * sqr_exp;
  auto covariance_function = product + noise;

  auto xs = points_on_a_line(5);
  Eigen::MatrixXd C = covariance_function(xs);
  assert(C.rows() == cast::to_index(xs.size()));
  assert(C.cols() == cast::to_index(xs.size()));
}

/*
 * In the following we test any covariance functions which should support
 * Eigen::Vector feature vectors.
 */
template <typename T>
class TestVectorCovarianceFunctions : public ::testing::Test {
public:
  T covariance_function;
};

typedef ::testing::Types<
    SquaredExponential<EuclideanDistance>, SquaredExponential<RadialDistance>,
    Exponential<EuclideanDistance>, Exponential<AngularDistance>,
    Exponential<RadialDistance>>
    VectorCompatibleCovarianceFunctions;

TYPED_TEST_SUITE(TestVectorCovarianceFunctions,
                 VectorCompatibleCovarianceFunctions);

TYPED_TEST(TestVectorCovarianceFunctions, WorksWithEigen) {
  const auto xs = points_on_a_line(5);
  Eigen::MatrixXd C = this->covariance_function(xs);
  assert(C.rows() == cast::to_index(xs.size()));
  assert(C.cols() == cast::to_index(xs.size()));
  // Make sure C is positive definite.
  C.inverse();
}

TYPED_TEST(TestVectorCovarianceFunctions, WorksDirectlyOnCovarianceterms) {
  auto xs = points_on_a_line(5);
  Eigen::MatrixXd C = this->covariance_function(xs);
  assert(C.rows() == cast::to_index(xs.size()));
  assert(C.cols() == cast::to_index(xs.size()));
  // Make sure C is positive definite.
  C.inverse();
}

TYPED_TEST(TestVectorCovarianceFunctions, can_set_params) {
  const ParameterStore params(this->covariance_function.get_params());

  double to_add = 3.14159;
  for (const auto &pair : params) {
    this->covariance_function.set_param_value(pair.first,
                                              pair.second.value + to_add);
    EXPECT_DOUBLE_EQ(this->covariance_function.get_param_value(pair.first),
                     pair.second.value + to_add);
  }
}

class DummyCovariance : public CovarianceFunction<DummyCovariance> {
public:
  DummyCovariance(double foo_ = sqrt(2.), double bar_ = log(2.)) {
    foo.value = foo_;
    bar.value = bar_;
  }

  ALBATROSS_DECLARE_PARAMS(foo, bar)

  double _call_impl(const double &, const double &) const {
    return foo.value + bar.value;
  }

  std::string name_ = "dummy";
};

/*
 * In the following we test any covariance functions which should support
 * Eigen::Vector feature vectors.
 */
template <typename T>
class TestDoubleCovarianceFunctions : public ::testing::Test {
public:
  T covariance_function;
};

typedef ::testing::Types<
    DummyCovariance, IndependentNoise<double>, Nugget, Polynomial<2>,
    SumOfCovarianceFunctions<IndependentNoise<double>, Polynomial<2>>,
    SumOfCovarianceFunctions<IndependentNoise<double>, DummyCovariance>>
    DoubleCompatibleCovarianceFunctions;

TYPED_TEST_SUITE(TestDoubleCovarianceFunctions,
                 DoubleCompatibleCovarianceFunctions);

TYPED_TEST(TestDoubleCovarianceFunctions, works_with_eigen) {
  const auto xs = points_on_a_line(5);
  std::vector<double> features;
  const auto x_size = cast::to_index(xs.size());
  for (Eigen::Index i = 0; i < x_size; ++i) {
    features.push_back(xs[cast::to_size(i)][0]);
  }

  Eigen::MatrixXd C = this->covariance_function(features);
  assert(C.rows() == x_size);
  assert(C.cols() == x_size);
  // Make sure C is positive definite.
  C.inverse();
}

TYPED_TEST(TestDoubleCovarianceFunctions, can_set_params) {
  const ParameterStore params(this->covariance_function.get_params());

  const double to_add = 3.14159;
  for (const auto &pair : params) {
    this->covariance_function.set_param_value(pair.first,
                                              pair.second.value + to_add);
    EXPECT_DOUBLE_EQ(this->covariance_function.get_param_value(pair.first),
                     pair.second.value + to_add);
  }
}

class SsrX : public CovarianceFunction<SsrX> {
public:
  std::vector<X> _ssr_impl(const std::vector<double> &) const { return {X()}; }
};

class SsrY : public CovarianceFunction<SsrY> {
public:
  std::vector<Y> _ssr_impl(const std::vector<double> &) const { return {Y()}; }
};

TEST(test_covariance_functions, test_state_space_representation) {
  SsrX with_x;
  SsrY with_y;

  auto with_both = with_x + with_y;

  std::vector<double> features = {0.};

  auto xs = with_x.state_space_representation(features);
  EXPECT_TRUE(bool(std::is_same<decltype(xs), std::vector<X>>::value));

  auto ys = with_y.state_space_representation(features);
  EXPECT_TRUE(bool(std::is_same<decltype(ys), std::vector<Y>>::value));

  auto xs_and_ys = with_both.state_space_representation(features);
  EXPECT_TRUE(bool(
      std::is_same<decltype(xs_and_ys), std::vector<variant<X, Y>>>::value));
}

struct IntEquivalent {
  int value;

  bool operator<(const IntEquivalent &other) const {
    return value < other.value;
  }
};

class HasEquivalent : public CovarianceFunction<HasEquivalent> {
public:
  double _call_impl(const int &x, const int &y) const {
    return 1.0 * (x == y);
  };

  double _call_impl(const IntEquivalent &x, const int &y) const {
    return this->_call_impl(x.value, y);
  }

  double _call_impl(const IntEquivalent &x, const IntEquivalent &y) const {
    return this->_call_impl(x.value, y.value);
  }
};

TEST(test_covariance_functions, test_covariance_utils) {
  HasEquivalent cov;

  std::vector<int> features = {0, 1, 2, 3, 3, 2, 1, 5};

  auto to_equivalent = [](const int &x) { return IntEquivalent{x}; };

  expect_converted_feature_equivalence(cov, features, to_equivalent);

  std::set<IntEquivalent> unique_equivalents;
  for (const auto &f : features) {
    unique_equivalents.emplace(IntEquivalent{f});
  }
  std::vector<IntEquivalent> inducing_points(unique_equivalents.begin(),
                                             unique_equivalents.end());

  auto int_grouper = [](const int &x) { return x; };

  expect_inducing_points_capture_cross_correlation(
      cov, features, inducing_points, int_grouper);

  expect_zero_covariance_across_groups(cov, features, int_grouper);

  expect_non_zero_covariance_within_groups(cov, features, int_grouper);
}

TEST(test_covariance_functions, test_nugget) {
  struct Foo {
    Foo(int x_) : x(x_) {}
    bool operator==(const Foo &other) const { return x == other.x; }
    int x;
  };

  int x_int = 1;
  int y_int = 2;

  Foo x_foo(x_int);
  Foo y_foo(y_int);

  Nugget nugget;
  const double expected = nugget.nugget_sigma.value * nugget.nugget_sigma.value;

  EXPECT_GT(nugget(x_int, x_int), 0.);

  EXPECT_EQ(nugget(x_int, x_int), expected);
  EXPECT_EQ(nugget(x_foo, x_foo), expected);

  EXPECT_EQ(nugget(y_int, y_int), expected);
  EXPECT_EQ(nugget(y_foo, y_foo), expected);

  EXPECT_EQ(nugget(x_int, y_int), 0.);
  EXPECT_EQ(nugget(x_foo, y_foo), 0.);
}

} // namespace albatross
