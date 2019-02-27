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

#include <map>
#include <string>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <functional>
#include <memory>
#include <vector>

#include "core/declarations.h"
#include "core/distribution.h"

#include "core/traits.h"

//#include "Core"
//#include "CovarianceFunctions"

namespace albatross {

std::vector<Eigen::Vector3d> points_on_a_line(const int n) {
  std::vector<Eigen::Vector3d> xs;
  for (int i = 0; i < n; i++) {
    Eigen::Vector3d x;
    for (int j = 0; j < 3; j++)
      x[static_cast<std::size_t>(j)] = 1000 * i + j;
    xs.push_back(x);
  }
  return xs;
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
  assert(C.rows() == static_cast<Eigen::Index>(xs.size()));
  assert(C.cols() == static_cast<Eigen::Index>(xs.size()));
}

///*
// * In the following we test any covariance functions which should support
// * Eigen::Vector feature vectors.
// */
//template <typename T>
//class TestVectorCovarianceFunctions : public ::testing::Test {
//
//public:
//  T covariance_function;
//};
//
//typedef ::testing::Types<
//    SquaredExponential<EuclideanDistance>, SquaredExponential<RadialDistance>,
//    Exponential<EuclideanDistance>, Exponential<AngularDistance>,
//    Exponential<RadialDistance>>
//    VectorCompatibleCovarianceFunctions;
//
//TYPED_TEST_CASE(TestVectorCovarianceFunctions,
//                VectorCompatibleCovarianceFunctions);
//
//TYPED_TEST(TestVectorCovarianceFunctions, WorksWithEigen) {
//  const auto xs = points_on_a_line(5);
//  Eigen::MatrixXd C = this->covariance_function(xs);
//  assert(C.rows() == static_cast<Eigen::Index>(xs.size()));
//  assert(C.cols() == static_cast<Eigen::Index>(xs.size()));
//  // Make sure C is positive definite.
//  C.inverse();
//}
//
//TYPED_TEST(TestVectorCovarianceFunctions, WorksDirectlyOnCovarianceterms) {
//  auto xs = points_on_a_line(5);
//  Eigen::MatrixXd C = this->covariance_function(xs);
//  assert(C.rows() == static_cast<Eigen::Index>(xs.size()));
//  assert(C.cols() == static_cast<Eigen::Index>(xs.size()));
//  // Make sure C is positive definite.
//  C.inverse();
//}
//
//TYPED_TEST(TestVectorCovarianceFunctions, can_set_params) {
//
//  const ParameterStore params(this->covariance_function.get_params());
//
//  double to_add = 3.14159;
//  for (const auto &pair : params) {
//    this->covariance_function.set_param_value(pair.first,
//                                              pair.second.value + to_add);
//    EXPECT_DOUBLE_EQ(this->covariance_function.get_param_value(pair.first),
//                     pair.second.value + to_add);
//  }
//}
//
//class DummyCovariance : public CovarianceFunction<DummyCovariance> {
//public:
//  DummyCovariance(double foo_ = sqrt(2.), double bar_ = log(2.)) {
//    foo.value = foo_;
//    bar.value = bar_;
//  }
//
//  ALBATROSS_DECLARE_PARAMS(foo, bar)
//
//  double call_impl_(const double &, const double &) const {
//    return foo.value + bar.value;
//  }
//
//  std::string name_ = "dummy";
//};
//
///*
// * In the following we test any covariance functions which should support
// * Eigen::Vector feature vectors.
// */
//template <typename T>
//class TestDoubleCovarianceFunctions : public ::testing::Test {
//
//public:
//  T covariance_function;
//};
//
//typedef ::testing::Types<
//    DummyCovariance, IndependentNoise<double>, Polynomial<2>,
//    SumOfCovarianceFunctions<IndependentNoise<double>, Polynomial<2>>,
//    SumOfCovarianceFunctions<IndependentNoise<double>, DummyCovariance>>
//    DoubleCompatibleCovarianceFunctions;
//
//TYPED_TEST_CASE(TestDoubleCovarianceFunctions,
//                DoubleCompatibleCovarianceFunctions);
//
//TYPED_TEST(TestDoubleCovarianceFunctions, works_with_eigen) {
//  auto xs = points_on_a_line(5);
//  std::vector<double> features;
//  const auto x_size = static_cast<Eigen::Index>(xs.size());
//  for (Eigen::Index i = 0; i < x_size; ++i) {
//    features.push_back(xs[i][0]);
//  }
//
//  Eigen::MatrixXd C = this->covariance_function(features);
//  assert(C.rows() == x_size);
//  assert(C.cols() == x_size);
//  // Make sure C is positive definite.
//  C.inverse();
//}
//
//TYPED_TEST(TestDoubleCovarianceFunctions, can_set_params) {
//
//  const ParameterStore params(this->covariance_function.get_params());
//
//  double to_add = 3.14159;
//  for (const auto &pair : params) {
//    this->covariance_function.set_param_value(pair.first,
//                                              pair.second.value + to_add);
//    EXPECT_DOUBLE_EQ(this->covariance_function.get_param_value(pair.first),
//                     pair.second.value + to_add);
//  }
//}

} // namespace albatross
