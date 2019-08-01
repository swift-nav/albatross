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

#include <albatross/Evaluation>
#include <gtest/gtest.h>

#include "test_utils.h"

namespace albatross {

/*
 * This test data describes the following situation.
 *
 * Assume you have some thin translucent sheet which attenuates
 * the intensity of light passing through it but with an unknown
 * attenuation.  You then make some measurements of the attenuation
 * by emitting light and measuring the intesity at different
 * locations on the other side of the sheet,
 *
 *        0  emitter (at x = 1)
 *       ...
 *      . . .
 * ====.==.==.===== <- thin sheet
 *    .   .   .
 *   .    .    .
 *  .     .     .
 * ^      ^      ^  collectors
 *
 * 0------1------2
 * x axis ->
 *
 * In this case you have multiple measurements, y, each made at different
 * locations, x.  Each measurement passes through the sheet at a slightly
 * different but deterministically known angle.  We can then write
 * the amount of observed attenuation, y, as
 *
 *   y = f(x) * z + u
 *
 * where f(x) descibes how much longer the signal spends in the material
 * as a function of the location, x, of the collector. z is the unknown
 * attenuation and u is noise.
 *
 * The amount of material each ray passes through relative to the
 * width is given by 1 / (cos(theta)) where theta = atan(x / 1)
 *
 */

double obliquity_function(double x) { return 1. / cos(atan(x - 1.)); }

class ObliquityScaling : public ScalingFunction {
public:
  ObliquityScaling() : ScalingFunction(){};

  std::string get_name() const { return "obliquity_scaling"; }

  double _call_impl(const double &x) const { return obliquity_function(x); }
};

auto make_attenuation_data(const double attenuation = 3.14159,
                           const double sigma_noise = 0.01) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  gen.seed(3);
  std::normal_distribution<> d{0., sigma_noise};

  std::size_t n = 10;
  std::vector<double> features;
  Eigen::VectorXd targets(n);

  for (std::size_t i = 0; i < n; i++) {
    // x ranges from 0 to 2
    double x = static_cast<double>(i) * (2. / static_cast<double>(n));
    features.push_back(x);
    targets[i] = obliquity_function(x) * attenuation + d(gen);
  }

  return RegressionDataset<double>(features, targets);
}

/*
 * This test makes sure that we can make predictions of what
 * the attenuation of a signal would be at some unobserved location.
 */
TEST(test_scaling_functions, test_predicts) {
  using Feature = double;
  using Noise = IndependentNoise<Feature>;

  Constant constant(10.);
  Noise noise(0.01);
  using TestScalingTerm = ScalingTerm<ObliquityScaling>;
  TestScalingTerm scaling;

  auto dataset = make_attenuation_data();

  // This will create a covariance function that represents some constant
  // unknown value, that is scaled according to a known deterministic
  // function, then noisy measurements are taken.
  auto covariance_function = constant * scaling + noise;

  auto model = gp_from_covariance(covariance_function);

  RootMeanSquareError rmse;
  LeaveOneOut loo;
  auto cv_scores = model.cross_validate().scores(rmse, dataset, loo);
  EXPECT_LE(cv_scores.mean(), 0.01);
}

/*
 * This test make sure (and illustrates how) we can perform inference
 * on the unknown attenuation constant of the material in the test
 * case described above.
 */
TEST(test_scaling_functions, test_inference) {
  using Feature = double;
  using Noise = IndependentNoise<Feature>;

  double attenuation = 3.14159;
  double sigma = 0.01;

  Constant constant(2 * attenuation);
  Noise noise(sigma);

  using TestScalingTerm = ScalingTerm<ObliquityScaling>;
  TestScalingTerm scaling;

  auto dataset = make_attenuation_data(attenuation, sigma);

  // This will create a covariance function that represents some constant
  // unknown value, that is scaled according to a known deterministic
  // function, then noisy measurements are taken.
  auto covariance_function = constant * scaling + noise;

  auto model = gp_from_covariance(covariance_function);

  auto state_space = constant.get_state_space_representation(dataset.features);
  auto state_estimate = model.fit(dataset).predict(state_space);
  // Make sure our estimate of the attenuation term is close, despite the fact
  // that we made scaled observations of it.
  EXPECT_LE(fabs(state_estimate.mean()[0] - attenuation), 1e-2);
}

class DummyCovariance : public CovarianceFunction<DummyCovariance> {
public:
  template <typename X, typename Y>
  double _call_impl(const X &x, const Y &y) const {
    return 0.;
  }
};

/*
 * This test make sure (and illustrates how) we can perform inference
 * on the unknown attenuation constant of the material in the test
 * case described above.
 */
TEST(test_scaling_functions, test_operations) {
  double sigma = 0.01;

  using TestScalingTerm = ScalingTerm<ObliquityScaling>;
  TestScalingTerm scaling;

  using DoubleNoise = IndependentNoise<double>;
  DoubleNoise double_noise(sigma);

  DummyCovariance dummy;

  auto rhs_cov_func = double_noise * scaling + dummy;

  double a = 0.;
  double b = 1.;

  struct X {};
  struct Y {};

  X x;
  Y y;

  EXPECT_EQ(rhs_cov_func(x, y), 0.);
  EXPECT_EQ(rhs_cov_func(x, x), 0.);
  EXPECT_EQ(rhs_cov_func(x, y), 0.);
  EXPECT_EQ(rhs_cov_func(a, y), 0.);
  EXPECT_EQ(rhs_cov_func(a, x), 0.);
  EXPECT_EQ(rhs_cov_func(a, b), 0.);
  EXPECT_GT(rhs_cov_func(a, a), 0.);

  /*
   * The scaling functions need to be defined differently
   * for left hand side (LHS) and right hand side (RHS)
   * products, so we test both.
   */
  auto lhs_cov_func = scaling * double_noise + dummy;

  EXPECT_EQ(lhs_cov_func(x, y), 0.);
  EXPECT_EQ(lhs_cov_func(x, x), 0.);
  EXPECT_EQ(lhs_cov_func(x, y), 0.);
  EXPECT_EQ(lhs_cov_func(a, y), 0.);
  EXPECT_EQ(lhs_cov_func(a, x), 0.);
  EXPECT_EQ(lhs_cov_func(a, b), 0.);
  EXPECT_GT(lhs_cov_func(a, a), 0.);
}

} // namespace albatross
