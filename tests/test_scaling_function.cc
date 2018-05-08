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

#include "covariance_functions/covariance_functions.h"
#include "test_utils.h"
#include <gtest/gtest.h>
#include <iostream>

namespace albatross {

/*
 * This test data descibes the following situation.
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

  ~ObliquityScaling(){};

  std::string get_name() const { return "obliquity_scaling"; }

  double operator()(const double &x) const { return obliquity_function(x); }
};

auto make_attenuation_data(const double attenuation = 3.14159,
                           const double sigma_noise = 0.01) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  gen.seed(3);
  std::normal_distribution<> d{0., sigma_noise};

  s32 n = 10;
  std::vector<double> features;
  Eigen::VectorXd targets(n);

  for (s32 i = 0; i < n; i++) {
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

  CovarianceFunction<Constant> constant = {Constant(10.)};
  CovarianceFunction<Noise> noise = {Noise(0.01)};
  using TestScalingTerm = ScalingTerm<ObliquityScaling>;
  CovarianceFunction<TestScalingTerm> scaling = {TestScalingTerm()};

  auto dataset = make_attenuation_data();

  // This will create a covariance function that represents some constant
  // unknown value, that is scaled according to a known deterministic
  // function, then noisy measurements are taken.
  auto covariance_function = constant * scaling + noise;

  auto model = gp_from_covariance<double>(covariance_function);

  auto folds = leave_one_out(dataset);
  auto cv_scores =
      cross_validated_scores(root_mean_square_error, folds, &model);

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

  CovarianceFunction<Constant> constant = {Constant(2 * attenuation)};
  CovarianceFunction<Noise> noise = {Noise(sigma)};
  using TestScalingTerm = ScalingTerm<ObliquityScaling>;
  CovarianceFunction<TestScalingTerm> scaling = {TestScalingTerm()};

  auto dataset = make_attenuation_data(attenuation, sigma);

  // This will create a covariance function that represents some constant
  // unknown value, that is scaled according to a known deterministic
  // function, then noisy measurements are taken.
  auto covariance_function = constant * scaling + noise;

  auto model = gp_from_covariance<double>(covariance_function);

  auto state_space =
      constant.term.get_state_space_representation(dataset.features);
  model.fit(dataset);
  auto state_estimate = model.inspect(state_space);
  // Make sure our estimate of the attenuation term is close, despite the fact
  // that we made scaled observations of it.
  EXPECT_LE(fabs(state_estimate.mean[0] - attenuation), 1e-2);
}
}
