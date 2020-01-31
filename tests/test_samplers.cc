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

#include <gtest/gtest.h>

#include <albatross/Core>
#include <albatross/Samplers>
#include <albatross/SparseGP>

#include <fstream>

#include "test_models.h"

namespace albatross {

TEST(test_samplers, test_samplers_from_normal_distribution) {

  const double sd = M_PI;

  auto gaussian_ll = [&](const std::vector<double> &xs) {
    assert(xs.size() == 1);
    return -0.5 * std::pow(xs[0] / sd, 2.);
  };

  std::default_random_engine gen(2012);
  std::size_t walkers = 10;

  std::normal_distribution<double> jitter_distribution(0., 0.1 * sd);
  std::vector<std::vector<double>> initial_params;
  for (std::size_t i = 0; i < walkers; ++i) {
    initial_params.push_back({jitter_distribution(gen)});
  }

  std::size_t burn_in = 100;
  std::size_t thin = 10;
  std::size_t max_iterations = 2000;
  const auto ensemble_samples =
      ensemble_sampler(gaussian_ll, initial_params, max_iterations, gen);

  std::vector<double> cdfs;
  for (std::size_t i = burn_in; i < ensemble_samples.size(); i += thin) {
    for (const auto &sample : ensemble_samples[i]) {
      const double value = sample.params[0];
      const double chi_squared_sample = std::pow(value / sd, 2.);
      cdfs.push_back(chi_squared_cdf(chi_squared_sample, 1));
    }
  }

  EXPECT_LT(*std::min_element(cdfs.begin(), cdfs.end()), 0.1);
  EXPECT_GT(*std::max_element(cdfs.begin(), cdfs.end()), 0.9);
  double ks = uniform_ks_test(cdfs);
  EXPECT_LT(ks, 0.05);
}

TEST(test_samplers, test_samplers_from_uniform_distribution) {

  auto uniform_ll = [&](const std::vector<double> &xs) {
    assert(xs.size() == 1);
    if (xs[0] >= 0. && xs[0] <= 1.) {
      return 0.;
    } else {
      return -LARGE_VAL;
    }
  };

  std::default_random_engine gen(2012);
  std::size_t walkers = 10;

  std::uniform_real_distribution<double> jitter_distribution(0., 1.);
  std::vector<std::vector<double>> initial_params;
  for (std::size_t i = 0; i < walkers; ++i) {
    initial_params.push_back({jitter_distribution(gen)});
  }

  // Insert some invalid initial parameters to make sure
  // we properly initialize the sampler.
  initial_params[0][0] = -1.;
  initial_params[initial_params.size() - 1][0] = 10.;

  std::size_t burn_in = 100;
  std::size_t thin = 10;
  std::size_t max_iterations = 2000;
  const auto ensemble_samples =
      ensemble_sampler(uniform_ll, initial_params, max_iterations, gen);

  std::vector<double> cdfs;
  for (std::size_t i = burn_in; i < ensemble_samples.size(); i += thin) {
    for (const auto &sample : ensemble_samples[i]) {
      const double value = sample.params[0];
      cdfs.push_back(value);
    }
  }

  EXPECT_LT(*std::min_element(cdfs.begin(), cdfs.end()), 0.1);
  EXPECT_GT(*std::max_element(cdfs.begin(), cdfs.end()), 0.9);
  double ks = uniform_ks_test(cdfs);
  EXPECT_LT(ks, 0.05);
}

TEST(test_samplers, test_samplers_gp) {
  const double a = 3.14;
  const double b = sqrt(2.);
  const double meas_noise_sd = 1.;
  const auto dataset = make_toy_linear_data(a, b, meas_noise_sd, 10);

  std::default_random_engine gen(2012);
  const std::size_t walkers = 10;

  using Noise = IndependentNoise<double>;
  Noise indep_noise(meas_noise_sd);
  indep_noise.sigma_independent_noise.prior = LogScaleUniformPrior(1e-3, 1e2);
  LinearMean linear;

  auto model = gp_from_covariance_and_mean(indep_noise, linear);

  std::size_t max_iterations = 100;

  std::shared_ptr<std::ostringstream> oss =
      std::make_shared<std::ostringstream>();
  std::shared_ptr<std::ostream> ostream =
      static_cast<std::shared_ptr<std::ostream>>(oss);

  auto callback = get_csv_writing_callback(model, ostream);

  const auto ensemble_samples =
      ensemble_sampler(model, dataset, walkers, max_iterations, gen, callback);

  EXPECT_GT(oss->str().size(), 1);
}

inline long int get_group(const double &f) {
  return static_cast<double>(floor(f / 5.));
}

struct LeaveOneIntervalOut {
  long int operator()(const double &f) const { return get_group(f); }
};

TEST(test_samplers, test_samplers_sparse_gp) {
  const double a = 3.14;
  const double b = sqrt(2.);
  const double meas_noise_sd = 1.;
  const auto dataset = make_toy_linear_data(a, b, meas_noise_sd, 50);

  using Noise = IndependentNoise<double>;
  Noise indep_noise(meas_noise_sd);
  indep_noise.sigma_independent_noise.value = 1.;
  indep_noise.sigma_independent_noise.prior = LogScaleUniformPrior(1e-3, 1e2);
  auto meas_noise = measurement_only(indep_noise);
  LinearMean linear;
  linear.offset.value = a;
  linear.slope.value = b;

  LeaveOneIntervalOut grouper;
  UniformlySpacedInducingPoints strategy(5);

  auto model = sparse_gp_from_covariance_and_mean(meas_noise, linear, grouper,
                                                  strategy, "test");
  model.set_prior("inducing_nugget", FixedPrior());
  model.set_prior("measurement_nugget", FixedPrior());

  std::default_random_engine gen(2012);
  const std::size_t walkers = 32;

  std::size_t max_iterations = 100;

  std::shared_ptr<std::ostringstream> oss =
      std::make_shared<std::ostringstream>();
  std::shared_ptr<std::ostream> os =
      static_cast<std::shared_ptr<std::ostream>>(oss);

  auto csv_callback = CsvWritingCallback(model.get_params(), os);
  const auto samples = ensemble_sampler(model, dataset, walkers, max_iterations,
                                        gen, csv_callback);

  EXPECT_GT(samples.size(), 0);
  EXPECT_GT(oss->str().size(), 1);

  std::istringstream iss(oss->str());
  const auto initial_params = initial_params_from_csv(model.get_params(), iss);

  EXPECT_EQ(initial_params.size(), samples[0].size());

  oss = std::make_shared<std::ostringstream>();
  auto ml_callback = MaximumLikelihoodTrackingCallback(model.get_params(), oss);
  ensemble_sampler(model, dataset, walkers, max_iterations, gen, ml_callback);
}

} // namespace albatross
