/*
 * Copyright (C) 2020 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <albatross/Tune>
#include <csv.h>
#include <fstream>
#include <gflags/gflags.h>
#include <iostream>

#include "sinc_example_utils.h"

DEFINE_string(input, "", "path to csv containing input data.");
DEFINE_string(output, "", "path where predictions will be written in csv.");
DEFINE_string(n, "30", "number of training points to use.");
DEFINE_string(mode, "radial", "which modelling approach to use.");
DEFINE_bool(tune, false, "a flag indication parameters should be tuned first.");

using albatross::get_tuner;
using albatross::ParameterStore;
using albatross::RegressionDataset;

template <typename ModelType>
void run_model(ModelType &model, RegressionDataset<double> &data, double low,
               double high) {

  if (FLAGS_tune) {
    albatross::LeaveOneOutLikelihood<> loo_nll;
    model.set_params(get_tuner(model, loo_nll, data).tune());
  }

  std::cout << pretty_param_details(model.get_params()) << std::endl;
  const auto fit_model = model.fit(data);

  std::ofstream output;
  output.open(FLAGS_output);

  const std::size_t k = 161;
  auto grid_xs = uniform_points_on_line(k, low - 1., high + 1.);

  std::vector<albatross::Derivative<double>> derivs;
  for (const auto &d : grid_xs) {
    derivs.emplace_back(d);
  }

  auto prediction =
      fit_model.predict_with_measurement_noise(derivs).marginal();

  Eigen::VectorXd targets(static_cast<Eigen::Index>(k));
  for (std::size_t i = 0; i < k; i++) {
    targets[static_cast<Eigen::Index>(i)] = std::sin(grid_xs[i]);
  }

  const albatross::RegressionDataset<double> dataset(grid_xs, targets);

  albatross::write_to_csv(output, dataset, prediction);

  output.close();
}

int main(int argc, char *argv[]) {

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  using namespace albatross;

  int n = std::stoi(FLAGS_n);
  const double low = 0.;
  const double high = 10.;
  const double meas_noise_sd = std::sqrt(std::numeric_limits<double>::epsilon());

  auto xs = random_points_on_line(n, low, high);

  std::default_random_engine generator;
  std::normal_distribution<double> noise_distribution(0., meas_noise_sd);

  Eigen::VectorXd ys(n);

  for (int i = 0; i < n; i++) {
    double noise = noise_distribution(generator);
    ys[i] = std::sin(xs[i]) + noise;
  }

  albatross::RegressionDataset<double> dataset(xs, ys);

  IndependentNoise<double> indep_noise(meas_noise_sd);
  indep_noise.sigma_independent_noise = {meas_noise_sd, FixedPrior()};

  // this approach uses a squared exponential radial function to capture
  // the function we're estimating using non-parametric techniques
  const SquaredExponential<EuclideanDistance> squared_exponential(3.5, 100.);
  auto cov = squared_exponential + measurement_only(indep_noise);
  auto model = gp_from_covariance(cov);

  std::map<std::string, double> params = {
      {"sigma_squared_exponential", 0.398683},
      {"squared_exponential_length_scale", 3.42727},
  };

  model.set_param_values(params);

  run_model(model, dataset, low, high);

}
