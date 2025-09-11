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

#include <csv.h>
#include <gflags/gflags.h>
#include <albatross/Tune>
#include <fstream>
#include <iostream>

#include "sinc_example_utils.h"

DEFINE_string(input, "", "path to csv containing input data.");
DEFINE_string(output, "", "path where predictions will be written in csv.");
DEFINE_string(n, "10", "number of training points to use.");
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

  /*
   * Make predictions at a bunch of locations which we can then
   * visualize if desired.
   */
  write_predictions_to_csv(FLAGS_output, fit_model, low, high);
}

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int n = std::stoi(FLAGS_n);
  const double low = -10.;
  const double high = 23.;
  const double meas_noise_sd = 1.;

  if (FLAGS_input == "") {
    FLAGS_input = "input.csv";
  }
  maybe_create_training_data(FLAGS_input, n, low, high, meas_noise_sd);

  using namespace albatross;

  std::cout << "Reading the input data." << std::endl;
  RegressionDataset<double> data = read_csv_input(FLAGS_input);

  std::cout << "Defining the model." << std::endl;

  IndependentNoise<double> indep_noise(meas_noise_sd);
  indep_noise.sigma_independent_noise = {1., LogScaleUniformPrior(1e-3, 1e2)};

  if (FLAGS_mode == "radial_only") {
    // this approach uses a squared exponential radial function to capture
    // the function we're estimating using non-parametric techniques
    const SquaredExponential<EuclideanDistance> squared_exponential(3.5, 100.);
    auto cov = squared_exponential + measurement_only(indep_noise);
    auto model = gp_from_covariance(cov);
    run_model(model, data, low, high);
  } else if (FLAGS_mode == "radial") {
    // this approach uses a squared exponential radial function to capture
    // the function we're estimating using non-parametric techniques
    const Polynomial<1> linear(100.);
    const SquaredExponential<EuclideanDistance> squared_exponential(3.5, 5.7);
    auto cov = linear + squared_exponential + measurement_only(indep_noise);
    auto model = gp_from_covariance(cov);
    run_model(model, data, low, high);
  } else if (FLAGS_mode == "parametric") {
    // Here we assume we know the "truth" is made up of a linear trend and
    // a scaled and translated sinc function with added noise and capture
    // this all through the use of a mean function.
    const LinearMean linear;
    const SincFunction sinc;
    auto model = gp_from_covariance_and_mean(measurement_only(indep_noise),
                                             linear + sinc);
    run_model(model, data, low, high);
  }
}
