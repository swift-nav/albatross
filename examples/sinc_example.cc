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

#include "gflags/gflags.h"

#include "csv.h"
#include <fstream>
#include <iostream>

#include "sinc_example_utils.h"

#include "Tune"

DEFINE_string(input, "", "path to csv containing input data.");
DEFINE_string(output, "", "path where predictions will be written in csv.");
DEFINE_string(n, "10", "number of training points to use.");
DEFINE_bool(tune, false, "a flag indication parameters should be tuned first.");

using albatross::ParameterStore;
using albatross::RegressionDataset;
using albatross::get_tuner;

template <typename ModelType>
albatross::ParameterStore tune_model(ModelType &model,
                                     RegressionDataset<double> &data) {
  /*
   * Now we tune the model by finding the hyper parameters that
   * maximize the likelihood (or minimize the negative log likelihood).
   */
  std::cout << "Tuning the model." << std::endl;

  albatross::LeaveOneOutLikelihood<> loo_nll;

  return get_tuner(model, loo_nll, data).tune();
}

int main(int argc, char *argv[]) {

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int n = std::stoi(FLAGS_n);
  const double low = -3.;
  const double high = 13.;
  const double meas_noise = 1.;

  if (FLAGS_input == "") {
    FLAGS_input = "input.csv";
  }
  maybe_create_training_data(FLAGS_input, n, low, high, meas_noise);

  using namespace albatross;

  std::cout << "Reading the input data." << std::endl;
  RegressionDataset<double> data = read_csv_input(FLAGS_input);

  std::cout << "Defining the model." << std::endl;
  using Noise = IndependentNoise<double>;
  using SquaredExp = SquaredExponential<EuclideanDistance>;

  Polynomial<1> polynomial(100.);
  Noise noise(meas_noise);
  SquaredExp squared_exponential(3.5, 5.7);
  auto cov = polynomial + noise + squared_exponential;

  std::cout << cov.pretty_string() << std::endl;

  auto model = gp_from_covariance(cov);

  if (FLAGS_tune) {
    model.set_params(tune_model(model, data));
  }

  std::cout << pretty_param_details(model.get_params()) << std::endl;
  const auto fit_model = model.fit(data);

  /*
   * Make predictions at a bunch of locations which we can then
   * visualize if desired.
   */
  write_predictions_to_csv(FLAGS_output, fit_model, low, high);
}
