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

#include <albatross/Tune>

#include <albatross/SparseGP>

#include <csv.h>
#include <fstream>
#include <gflags/gflags.h>
#include <iostream>

#define EXAMPLE_SLOPE_VALUE 0.
#define EXAMPLE_CONSTANT_VALUE 0.

#include "sinc_example_utils.h"

DEFINE_string(input, "", "path to csv containing input data.");
DEFINE_string(output, "", "path where predictions will be written in csv.");
DEFINE_int32(n, 10, "number of training points to use.");
DEFINE_int32(k, 5, "number of training points to use.");
DEFINE_bool(tune, false, "a flag indication parameters should be tuned first.");

using albatross::get_tuner;
using albatross::ParameterStore;
using albatross::RegressionDataset;

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

  int n = FLAGS_n;
  const double low = -3.;
  const double high = 13.;
  const double meas_noise = 0.1;

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

  Noise noise(meas_noise);
  SquaredExp squared_exponential(3.5, 5.7);
  auto cov = noise + squared_exponential;

  std::cout << cov.pretty_string() << std::endl;

  LeaveOneOutGrouper loo;
  UniformlySpacedInducingPoints strategy(FLAGS_k);
  auto model = sparse_gp_from_covariance(cov, loo, strategy, "example");
  //  auto model = gp_from_covariance(cov, "example");

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
