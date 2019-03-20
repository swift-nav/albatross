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

#include "example_utils.h"

#include "GP"

DEFINE_string(input, "", "path to csv containing input data.");
DEFINE_string(n, "10", "number of training points to use.");

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int n = std::stoi(FLAGS_n);
  const double low = -3.;
  const double high = 13.;
  const double meas_noise = 1.;

  maybe_create_training_data(FLAGS_input, n, low, high, meas_noise);

  auto data = read_csv_input(FLAGS_input);

  using namespace albatross;

  std::cout << "Defining the model." << std::endl;
  using Noise = IndependentNoise<double>;
  using SquaredExp = SquaredExponential<EuclideanDistance>;

  Constant constant(100.);
  Noise noise(meas_noise);
  SquaredExp squared_exponential(3.5, 5.7);
  auto cov = constant + noise + squared_exponential;

  auto model = gp_from_covariance<double>(cov);

  std::cout << "Using Model:" << std::endl;
  std::cout << model.pretty_string() << std::endl;

  model.fit(data);

  const auto constant_state =
      constant.get_state_space_representation(data.features);

  auto posterior_state = model.inspect(constant_state);
  std::cout << "The posterior estimate of the constant term is: ";
  std::cout << posterior_state.mean << " +/- " << posterior_state.covariance
            << std::endl;
}
