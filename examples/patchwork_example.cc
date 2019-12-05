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

#include <albatross/PatchworkGP>

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

struct PatchworkFunctions {

  double width = 2.;

  long int grouper(const double &x) const { return lround(x / width); }

  std::vector<double> boundary(long int x, long int y) const {
    if (fabs(x - y) == 1) {
      double center = width * 0.5 * (x + y);
      std::vector<double> boundary = {center - 1., center, center + 1.};
      return boundary;
    }
    return {};
  }

  long int nearest_group(const std::vector<long int> &groups,
                         const long int &query) const {
    long int nearest = query;
    long int nearest_distance = std::numeric_limits<long int>::max();
    for (const auto &group : groups) {
      const auto distance = abs(query - group);
      if (distance < nearest_distance) {
        nearest = group;
        nearest_distance = distance;
      }
    }
    return nearest;
  }
};

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
  auto cov_func = noise + squared_exponential;

  std::cout << cov_func.pretty_string() << std::endl;

  const auto m = gp_from_covariance(cov_func, "example_model");

  auto fit_to_interval = [&](const auto &dataset) { return m.fit(dataset); };

  PatchworkFunctions patchwork_functions;

  auto grouper = [&](const auto &f) { return patchwork_functions.grouper(f); };

  const auto fit_models = data.group_by(grouper).apply(fit_to_interval);

  const auto patchwork =
      patchwork_gp_from_covariance(cov_func, patchwork_functions);

  const auto patchwork_fit = patchwork.from_fit_models(fit_models);

  const std::size_t k = 161;
  auto grid_xs = uniform_points_on_line(k, low - 2., high + 2.);
  Eigen::VectorXd truth_mean(k);
  for (Eigen::Index i = 0; i < truth_mean.size(); ++i) {
    truth_mean[i] = truth(grid_xs[i]);
  }
  MarginalDistribution targets(truth_mean);
  RegressionDataset<double> dataset(grid_xs, targets);

  auto prediction = patchwork_fit.predict(dataset.features).marginal();

  std::ofstream output;
  output.open(FLAGS_output);

  albatross::write_to_csv(output, dataset, prediction);
}
