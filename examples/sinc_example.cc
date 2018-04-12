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

DEFINE_string(input, "", "path to csv containing input data.");
DEFINE_string(output, "", "path where predictions will be written in csv.");
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

  using Noise = IndependentNoise<double>;
  using SqrExp = SquaredExponential<ScalarDistance>;

  CovarianceFunction<Constant> mean = {Constant(100.)};
  CovarianceFunction<SlopeTerm> slope = {SlopeTerm(100.)};
  CovarianceFunction<Noise> noise = {Noise(meas_noise)};
  CovarianceFunction<SqrExp> sqrexp = {SqrExp(2., 5.)};

  auto linear_model = mean + slope + noise + sqrexp;

  auto model = gp_from_covariance<double>(linear_model);

  std::cout << "Using Model:" << std::endl;
  std::cout << model.pretty_string() << std::endl;

  model.fit(data);

  std::ostringstream oss;
  {
    cereal::JSONOutputArchive archive(oss);
    archive(cereal::make_nvp(model.get_name(), model));
  }
  std::istringstream iss(oss.str());
  auto untrained_model = gp_from_covariance<double>(linear_model);
  {
    cereal::JSONInputArchive archive(iss);
    archive(cereal::make_nvp(model.get_name(), untrained_model));
  }

  write_predictions_to_csv(FLAGS_output, untrained_model, low, high);
}
