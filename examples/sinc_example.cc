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

#include "evaluate.h"
#include "example_utils.h"
#include "gflags/gflags.h"
#include "tune.h"
#include <functional>

DEFINE_string(input, "", "path to csv containing input data.");
DEFINE_string(output, "", "path where predictions will be written in csv.");
DEFINE_string(n, "10", "number of training points to use.");

double loo_nll(const albatross::RegressionDataset<double> &dataset,
               albatross::RegressionModel<double> *model) {
  auto loo_folds = albatross::leave_one_out(dataset);
  return albatross::cross_validated_scores(
             albatross::evaluation_metrics::negative_log_likelihood, loo_folds,
             model)
      .mean();
}

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  int n = std::stoi(FLAGS_n);
  const double low = -3.;
  const double high = 13.;
  const double meas_noise = 1.;

  maybe_create_training_data(FLAGS_input, n, low, high, meas_noise);

  using namespace albatross;

  std::cout << "Reading the input data." << std::endl;
  RegressionDataset<double> data = read_csv_input(FLAGS_input);

  std::cout << "Defining the model." << std::endl;
  using Noise = IndependentNoise<double>;
  using SqrExp = SquaredExponential<EuclideanDistance>;

  CovarianceFunction<Constant> mean = {Constant(100.)};
  CovarianceFunction<SlopeTerm> slope = {SlopeTerm(100.)};
  CovarianceFunction<Noise> noise = {Noise(10.)};
  CovarianceFunction<SqrExp> sqrexp = {SqrExp(1.5, 100.)};
  auto linear_model = mean + slope + noise + sqrexp;

  /*
   * A side effect of having statically composable covariance
   * functions is that we don't explicitly know the type of the
   * resulting Gaussian process, so to instantiate the model
   * we need to ride off of template inferrence using a helper
   * function.
   */
  std::cout << "Instantiating the model." << std::endl;
  auto model = gp_from_covariance<double>(linear_model);
  model.fit(data);

  /*
   * This step could be skipped but both tests and illustrates how a
   * Gaussian process can be serialized, then deserialized.
   */
  std::ostringstream oss;
  std::cout << "Serializing the model." << std::endl;
  {
    cereal::JSONOutputArchive archive(oss);
    archive(cereal::make_nvp(model.get_name(), model));
  }
  std::istringstream iss(oss.str());
  auto deserialized = gp_from_covariance<double>(linear_model);
  std::cout << "Deserializing the model." << std::endl;
  {
    cereal::JSONInputArchive archive(iss);
    archive(cereal::make_nvp(model.get_name(), deserialized));
  }

  /*
   * Make predictions at a bunch of locations which we can then
   * visualize if desired.
   */
  write_predictions_to_csv(FLAGS_output, deserialized, low, high);
}
