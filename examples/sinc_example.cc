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
DEFINE_bool(tune, false, "a flag indication parameters should be tuned first.");

using albatross::ParameterStore;
using albatross::RegressionDataset;
using albatross::RegressionModelCreator;
using albatross::TuningMetric;
using albatross::TuneModelConfig;
using albatross::tune_regression_model;

albatross::ParameterStore
tune_model(RegressionModelCreator<double> &model_creator,
           RegressionDataset<double> &data) {
  /*
   * Now we tune the model by finding the hyper parameters that
   * maximize the likelihood (or minimize the negative log likelihood).
   */
  std::cout << "Tuning the model." << std::endl;

  TuningMetric<double> metric = albatross::gp_fast_loo_nll<double>;

  TuneModelConfig<double> config(model_creator, data, metric);
  return tune_regression_model<double>(config);
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
  using SquaredExp = SquaredExponential<EuclideanDistance>;
  using PolynomialTerm = Polynomial<1>;

  CovarianceFunction<Polynomial<1>> polynomial = {Polynomial<1>(100.)};
  CovarianceFunction<Noise> noise = {Noise(meas_noise)};
  CovarianceFunction<SquaredExp> squared_exponential = {SquaredExp(3.5, 5.7)};
  auto cov = polynomial + noise + squared_exponential;

  std::cout << cov.pretty_string() << std::endl;

  RegressionModelCreator<double> model_creator = [&]() {
    /*
     * A side effect of having statically composable covariance
     * functions is that we don't explicitly know the type of the
     * resulting Gaussian process, so to instantiate the model
     * we need to ride off of template inferrence using a helper
     * function.
     */
    auto model = gp_pointer_from_covariance<double>(cov);
    return model;
  };

  auto model = model_creator();

  if (FLAGS_tune) {
    model->set_params(tune_model(model_creator, data));
  }

  std::cout << pretty_param_details(model->get_params()) << std::endl;
  model->fit(data);

  /*
   * Make predictions at a bunch of locations which we can then
   * visualize if desired.
   */
  write_predictions_to_csv(FLAGS_output, model.get(), low, high);
}
