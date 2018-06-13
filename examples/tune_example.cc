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

#include "tune.h"
#include "example_utils.h"
#include "gflags/gflags.h"
#include <functional>

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

  /*
   * Tuning works by iteratively creating new models and assigning
   * them different parameters.  In order to do so it needs a function
   * that will generate a pointer to a new model which we define here
   * using lambdas.
   */
  RegressionModelCreator<double> model_creator = [linear_model]() {
    return gp_pointer_from_covariance<double>(linear_model);
  };

  /*
   * Now we tune the model by finding the hyper parameters that
   * maximize the likelihood (or minimize the negative log likelihood).
   */
  std::cout << "Tuning the model." << std::endl;
  TuningMetric<double> metric = albatross::gp_fast_loo_nll<double>;

  TuneModelConfg<double> config(model_creator, data, metric);
  auto params = tune_regression_model<double>(config);
}
