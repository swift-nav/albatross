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

#include <albatross/Core>
#include <albatross/Samplers>

DEFINE_string(input, "", "path to csv containing input data.");
DEFINE_string(output, "", "path where samples will be written in csv.");
DEFINE_int32(n, 10, "number of training points to use.");
DEFINE_int32(maxiter, 1000, "number of iterations.");
DEFINE_string(mode, "radial", "which modelling approach to use.");

using albatross::ParameterStore;
using albatross::RegressionDataset;

namespace albatross {

template <typename ModelType, typename FeatureType>
void run_sampler(const ModelType &model_,
                 const RegressionDataset<FeatureType> &data) {
  ModelType model(model_);

  albatross::GaussianProcessNegativeLogLikelihood nll;
  auto tuner = get_tuner(model, nll, data);
  tuner.optimizer.set_ftol_abs(1e-3);
  tuner.optimizer.set_ftol_rel(1e-3);
  tuner.optimizer.set_maxeval(200);
  model.set_params(tuner.tune());

  std::default_random_engine gen(2012);

  std::size_t max_iterations = static_cast<std::size_t>(FLAGS_maxiter);
  std::shared_ptr<std::ostream> ostream =
      std::make_shared<std::ofstream>(FLAGS_output);
  auto callback = get_csv_writing_callback(model, ostream);

  const std::size_t walkers = 3 * model.get_params().size();

  ensemble_sampler(model, data, walkers, max_iterations, gen, callback);
}

}  // namespace albatross

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  Eigen::Index n = static_cast<Eigen::Index>(FLAGS_n);
  const double low = -10.;
  const double high = 23.;
  const double meas_noise_sd = 0.1;

  if (FLAGS_input == "") {
    FLAGS_input = "input.csv";
  }
  maybe_create_training_data(FLAGS_input, n, low, high, meas_noise_sd);

  using namespace albatross;

  std::cout << "Reading the input data." << std::endl;
  RegressionDataset<double> data = read_csv_input(FLAGS_input);

  std::cout << "Defining the model." << std::endl;

  using Noise = IndependentNoise<double>;
  Noise indep_noise(meas_noise_sd);
  indep_noise.sigma_independent_noise.prior =
      PositivePrior();  // LogScaleUniformPrior(1e-3, 1e2);

  if (FLAGS_mode == "radial") {
    // this approach uses a squared exponential radial function to capture
    // the function we're estimating using non-parametric techniques
    //    const Polynomial<1> polynomial(100.);
    using SquaredExp = SquaredExponential<EuclideanDistance>;
    const SquaredExp squared_exponential(3.5, 5.7);
    auto cov = squared_exponential + measurement_only(indep_noise);
    const LinearMean linear;
    auto model = gp_from_covariance_and_mean(cov, linear);

    run_sampler(model, data);
  } else if (FLAGS_mode == "parametric") {
    // Here we assume we know the "truth" is made up of a linear trend and
    // a scaled and translated sinc function with added noise and capture
    // this all through the use of a mean function.
    const LinearMean linear;
    const SincFunction sinc;
    auto model = gp_from_covariance_and_mean(indep_noise, linear + sinc);

    run_sampler(model, data);
  } else if (FLAGS_mode == "test_gp") {
    /*
     * Create random noisy observations to use as train data.
     */
    std::default_random_engine generator;
    std::normal_distribution<double> noise_distribution(0., meas_noise_sd);
    std::vector<double> xs(albatross::cast::to_size(n));
    Eigen::VectorXd ys(n);
    for (Eigen::Index i = 0; i < n; i++) {
      xs[albatross::cast::to_size(i)] = static_cast<double>(i);
      double noise = noise_distribution(generator);
      ys[i] = noise;
    }
    const albatross::RegressionDataset<double> test_data(xs, ys);
    const auto cov = indep_noise;
    auto model = gp_from_covariance(cov);
    run_sampler(model, test_data);
  }
}
