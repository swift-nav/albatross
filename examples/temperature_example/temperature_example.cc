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

#include "temperature_example_utils.h"

#include "Tune"

DEFINE_string(input, "", "path to csv containing input data.");
DEFINE_string(predict, "", "path to csv containing prediction locations.");
DEFINE_string(output, "", "path where predictions will be written in csv.");
DEFINE_string(thin, "1", "path where predictions will be written in csv.");

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  using namespace albatross;

  std::cout << "Reading the input data." << std::endl;
  auto data = read_temperature_csv_input(FLAGS_input, std::stoi(FLAGS_thin));
  std::cout << "Using " << data.features.size() << " data points" << std::endl;

  std::cout << "Defining the model." << std::endl;
  // Measurement Noise
  IndependentNoise<Station> noise(2.0);

  // A Constant temperature value
  Constant mean(1.5);

  // Scale the constant temperature value in a way that defaults
  // to colder values for higher elevations.
  using ElevationScalar = ScalingTerm<ElevationScalingFunction>;
  ElevationScalar elevation_scalar;
  auto elevation_scaled_mean = elevation_scalar * mean;

  // Radial distance is the difference in lengths of the X, Y, Z
  // locations, which translates into a difference in height so
  // this term means "station at different elevations will be less correlated"
  using RadialSqrExp = SquaredExponential<StationDistance<RadialDistance>>;
  RadialSqrExp radial_sqr_exp(15000., 2.5);

  // The angular distance is equivalent to the great circle distance
  using AngularExp = Exponential<StationDistance<AngularDistance>>;
  AngularExp angular_exp(9e-2, 3.5);

  // We multiply the angular and elevation covariance terms.  To justify this
  // think of the extremes.  If two stations are really far apart, regardless
  // of their elevation they should be decorrelated.  Similarly if two stations
  // are close but are at extremely different elevations they should be
  // decorrelated.
  auto spatial_cov = angular_exp * radial_sqr_exp;

  auto covariance = elevation_scaled_mean + noise + spatial_cov;
  auto model = gp_from_covariance(covariance);

  model.set_param("sigma_exponential", {1., std::make_shared<FixedPrior>()});

  //   These parameters are that came from tuning the model to the leave
  //   one out negative log likelihood. which can be done like this:
  //
  //      albatross::LeaveOneOutLikelihood<MarginalDistribution> loo_nll;
  //      auto tuner = get_tuner(model, loo_nll, data);
  //      tuner.initialize_optimizer(nlopt::LN_NELDERMEAD);
  //      const auto params = tuner.tune();
  //      model.set_params(params);
  //
  model.set_param_values({
      {"elevation_scaling_center", 4446.5},
      {"elevation_scaling_factor", 0.000153439},
      {"exponential_length_scale", 1.10298},
      {"sigma_constant", 5.07288},
      {"sigma_exponential", 1},
      {"sigma_independent_noise", 1.75027},
      {"sigma_squared_exponential", 13.913},
      {"squared_exponential_length_scale", 5835.56},
  });

  std::cout << "Training the model." << std::endl;
  const auto fit_model = model.fit(data);

  auto predict_features = read_temperature_csv_input(FLAGS_predict, 1).features;
  std::cout << "Going to predict at " << predict_features.size() << " locations"
            << std::endl;
  write_predictions(FLAGS_output, predict_features, fit_model);
}
