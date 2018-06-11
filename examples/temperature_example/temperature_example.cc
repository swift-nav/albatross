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
#include "gflags/gflags.h"
#include "tune.h"
#include <functional>
#include "temperature_example_utils.h"

DEFINE_string(input, "", "path to csv containing input data.");
DEFINE_string(predict, "", "path to csv containing prediction locations.");
DEFINE_string(output, "", "path where predictions will be written in csv.");
DEFINE_string(thin, "", "path where predictions will be written in csv.");


int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  using namespace albatross;

  std::cout << "Reading the input data." << std::endl;
  auto data = read_temperature_csv_input(FLAGS_input, 1);
  std::cout << "Using " << data.features.size() << " data points" << std::endl;

  std::cout << "Defining the model." << std::endl;
  using Noise = IndependentNoise<Station>;
  CovarianceFunction<Constant> mean = {Constant(1.5)};
  CovarianceFunction<Noise> noise = {Noise(2.0)};

//  using EuclideanSqrExp = SquaredExponential<StationDistance<EuclideanDistance>>;
//  CovarianceFunction<EuclideanSqrExp> euclidean_sqrexp = {EuclideanSqrExp(200000., 8.)};
//  auto spatial_cov = euclidean_sqrexp;

  using RadialExp = Exponential<StationDistance<RadialDistance>>;
  CovarianceFunction<RadialExp> radial_exp = {RadialExp(15000., 2.5)};

  using AngularSqrExp = SquaredExponential<StationDistance<AngularDistance>>;
  CovarianceFunction<AngularSqrExp> angular_sqrexp = {AngularSqrExp(9e-2, 3.5)};
  auto spatial_cov = angular_sqrexp * radial_exp;

  using ElevationScalar = ScalingTerm<ElevationScalingFunction>;
  CovarianceFunction<ElevationScalar> elevation_scalar = {ElevationScalar()};
  auto elevation_scaled_mean = elevation_scalar * mean;

  auto covariance = elevation_scaled_mean + noise + spatial_cov;
  auto model = gp_from_covariance<Station>(covariance);

  ParameterStore params = {
      {"elevation_scaling_center", 3965.98},
      {"elevation_scaling_factor", 0.000810492},
      {"exponential_length_scale", 28197.6},
      {"length_scale", 0.0753042},
      {"sigma_constant", 1.66872},
      {"sigma_exponential", 2.07548},
      {"sigma_independent_noise", 1.8288},
      {"sigma_squared_exponential", 3.77329},
  };

  model.set_params(params);

//  data = albatross::trim_outliers(data, &model);

  /*
   * TUNING
   */
//  RegressionModelCreator<Station> model_creator = [covariance, params]() {
//    auto model_ptr = gp_pointer_from_covariance<Station>(covariance);
//    model_ptr->set_params(params);
//    return model_ptr;
//  };
//
//  albatross::TuningMetric<Station> metric =
//      albatross::gp_fast_loo_nll<Station>;
//
//  std::vector<RegressionDataset<Station>> datasets = {data};
//  albatross::TuneModelConfg<Station> tune_config(
//      model_creator, datasets, metric);
//
//  auto tuned_params =
//      albatross::tune_regression_model<Station>(tune_config);
//  model.set_params(tuned_params);

  std::cout << "Training the model." << std::endl;
  model.fit(data);

  const auto nll = gp_fast_loo_nll(data, &model);
  std::cout << nll << std::endl;

  auto predict_features = read_temperature_csv_input(FLAGS_predict, 1).features;
  std::cout << "Going to predict at " << predict_features.size() << " locations" << std::endl;
  write_predictions(FLAGS_output, predict_features, model);

//
//  /*
//   * This step could be skipped but both tests and illustrates how a
//   * Gaussian process can be serialized, then deserialized.
//   */
//  std::ostringstream oss;
//  std::cout << "Serializing the model." << std::endl;
//  {
//    cereal::JSONOutputArchive archive(oss);
//    archive(cereal::make_nvp(model.get_name(), model));
//  }
//  std::cout << "Serialized model is " << oss.str().size() << " bytes" << std::endl;
//  std::istringstream iss(oss.str());
//  auto deserialized = gp_from_covariance<Station>(covariance);
//  std::cout << "Deserializing the model." << std::endl;
//  {
//    cereal::JSONInputArchive archive(iss);
//    archive(cereal::make_nvp(model.get_name(), deserialized));
//  }

}
