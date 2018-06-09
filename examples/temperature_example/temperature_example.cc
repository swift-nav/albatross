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
DEFINE_string(output, "", "path where predictions will be written in csv.");


int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  using namespace albatross;

  std::cout << "Reading the input data." << std::endl;
  const auto data = read_temperature_csv_input(FLAGS_input);
  std::cout << "Using " << data.features.size() << " data points" << std::endl;

  std::cout << "Defining the model." << std::endl;
  using Noise = IndependentNoise<Station>;
  using SqrExp = SquaredExponential<StationDistance<EuclideanDistance>>;

  CovarianceFunction<Constant> mean = {Constant(5000.)};
  CovarianceFunction<Noise> noise = {Noise(2.)};
  CovarianceFunction<SqrExp> sqrexp = {SqrExp(350000, 13.)};
  auto covariance = mean + noise + sqrexp;

  std::cout << "Training the model." << std::endl;
  auto model = gp_from_covariance<Station>(covariance);
  auto trimmed_data = albatross::trim_outliers(data, &model);
  model.fit(trimmed_data);


  auto predict_features = build_prediction_grid();
  std::cout << "Going to predict at " << predict_features.size() << " locations" << std::endl;
  write_predictions("./predictions.csv", predict_features, model);

//  RegressionModelCreator<Station> model_creator = [covariance]() {
//    return gp_pointer_from_covariance<Station>(covariance);
//  };
//
//  albatross::TuningMetric<Station> metric =
//      albatross::gp_fast_loo_nll<Station>;
//
//  std::vector<RegressionDataset<Station>> datasets = {trimmed_data};
//  albatross::TuneModelConfg<Station> tune_config(
//      model_creator, datasets, metric);
//
//  auto tuned_params =
//      albatross::tune_regression_model<Station>(tune_config);



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

  /*
   * Make predictions at a bunch of locations which we can then
   * visualize if desired.
   */
//  write_predictions_to_csv(FLAGS_output, model);
}
