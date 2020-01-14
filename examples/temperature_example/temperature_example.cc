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

#include <albatross/Tune>
#include <gflags/gflags.h>

#include "temperature_example_utils.h"

namespace albatross {

void predict_temperature() {

  std::cout << "Reading the input data." << std::endl;
  auto training_dataset = read_temperature_csv_input(FLAGS_input, FLAGS_thin);
  std::cout << "Using " << training_dataset.features.size()
            << " training data points" << std::endl;

  NullModel model;

  std::cout << "Fitting the model." << std::endl;
  const auto fit_model = model.fit(training_dataset);

  auto predict_features = read_temperature_csv_input(FLAGS_predict, 1).features;
  std::cout << "Making predictions at " << predict_features.size()
            << " locations" << std::endl;
  write_predictions(FLAGS_output, predict_features, fit_model);
}

} // namespace albatross

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  albatross::predict_temperature();
}
