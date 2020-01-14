/*
 * Copyright (C) 2020 Swift Navigation Inc.
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

void predict_location() {

  std::cout << "Reading the input data." << std::endl;
  auto training_dataset = read_temperature_csv_input(FLAGS_input, FLAGS_thin);
  std::cout << "Using " << training_dataset.features.size() << " data points"
            << std::endl;

  auto unknown_dataset = read_temperature_csv_input(FLAGS_predict, 1);

  NullModel model;

  // This vector should hold one guess for the x, y, z ECEF coordinates of the
  // unknown stations location.
  std::vector<Eigen::Vector3d> location_estimates;

  auto get_station_id = [](const auto &obs) { return obs.station_id; };

  auto get_location = [](const auto &dataset) {
    assert(dataset.size() > 0);
    return dataset.features[0].ecef;
  };

  const auto station_locations =
      training_dataset.group_by(get_station_id).apply(get_location);
  for (const auto &loc : station_locations.values()) {
    location_estimates.push_back(loc);
  }

  write_location_predictions(FLAGS_output, location_estimates);
}

} // namespace albatross

int main(int argc, char *argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  albatross::predict_location();
}
