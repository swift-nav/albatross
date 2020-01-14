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

#ifndef ALBATROSS_TEMPERATURE_EXAMPLE_UTILS_H
#define ALBATROSS_TEMPERATURE_EXAMPLE_UTILS_H

#include <albatross/GP>
#include <albatross/NullModel>
#include <albatross/src/utils/csv_utils.hpp>
#include <csv.h>
#include <fstream>

DEFINE_string(input, "", "path to csv containing input data.");
DEFINE_string(predict, "", "path to csv containing prediction locations.");
DEFINE_string(output, "", "path where predictions will be written in csv.");
DEFINE_int32(thin, 1,
             "larger numbers let you randomly reduce the size of the data to "
             "avoid long compute times.");

namespace albatross {

/*
 * Holds the information about the time and location of a single
 * temperature observation.
 */
struct StationTime {
  int station_id;
  int day_of_year;
  Eigen::Vector3d ecef;

  bool operator==(const StationTime &rhs) const {
    return (station_id == rhs.station_id && day_of_year == rhs.day_of_year &&
            ecef == rhs.ecef);
  }

  template <typename Archive> void serialize(Archive &archive) {

    archive(cereal::make_nvp("station_id", station_id),
            cereal::make_nvp("day_of_year", day_of_year),
            cereal::make_nvp("ecef_x", ecef[0]),
            cereal::make_nvp("ecef_y", ecef[1]),
            cereal::make_nvp("ecef_z", ecef[2]));
  }
};

albatross::RegressionDataset<StationTime>
read_temperature_csv_input(const std::string &file_path, int thin = 5) {
  std::vector<StationTime> features;
  std::vector<double> targets;

  io::CSVReader<9> file_in(file_path);

  file_in.read_header(io::ignore_extra_column, "STATION", "LAT", "LON",
                      "ELEV(M)", "X", "Y", "Z", "TEMP", "DAY_OF_YEAR");

  bool more_to_parse = true;
  int count = 0;
  while (more_to_parse) {
    double temperature;
    double lat, lon, elev;
    StationTime station;
    more_to_parse = file_in.read_row(
        station.station_id, lat, lon, elev, station.ecef[0], station.ecef[1],
        station.ecef[2], temperature, station.day_of_year);
    if (more_to_parse && count % thin == 0) {
      features.push_back(station);
      targets.push_back(temperature);
    }
    count++;
  }
  Eigen::Map<Eigen::VectorXd> eigen_targets(&targets[0],
                                            static_cast<int>(targets.size()));
  return albatross::RegressionDataset<StationTime>(features, eigen_targets);
}

inline bool file_exists(const std::string &name) {
  std::ifstream f(name.c_str());
  return f.good();
}

template <typename ModelType, typename FitType>
void write_predictions(const std::string &output_path,
                       const std::vector<StationTime> &features,
                       const FitModel<ModelType, FitType> &fit_model) {

  std::ofstream ostream;
  ostream.open(output_path);

  Eigen::VectorXd targets =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(features.size()));

  albatross::RegressionDataset<StationTime> dataset(features, targets);

  const auto predictions = fit_model.predict(features).marginal();
  albatross::write_to_csv(ostream, dataset, predictions);
}

void write_location_predictions(const std::string &output_path,
                                const std::vector<Eigen::Vector3d> &locations) {

  std::ofstream ostream;
  ostream.open(output_path);

  ostream << "X,Y,Z" << std::endl;
  for (const auto &loc : locations) {
    ostream << loc[0] << "," << loc[1] << "," << loc[2] << std::endl;
  }
}

} // namespace albatross
#endif
