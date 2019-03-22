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

#include "csv.h"
#include <fstream>

#include "GP"
#include "csv_utils.h"

namespace albatross {

/*
 * Holds the information about a single station which
 * is used as the FeatureType for our Gaussian process.
 */
struct Station {
  int id;
  double lat;
  double lon;
  double height;
  Eigen::Vector3d ecef;

  bool operator==(const Station &rhs) const { return (ecef == rhs.ecef); }

  template <typename Archive> void serialize(Archive &archive) {
    archive(cereal::make_nvp("id", id), cereal::make_nvp("lat", lat),
            cereal::make_nvp("lon", lon), cereal::make_nvp("height", height),
            cereal::make_nvp("ecef_x", ecef[0]),
            cereal::make_nvp("ecef_y", ecef[1]),
            cereal::make_nvp("ecef_z", ecef[2]));
  }
};

/*
 * Provides an interface which maps the Station.ecef
 * field to an arbitrary DistanceMetric defined on Eigen
 * vectors.
 */
template <typename DistanceMetricType>
class StationDistance : public DistanceMetricType {
public:
  StationDistance(){};

  std::string get_name() const {
    std::ostringstream oss;
    oss << "station_" << DistanceMetricType::get_name();
    return oss.str();
  };

  ~StationDistance(){};

  double operator()(const Station &x, const Station &y) const {
    return DistanceMetricType::operator()(x.ecef, y.ecef);
  };
};

class ElevationScalingFunction : public albatross::ScalingFunction {
public:
  ElevationScalingFunction(double center = 1000., double factor = 3.5 / 300) {
    this->params_["elevation_scaling_center"] = {
        center, std::make_shared<UniformPrior>(0., 5000.)};
    this->params_["elevation_scaling_factor"] = {
        factor, std::make_shared<PositivePrior>()};
  };

  std::string get_name() const { return "elevation_scaled"; }

  double _call_impl(const Station &x) const {
    // This is the negative orientation rectifier function which
    // allows lower elevations to have a higher variance.
    double center = this->get_param_value("elevation_scaling_center");
    double factor = this->get_param_value("elevation_scaling_factor");
    return 1. + factor * fmax(0., (center - x.height));
  }
};

albatross::RegressionDataset<Station>
read_temperature_csv_input(const std::string &file_path, int thin = 5) {
  std::vector<Station> features;
  std::vector<double> targets;

  io::CSVReader<8> file_in(file_path);

  file_in.read_header(io::ignore_extra_column, "STATION", "LAT", "LON",
                      "ELEV(M)", "X", "Y", "Z", "TEMP");

  bool more_to_parse = true;
  int count = 0;
  while (more_to_parse) {
    double temperature;
    Station station;
    more_to_parse = file_in.read_row(
        station.id, station.lat, station.lon, station.height, station.ecef[0],
        station.ecef[1], station.ecef[2], temperature);
    if (more_to_parse && count % thin == 0) {
      features.push_back(station);
      targets.push_back(temperature);
    }
    count++;
  }
  Eigen::Map<Eigen::VectorXd> eigen_targets(&targets[0],
                                            static_cast<int>(targets.size()));
  return albatross::RegressionDataset<Station>(features, eigen_targets);
}

inline bool file_exists(const std::string &name) {
  std::ifstream f(name.c_str());
  return f.good();
}

template <typename ModelType, typename FitType>
void write_predictions(const std::string &output_path,
                       const std::vector<Station> &features,
                       const FitModel<ModelType, FitType> &fit_model) {

  std::ofstream ostream;
  ostream.open(output_path);

  Eigen::VectorXd targets =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(features.size()));

  albatross::RegressionDataset<Station> dataset(features, targets);

  const auto predictions = fit_model.predict(features).marginal();
  albatross::write_to_csv(ostream, dataset, predictions);
}

} // namespace albatross
#endif
