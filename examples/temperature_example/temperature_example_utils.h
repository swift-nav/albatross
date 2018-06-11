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
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

#include "core/model.h"
#include "covariance_functions/covariance_functions.h"
#include "models/gp.h"

namespace albatross {

template <typename FeatureType, typename SubFeatureType=FeatureType>
RegressionDataset<FeatureType> trim_outliers(const RegressionDataset<FeatureType> &dataset,
                                             SerializableGaussianProcess<FeatureType, SubFeatureType> *model) {
  auto loo_predictions = fast_gp_loo_cross_validated_predict(dataset, model);

  Eigen::VectorXd errors = loo_predictions.mean - dataset.targets.mean;
  Eigen::VectorXd error_stddev = errors.array().cwiseQuotient(loo_predictions.covariance.diagonal().array().sqrt()).matrix();

  double mean = error_stddev.array().mean();
  double stddev = std::sqrt((error_stddev.array() - mean).pow(2.).mean());

  std::vector<Eigen::Index> good_indices;
  for (Eigen::Index i=0; i < errors.size(); i++) {
    if (fabs(error_stddev[i]) <= 3.) {
      good_indices.push_back(i);
    }
  }

  std::vector<FeatureType> feature_subset = subset(good_indices, dataset.features);
  TargetDistribution target_subset = subset(good_indices, dataset.targets);
  RegressionDataset<FeatureType> subset_data(feature_subset, target_subset);

  std::cout << "Removing " << errors.size() - good_indices.size() << " data entries" << std::endl;

  return subset_data;
}
}

struct Station {
  int id;
  double lat;
  double lon;
  double height;
  Eigen::Vector3d ecef;

  bool operator==(const Station &rhs) const {
    return (ecef == rhs.ecef);
  }

  template <typename Archive>
  void serialize(Archive &archive) {
    archive(id, lat, lon, height, ecef);
  }

};

/*
 * Provides an interface which maps the PiercePoint.ecef
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

  double operator()(const Station& x, const Station& y) const {
    return DistanceMetricType::operator()(x.ecef, y.ecef);
  };
};


class ElevationScalingFunction : public albatross::ScalingFunction {
 public:

  ElevationScalingFunction(double center = 1000., double factor = 3.5 / 300) {
    this->params_["elevation_scaling_center"] = center;
    this->params_["elevation_scaling_factor"] = factor;
  };


  std::string get_name() const {
    return "elevation_scaled";
  }

  double operator() (const Station &x) const {
    // This is the negative orientation rectifier function which
    // allows lower elevations to have a higher variance.
    double center = this->params_.at("elevation_scaling_center");
    return 1. + this->params_.at("elevation_scaling_factor") * fmax(0., (center - x.height));
  }

};


albatross::RegressionDataset<Station> read_temperature_csv_input(std::string file_path,
                                                                 int thin = 5) {
  std::vector<Station> features;
  std::vector<double> targets;

  io::CSVReader<8> file_in(file_path);

  file_in.read_header(io::ignore_extra_column, "STATION", "LAT", "LON", "ELEV(M)", "X", "Y", "Z", "TEMP");

  bool more_to_parse = true;
  int count = 0;
  while (more_to_parse) {
    double temperature;
    Station station;
    more_to_parse = file_in.read_row(station.id,
                                     station.lat, station.lon, station.height,
                                     station.ecef[0],
                                     station.ecef[1],
                                     station.ecef[2],
                                     temperature);
    if (more_to_parse && count % thin == 0) {
      features.push_back(station);
      targets.push_back(temperature);
    }
    count++;
  }
  Eigen::Map<Eigen::VectorXd> eigen_targets(&targets[0], static_cast<int>(targets.size()));
  return albatross::RegressionDataset<Station>(features, eigen_targets);
}

inline bool file_exists(const std::string &name) {
  std::ifstream f(name.c_str());
  return f.good();
}


/** Semi-major axis of the Earth, \f$ a \f$, in meters.
 * This is a defining parameter of the WGS84 ellipsoid. */
#define WGS84_A 6378137.0
/** Inverse flattening of the Earth, \f$ 1/f \f$.
 * This is a defining parameter of the WGS84 ellipsoid. */
#define WGS84_IF 298.257223563
/** The flattening of the Earth, \f$ f \f$. */
#define WGS84_F (1 / WGS84_IF)
/** Semi-minor axis of the Earth in meters, \f$ b = a(1-f) \f$. */
#define WGS84_B (WGS84_A * (1 - WGS84_F))
/** Eccentricity of the Earth, \f$ e \f$ where \f$ e^2 = 2f - f^2 \f$ */
#define WGS84_E (sqrt(2 * WGS84_F - WGS84_F * WGS84_F))

Eigen::Vector3d lat_lon_to_ecef(const double lat,
                                const double lon,
                                const double height) {
  double d = WGS84_E * sin(lat);
  double N = WGS84_A / sqrt(1. - d * d);

  Eigen::Vector3d ecef;
  ecef[0] = (N + height) * cos(lat) * cos(lon);
  ecef[1] = (N + height) * cos(lat) * sin(lon);
  ecef[2] = ((1 - WGS84_E * WGS84_E) * N + height) * sin(lat);
  return ecef;
}

std::vector<Station> build_prediction_grid() {
  double lon_low = -125.;
  double lon_high = -60;
  double lat_low = 25.;
  double lat_high = 50.;

//    double lon_low = -100.;
//    double lon_high = -80;
//    double lat_low = 25.;
//    double lat_high = 45.;
  double spacing = 0.5;

  int lon_count = ceil((lon_high - lon_low) / spacing);
  int lat_count = ceil((lat_high - lat_low) / spacing);

  double lon_stride = (lon_high - lon_low) / (lon_count);
  double lat_stride = (lat_high - lat_low) / (lat_count);

  std::vector<Station> features;
  for (int lon_i = 0; lon_i < lon_count; lon_i++) {
    for (int lat_i = 0; lat_i < lat_count; lat_i++) {
      Station grid_location;
      grid_location.lat = lat_low + lat_i * lat_stride;
      grid_location.lon = lon_low + lon_i * lon_stride;
      grid_location.ecef = lat_lon_to_ecef(grid_location.lat * M_PI / 180.,
                                           grid_location.lon * M_PI / 180.,
                                           0.);
      grid_location.id = lon_i * 10000 + lat_i;
      features.push_back(grid_location);
    }
  }
  return features;
}

void write_predictions(const std::string output_path,
                       const std::vector<Station> features,
                       const albatross::RegressionModel<Station> &model) {

  std::ofstream ostream;
  ostream.open(output_path);
  ostream << "STATION,LAT,LON,ELEV(M),X,Y,Z,TEMP,VARIANCE" << std::endl;

  std::size_t n = features.size();
  std::size_t count = 0;
  for (const auto &f : features) {
    ostream << std::to_string(f.id);
    ostream << ", " << std::to_string(f.lat);
    ostream << ", " << std::to_string(f.lon);
    ostream << ", " << std::to_string(f.height);
    ostream << ", " << std::to_string(f.ecef[0]);
    ostream << ", " << std::to_string(f.ecef[1]);
    ostream << ", " << std::to_string(f.ecef[2]);

    std::vector<Station> one_feature = {f};
    const auto pred = model.predict(one_feature);

    ostream << ", " << std::to_string(pred.mean[0]);
    ostream << ", " << std::to_string(std::sqrt(pred.covariance(0, 0)));
    ostream << std::endl;
    if (count % 100 == 0) {
      std::cout << count + 1 << "/" << n << std::endl;
    }
    count++;
  }

}

#endif
