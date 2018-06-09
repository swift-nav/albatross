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

#ifndef ALBATROSS_SPATIAL_EXAMPLE_UTILS_H
#define ALBATROSS_SPATIAL_EXAMPLE_UTILS_H

#include "csv.h"
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

#include "core/model.h"
#include "covariance_functions/covariance_functions.h"
#include "models/gp.h"

/*
 * Creates a grid of n points between low and high.
 */
std::vector<Eigen::VectorXd> uniform_points_in_2d(const int n, const double low,
                                                  const double high) {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(low, high);

  std::vector<Eigen::VectorXd> features;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double x_ratio = (double)i / (double)(n - 1);
      double y_ratio = (double)i / (double)(n - 1);

      double x = low + x_ratio * (high - low);
      double y = low + y_ratio * (high - low);
      Eigen::VectorXd f;
      f << x, y;
      features.push_back(f);
    }
  }
  return features;
};

/*
 * The noise free function we're attempting to estimate.
 */
double truth(const Eigen::VectorXd &x) {
  return sin(x.norm()) / x.norm();
}


albatross::RegressionDataset<Eigen::VectorXd> create_circular_data(const int n,
                                                                   const double meas_noise) {
  std::random_device rd{};
  std::mt19937 gen{rd()};
  gen.seed(3);

  std::uniform_readl_distribution<double> rand_angle(0., 2 * M_PI);
  std::uniform_readl_distribution<double> rand_radius(0., 1.);

  std::normal_distribution<> d{0., meas_noise};

  std::vector<Eigen::VectorXd> features;
  Eigen::VectorXd targets(n);

  for (int i = 0; i < n; i++) {
    double theta = rand_angle(gen);
    double radius = rand_radius(gen);
    Eigen::VectorXd x(2);
    x << cos(theta) * radius, sin(theta) * radius;
    features.push_back(x);
    targets[i] = truth(x) + d(gen);
  }

  return albatross::RegressionDataset<Eigen::VectorXd>(features, targets);
}


albatross::RegressionDataset<Eigen::VectorXd> read_csv_input(std::string file_path) {
  std::vector<Eigen::VectorXd> features;
  std::vector<double> targets;

  io::CSVReader<2> file_in(file_path);

  file_in.read_header(io::ignore_extra_column, "x", "y", "target");
  double x, y, target;
  bool more_to_parse = true;
  while (more_to_parse) {
    more_to_parse = file_in.read_row(x, y, target);
    if (more_to_parse) {
      Eigen::VectorXd f;
      f << x, y;
      features.push_back(f);
      targets.push_back(target);
    }
  }
  Eigen::Map<Eigen::VectorXd> eigen_targets(&targets[0], static_cast<int>(targets.size()));
  return albatross::RegressionDataset<double>(features, eigen_targets);
}

inline bool file_exists(const std::string &name) {
  std::ifstream f(name.c_str());
  return f.good();
}

void maybe_create_training_data(std::string input_path, const int n,
                                const double low, const double high,
                                const double meas_noise) {
  /*
   * Either read the input data from file, or if it doesn't exist
   * generate new input data and write it to file.
   */
  if (file_exists(input_path)) {
    std::cout << "reading data from : " << input_path << std::endl;
  } else {
    std::cout << "creating training data and writing it to : " << input_path
              << std::endl;
    auto data = create_circular_data(n, meas_noise);
    std::ofstream train;
    train.open(input_path);
    train << "x,y" << std::endl;
    for (int i = 0; i < static_cast<int>(data.features.size()); i++) {
      train << data.features[i] << ", " << data.targets.mean[i] << std::endl;
    }
  }
}

void write_predictions_to_csv(const std::string output_path,
                              const albatross::RegressionModel<Eigen::VectorXd> &model,
                              const double low, const double high) {
  std::ofstream output;
  output.open(output_path);

  const int k = 161;
  auto grid_xs = uniform_points_on_line(k, low - 2., high + 2.);

  auto predictions = model.predict(grid_xs);

  std::cout << "writing predictions to : " << output_path << std::endl;
  output << "x,y,variance,truth" << std::endl;
  for (int i = 0; i < k; i++) {
    output << grid_xs[i];
    output << "," << predictions.mean[i];
    output << "," << predictions.covariance(i, i);
    output << "," << truth(grid_xs[i]);
    output << std::endl;
  }
  output.close();
}

#endif
