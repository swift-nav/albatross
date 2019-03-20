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

#ifndef ALBATROSS_EXAMPLE_UTILS_H
#define ALBATROSS_EXAMPLE_UTILS_H

#include "csv.h"
#include <fstream>
#include <iostream>

#include "Core"
#include "csv_utils.h"

#define EXAMPLE_SLOPE_VALUE sqrt(2.)
#define EXAMPLE_CONSTANT_VALUE 3.14159

/*
 * Randomly samples n points between low and high.
 */
std::vector<double> random_points_on_line(const int n, const double low,
                                          const double high) {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(low, high);

  std::vector<double> xs;
  for (int i = 0; i < n; i++) {
    xs.push_back(distribution(generator));
  }
  return xs;
};

/*
 * Creates a grid of n points between low and high.
 */
std::vector<double> uniform_points_on_line(const std::size_t n,
                                           const double low,
                                           const double high) {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(low, high);

  std::vector<double> xs;
  for (std::size_t i = 0; i < n; i++) {
    double ratio = (double)i / (double)(n - 1);
    xs.push_back(low + ratio * (high - low));
  }
  return xs;
};

/*
 * The noise free function we're attempting to estimate.
 */
double truth(double x) {
  double sinx_x = x == 0 ? 1. : sin(x) / x;
  return x * EXAMPLE_SLOPE_VALUE + EXAMPLE_CONSTANT_VALUE + 10. * sinx_x;
}

/*
 * Create random noisy observations to use as train data.
 */
albatross::RegressionDataset<double>
create_train_data(const int n, const double low, const double high,
                  const double measurement_noise) {
  auto xs = random_points_on_line(n, low, high);

  std::default_random_engine generator;
  std::normal_distribution<double> noise_distribution(0., measurement_noise);

  Eigen::VectorXd ys(n);

  for (int i = 0; i < n; i++) {
    double noise = noise_distribution(generator);
    ys[i] = (truth(xs[i]) + noise);
  }

  return albatross::RegressionDataset<double>(xs, ys);
}

albatross::RegressionDataset<double> read_csv_input(std::string file_path) {
  std::vector<double> xs;
  std::vector<double> ys;

  io::CSVReader<2> file_in(file_path);

  file_in.read_header(io::ignore_extra_column, "x", "y");
  double x, y;
  bool more_to_parse = true;
  while (more_to_parse) {
    more_to_parse = file_in.read_row(x, y);
    if (more_to_parse) {
      xs.push_back(x);
      ys.push_back(y);
    }
  }
  Eigen::Map<Eigen::VectorXd> eigen_ys(&ys[0], static_cast<int>(ys.size()));
  return albatross::RegressionDataset<double>(xs, eigen_ys);
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
    auto data = create_train_data(n, low, high, meas_noise);
    std::ofstream train;
    train.open(input_path);
    train << "x,y" << std::endl;
    for (int i = 0; i < static_cast<int>(data.features.size()); i++) {
      train << data.features[i] << ", " << data.targets.mean[i] << std::endl;
    }
  }
}

void write_predictions_to_csv(const std::string output_path,
                              const albatross::RegressionModel<double> *model,
                              const double low, const double high) {
  std::ofstream output;
  output.open(output_path);

  const std::size_t k = 161;
  auto grid_xs = uniform_points_on_line(k, low - 2., high + 2.);

  auto predictions = model->predict(grid_xs);

  Eigen::VectorXd targets(static_cast<Eigen::Index>(k));
  for (std::size_t i = 0; i < k; i++) {
    targets[static_cast<Eigen::Index>(i)] = truth(grid_xs[i]);
  }

  const albatross::RegressionDataset<double> dataset(grid_xs, targets);

  albatross::write_to_csv(output, dataset, predictions);

  output.close();
}

#endif
