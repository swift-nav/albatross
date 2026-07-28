/*
 * Copyright (C) 2026 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_BENCHMARKS_BENCH_UTILS_H
#define ALBATROSS_BENCHMARKS_BENCH_UTILS_H

#include <albatross/GP>

#include <random>
#include <vector>

namespace albatross {
namespace bench {

// Uniform random features on [0, 10].
inline std::vector<double> random_features(std::size_t n,
                                           std::uint32_t seed = 0) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(0., 10.);
  std::vector<double> features(n);
  for (auto &f : features) {
    f = dist(gen);
  }
  return features;
}

inline Eigen::VectorXd random_vector(Eigen::Index n, std::uint32_t seed = 1) {
  std::mt19937 gen(seed);
  std::normal_distribution<double> dist(0., 1.);
  Eigen::VectorXd out(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    out[i] = dist(gen);
  }
  return out;
}

inline Eigen::MatrixXd random_matrix(Eigen::Index rows, Eigen::Index cols,
                                     std::uint32_t seed = 2) {
  std::mt19937 gen(seed);
  std::normal_distribution<double> dist(0., 1.);
  Eigen::MatrixXd out(rows, cols);
  for (Eigen::Index j = 0; j < cols; ++j) {
    for (Eigen::Index i = 0; i < rows; ++i) {
      out(i, j) = dist(gen);
    }
  }
  return out;
}

// A representative covariance function: squared exponential plus
// independent measurement noise.
inline auto bench_covariance() {
  SquaredExponential<EuclideanDistance> squared_exponential(1., 1.);
  IndependentNoise<double> noise(0.1);
  return squared_exponential + noise;
}

// A well conditioned positive definite matrix built from an actual
// covariance function so the spectra resemble the real workloads.
inline Eigen::MatrixXd random_psd_matrix(Eigen::Index n,
                                         std::uint32_t seed = 3) {
  const auto features = random_features(albatross::cast::to_size(n), seed);
  const auto cov = bench_covariance();
  return cov(features);
}

inline RegressionDataset<double> random_dataset(std::size_t n,
                                                std::uint32_t seed = 4) {
  const auto features = random_features(n, seed);
  Eigen::VectorXd targets(albatross::cast::to_index(n));
  for (std::size_t i = 0; i < n; ++i) {
    targets[albatross::cast::to_index(i)] =
        std::sin(features[i]) + 0.1 * std::cos(10. * features[i]);
  }
  return RegressionDataset<double>(features, targets);
}

} // namespace bench
} // namespace albatross

#endif // ALBATROSS_BENCHMARKS_BENCH_UTILS_H
