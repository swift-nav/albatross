/*
 * Copyright (C) 2026 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 */

#include <albatross/GP>
#include <benchmark/benchmark.h>

#include <random>
#include <vector>

namespace {

albatross::RegressionDataset<double> make_dataset(std::size_t n) {
  std::mt19937 rng(23);
  std::uniform_real_distribution<double> xd(-100.0, 100.0);
  std::normal_distribution<double> noise(0.0, 0.1);
  std::vector<double> xs;
  Eigen::VectorXd ys(n);
  xs.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double x = xd(rng);
    xs.push_back(x);
    ys[static_cast<Eigen::Index>(i)] =
        std::sin(0.05 * x) + 0.5 * std::cos(0.01 * x) + noise(rng);
  }
  return albatross::RegressionDataset<double>(xs, ys);
}

auto make_model() {
  albatross::SquaredExponential<albatross::EuclideanDistance> se;
  albatross::IndependentNoise<double> noise;
  return albatross::gp_from_covariance(se + noise, "bench");
}

}  // namespace

// GP fit: O(N^3). Dominated by Cholesky / kernel-matrix build.
static void BM_GP_Fit(benchmark::State &state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto data = make_dataset(n);
  auto model = make_model();
  for (auto _ : state) {
    auto fit = model.fit(data);
    benchmark::DoNotOptimize(fit);
  }
}
BENCHMARK(BM_GP_Fit)->Arg(256)->Arg(512)->Arg(1024);

// GP marginal predict — exercises Fix #1 (kernel build for cross_cov),
// the cached LDLT solve, and the per-feature diagonal loop in gp.hpp.
static void BM_GP_PredictMarginal(benchmark::State &state) {
  const auto n_train = static_cast<std::size_t>(state.range(0));
  const auto n_test = static_cast<std::size_t>(state.range(1));
  const auto train = make_dataset(n_train);
  const auto test = make_dataset(n_test);
  auto model = make_model();
  auto fit = model.fit(train);
  for (auto _ : state) {
    auto pred = fit.predict(test.features).marginal();
    benchmark::DoNotOptimize(pred);
  }
}
BENCHMARK(BM_GP_PredictMarginal)
    ->Args({1024, 256})
    ->Args({1024, 1024});

// GP joint predict — forms the full M x M covariance.
static void BM_GP_PredictJoint(benchmark::State &state) {
  const auto n_train = static_cast<std::size_t>(state.range(0));
  const auto n_test = static_cast<std::size_t>(state.range(1));
  const auto train = make_dataset(n_train);
  const auto test = make_dataset(n_test);
  auto model = make_model();
  auto fit = model.fit(train);
  for (auto _ : state) {
    auto pred = fit.predict(test.features).joint();
    benchmark::DoNotOptimize(pred);
  }
}
BENCHMARK(BM_GP_PredictJoint)->Args({1024, 256})->Args({1024, 512});
