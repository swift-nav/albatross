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

#include <benchmark/benchmark.h>

#include "bench_utils.h"

namespace albatross {
namespace {

constexpr std::size_t cNumTrain = 512;
constexpr std::size_t cNumTest = 512;

// state.range(0) is the number of threads, 0 means serial (no pool).
std::shared_ptr<ThreadPool> make_pool(std::int64_t num_threads) {
  if (num_threads <= 0) {
    return nullptr;
  }
  return std::make_shared<ThreadPool>(cast::to_size(num_threads));
}

void BM_gp_fit(benchmark::State &state) {
  auto model = gp_from_covariance(bench::bench_covariance(), "bench_gp");
  model.set_thread_pool(make_pool(state.range(0)));
  const auto dataset = bench::random_dataset(cNumTrain, 31);
  for (auto _ : state) {
    auto fit_model = model.fit(dataset);
    benchmark::DoNotOptimize(fit_model);
  }
}
BENCHMARK(BM_gp_fit)->Arg(0)->Arg(4)->Arg(8);

void BM_gp_predict_joint(benchmark::State &state) {
  auto model = gp_from_covariance(bench::bench_covariance(), "bench_gp");
  model.set_thread_pool(make_pool(state.range(0)));
  const auto dataset = bench::random_dataset(cNumTrain, 32);
  const auto test_features = bench::random_features(cNumTest, 33);
  const auto fit_model = model.fit(dataset);
  for (auto _ : state) {
    JointDistribution pred = fit_model.predict(test_features).joint();
    benchmark::DoNotOptimize(pred);
  }
}
BENCHMARK(BM_gp_predict_joint)->Arg(0)->Arg(4)->Arg(8);

void BM_gp_predict_marginal(benchmark::State &state) {
  auto model = gp_from_covariance(bench::bench_covariance(), "bench_gp");
  model.set_thread_pool(make_pool(state.range(0)));
  const auto dataset = bench::random_dataset(cNumTrain, 34);
  const auto test_features = bench::random_features(cNumTest, 35);
  const auto fit_model = model.fit(dataset);
  for (auto _ : state) {
    MarginalDistribution pred = fit_model.predict(test_features).marginal();
    benchmark::DoNotOptimize(pred);
  }
}
BENCHMARK(BM_gp_predict_marginal)->Arg(0)->Arg(4)->Arg(8);

void BM_gp_predict_mean(benchmark::State &state) {
  auto model = gp_from_covariance(bench::bench_covariance(), "bench_gp");
  model.set_thread_pool(make_pool(state.range(0)));
  const auto dataset = bench::random_dataset(cNumTrain, 36);
  const auto test_features = bench::random_features(cNumTest, 37);
  const auto fit_model = model.fit(dataset);
  for (auto _ : state) {
    Eigen::VectorXd pred = fit_model.predict(test_features).mean();
    benchmark::DoNotOptimize(pred);
  }
}
BENCHMARK(BM_gp_predict_mean)->Arg(0)->Arg(4)->Arg(8);

} // namespace
} // namespace albatross
