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

#include <albatross/SparseGP>
#include <benchmark/benchmark.h>

#include "bench_utils.h"

namespace albatross {
namespace {

// The symmetric product at the heart of a dense GP joint prediction:
//   explained = cross_cov^T K^-1 cross_cov
void BM_gp_joint_prediction(benchmark::State &state) {
  const Eigen::Index n = state.range(0);
  const Eigen::SerializableLDLT train_covariance(
      bench::random_psd_matrix(n, 41));
  const Eigen::MatrixXd cross_cov = bench::random_matrix(n, n, 42);
  const Eigen::MatrixXd prior_cov = bench::random_psd_matrix(n, 43);
  const Eigen::VectorXd information = bench::random_vector(n, 44);
  for (auto _ : state) {
    JointDistribution pred = gp_joint_prediction(cross_cov, prior_cov,
                                                 information, train_covariance);
    benchmark::DoNotOptimize(pred);
  }
}
BENCHMARK(BM_gp_joint_prediction)->Arg(256)->Arg(512);

auto make_sparse_gp() {
  constexpr std::size_t cNumInducing = 64;
  const auto grouper = [](const double &f) { return static_cast<int>(f); };
  return sparse_gp_from_covariance(bench::bench_covariance(), grouper,
                                   UniformlySpacedInducingPoints(cNumInducing),
                                   "bench_sparse_gp");
}

// Exercises the per group P_cols^T P_cols products in
// compute_internal_components.
void BM_sparse_gp_fit(benchmark::State &state) {
  const auto dataset = bench::random_dataset(cast::to_size(state.range(0)), 45);
  auto model = make_sparse_gp();
  for (auto _ : state) {
    auto fit_model = model.fit(dataset);
    benchmark::DoNotOptimize(fit_model);
  }
}
BENCHMARK(BM_sparse_gp_fit)->Arg(1024)->Arg(2048);

// Exercises the Q_sqrt^T Q_sqrt and S_sqrt^T S_sqrt products in the
// sparse GP joint prediction.
void BM_sparse_gp_predict_joint(benchmark::State &state) {
  const auto dataset = bench::random_dataset(1024, 46);
  const auto test_features =
      bench::random_features(cast::to_size(state.range(0)), 47);
  auto model = make_sparse_gp();
  const auto fit_model = model.fit(dataset);
  for (auto _ : state) {
    JointDistribution pred = fit_model.predict(test_features).joint();
    benchmark::DoNotOptimize(pred);
  }
}
BENCHMARK(BM_sparse_gp_predict_joint)->Arg(512)->Arg(1024);

} // namespace
} // namespace albatross
