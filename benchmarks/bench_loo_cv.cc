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

#include <albatross/Evaluation>
#include <benchmark/benchmark.h>

#include "bench_utils.h"

namespace albatross {
namespace {

// The diagonal of the inverse from an LDLT, the core of leave one out
// cross validation.
void BM_inverse_diagonal(benchmark::State &state) {
  const Eigen::Index n = state.range(0);
  const Eigen::SerializableLDLT ldlt(bench::random_psd_matrix(n, 21));
  for (auto _ : state) {
    Eigen::VectorXd diag = ldlt.inverse_diagonal();
    benchmark::DoNotOptimize(diag);
  }
}
BENCHMARK(BM_inverse_diagonal)->Arg(256)->Arg(512)->Arg(1024);

void BM_leave_one_out_conditional(benchmark::State &state) {
  const Eigen::Index n = state.range(0);
  const Eigen::MatrixXd cov = bench::random_psd_matrix(n, 22);
  const Eigen::VectorXd mean = Eigen::VectorXd::Zero(n);
  const JointDistribution prior(mean, cov);
  const MarginalDistribution truth(bench::random_vector(n, 23),
                                   Eigen::VectorXd::Ones(n).asDiagonal());
  for (auto _ : state) {
    MarginalDistribution loo = leave_one_out_conditional(prior, truth);
    benchmark::DoNotOptimize(loo);
  }
}
BENCHMARK(BM_leave_one_out_conditional)->Arg(256)->Arg(512)->Arg(1024);

// Leave one *group* out marginal predictions; exercises
// SerializableLDLT::inverse_blocks and the per group inverse_diagonal
// calls in held_out_prediction.
void BM_leave_one_group_out_marginals(benchmark::State &state) {
  const Eigen::Index n = state.range(0);
  constexpr int cNumGroups = 16;
  const auto features = bench::random_features(cast::to_size(n), 24);
  const auto grouper = [](const double &f) {
    return static_cast<int>(f) % cNumGroups;
  };
  const auto indexer = group_by(features, grouper).indexers();

  const Eigen::MatrixXd cov = bench::random_psd_matrix(n, 25);
  const JointDistribution prior(Eigen::VectorXd::Zero(n), cov);
  const MarginalDistribution truth(bench::random_vector(n, 26),
                                   Eigen::VectorXd::Ones(n).asDiagonal());
  for (auto _ : state) {
    auto marginals =
        leave_one_group_out_conditional_marginals(prior, truth, indexer);
    benchmark::DoNotOptimize(marginals);
  }
}
BENCHMARK(BM_leave_one_group_out_marginals)->Arg(256)->Arg(512)->Arg(1024);

// Same as above but producing joint held out predictions; exercises
// the JointDistribution branch of held_out_prediction.
void BM_leave_one_group_out_joints(benchmark::State &state) {
  const Eigen::Index n = state.range(0);
  constexpr int cNumGroups = 16;
  const auto features = bench::random_features(cast::to_size(n), 24);
  const auto grouper = [](const double &f) {
    return static_cast<int>(f) % cNumGroups;
  };
  const auto indexer = group_by(features, grouper).indexers();

  const Eigen::MatrixXd cov = bench::random_psd_matrix(n, 25);
  const JointDistribution prior(Eigen::VectorXd::Zero(n), cov);
  const MarginalDistribution truth(bench::random_vector(n, 26),
                                   Eigen::VectorXd::Ones(n).asDiagonal());
  for (auto _ : state) {
    auto joints = leave_one_group_out_conditional_joints(prior, truth, indexer);
    benchmark::DoNotOptimize(joints);
  }
}
BENCHMARK(BM_leave_one_group_out_joints)->Arg(256)->Arg(512)->Arg(1024);

// End to end leave one group out cross validation through the GP
// model; exercises fit, FitModel::get_fit and held_out_predictions.
void BM_gp_loo_cv_marginals(benchmark::State &state) {
  const auto dataset = bench::random_dataset(cast::to_size(state.range(0)), 27);
  const auto grouper = [](const double &f) { return static_cast<int>(f) % 8; };
  auto model = gp_from_covariance(bench::bench_covariance(), "bench_gp");
  for (auto _ : state) {
    auto marginals =
        model.cross_validate().predict(dataset, grouper).marginals();
    benchmark::DoNotOptimize(marginals);
  }
}
BENCHMARK(BM_gp_loo_cv_marginals)->Arg(256)->Arg(512);

void BM_gp_loo_cv_joints(benchmark::State &state) {
  const auto dataset = bench::random_dataset(cast::to_size(state.range(0)), 28);
  const auto grouper = [](const double &f) { return static_cast<int>(f) % 8; };
  auto model = gp_from_covariance(bench::bench_covariance(), "bench_gp");
  for (auto _ : state) {
    auto joints = model.cross_validate().predict(dataset, grouper).joints();
    benchmark::DoNotOptimize(joints);
  }
}
BENCHMARK(BM_gp_loo_cv_joints)->Arg(256)->Arg(512);

} // namespace
} // namespace albatross
