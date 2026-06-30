/*
 * Copyright (C) 2026 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 */

#include <albatross/Core>
#include <albatross/Evaluation>
#include <albatross/Indexing>
#include <albatross/src/evaluation/folds.hpp>
#include <albatross/src/evaluation/prediction_metrics.hpp>
#include <albatross/src/indexing/subset.hpp>
#include <benchmark/benchmark.h>

#include <random>
#include <vector>

namespace {

std::vector<std::size_t> make_held_out(std::size_t n, std::size_t k) {
  std::mt19937 rng(29);
  std::vector<std::size_t> all(n);
  std::iota(all.begin(), all.end(), 0);
  std::shuffle(all.begin(), all.end(), rng);
  all.resize(k);
  std::sort(all.begin(), all.end());
  return all;
}

}  // namespace

// Fix #10: indices_complement currently builds two std::set<size_t>s.
static void BM_indices_complement(benchmark::State &state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto k = static_cast<std::size_t>(state.range(1));
  const auto held_out = make_held_out(n, k);
  for (auto _ : state) {
    auto comp = albatross::indices_complement(held_out, n);
    benchmark::DoNotOptimize(comp);
  }
}
BENCHMARK(BM_indices_complement)
    ->Args({10000, 100})
    ->Args({10000, 1000})
    ->Args({100000, 1000});

// Folds-style: simulate K-fold construction by repeatedly calling
// indices_complement on each fold's test indices.
static void BM_kfold_complements(benchmark::State &state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto k_folds = static_cast<std::size_t>(state.range(1));
  const std::size_t fold_size = n / k_folds;
  std::vector<std::vector<std::size_t>> folds(k_folds);
  for (std::size_t f = 0; f < k_folds; ++f) {
    folds[f].reserve(fold_size);
    for (std::size_t i = f * fold_size; i < (f + 1) * fold_size; ++i) {
      folds[f].push_back(i);
    }
  }
  for (auto _ : state) {
    for (const auto &fold : folds) {
      auto comp = albatross::indices_complement(fold, n);
      benchmark::DoNotOptimize(comp);
    }
  }
}
BENCHMARK(BM_kfold_complements)->Args({10000, 10})->Args({10000, 100});

// Fix #11: variogram_score with no weights allocates an N x N "all ones"
// weight matrix.
static void BM_variogram_score_no_weights(benchmark::State &state) {
  const auto n = static_cast<Eigen::Index>(state.range(0));
  std::mt19937 rng(31);
  std::normal_distribution<double> nd(0.0, 1.0);
  Eigen::MatrixXd A(n, n);
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index j = 0; j < n; ++j) {
      A(i, j) = nd(rng);
    }
  }
  Eigen::MatrixXd cov = A.transpose() * A;
  cov.diagonal().array() += 1.0;
  Eigen::VectorXd mean(n);
  Eigen::VectorXd truth(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    mean[i] = nd(rng);
    truth[i] = nd(rng);
  }
  albatross::JointDistribution prediction(mean, cov);
  for (auto _ : state) {
    double s = albatross::score::variogram_score(prediction, truth);
    benchmark::DoNotOptimize(s);
  }
}
BENCHMARK(BM_variogram_score_no_weights)->Arg(128)->Arg(512)->Arg(1024);
