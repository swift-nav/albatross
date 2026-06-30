/*
 * Copyright (C) 2026 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 */

#include <albatross/CovarianceFunctions>
#include <benchmark/benchmark.h>

#include <random>
#include <vector>

namespace {

std::vector<double> make_scalar_features(std::size_t n) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-100.0, 100.0);
  std::vector<double> xs;
  xs.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    xs.push_back(dist(rng));
  }
  return xs;
}

std::vector<Eigen::VectorXd> make_vector_features(std::size_t n,
                                                  Eigen::Index dim) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<Eigen::VectorXd> xs;
  xs.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    Eigen::VectorXd v(dim);
    for (Eigen::Index k = 0; k < dim; ++k) {
      v[k] = dist(rng);
    }
    xs.push_back(std::move(v));
  }
  return xs;
}

}  // namespace

// Symmetric covariance matrix build for SquaredExponential on scalar features.
// Tests SE-on-scalar; the scalar EuclideanDistance does not call sqrt, so this
// path is mostly an exp/pow benchmark.
static void BM_SE_Scalar_Symmetric(benchmark::State &state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto xs = make_scalar_features(n);
  albatross::SquaredExponential<albatross::EuclideanDistance> cov;
  for (auto _ : state) {
    Eigen::MatrixXd K = cov(xs);
    benchmark::DoNotOptimize(K);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n) *
                          static_cast<int64_t>(n));
}
BENCHMARK(BM_SE_Scalar_Symmetric)->Arg(256)->Arg(1024)->Arg(2048);

// Symmetric covariance matrix build for SquaredExponential on Eigen::VectorXd
// features. This is the path hit by Fix #1 — EuclideanDistance::operator() on
// vectors does sqrt, which SE then squares back.
static void BM_SE_VectorXd_Symmetric(benchmark::State &state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto xs = make_vector_features(n, 8);
  albatross::SquaredExponential<albatross::EuclideanDistance> cov;
  for (auto _ : state) {
    Eigen::MatrixXd K = cov(xs);
    benchmark::DoNotOptimize(K);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n) *
                          static_cast<int64_t>(n));
}
BENCHMARK(BM_SE_VectorXd_Symmetric)->Arg(256)->Arg(1024)->Arg(2048);

// Cross covariance matrix build (rectangular) for SquaredExponential.
static void BM_SE_VectorXd_Cross(benchmark::State &state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto m = static_cast<std::size_t>(state.range(1));
  const auto xs = make_vector_features(n, 8);
  const auto ys = make_vector_features(m, 8);
  albatross::SquaredExponential<albatross::EuclideanDistance> cov;
  for (auto _ : state) {
    Eigen::MatrixXd K = cov(xs, ys);
    benchmark::DoNotOptimize(K);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n) *
                          static_cast<int64_t>(m));
}
BENCHMARK(BM_SE_VectorXd_Cross)
    ->Args({1024, 256})
    ->Args({2048, 512});

// Matern52 — Fix #1 reasoning applies to the extent radial kernels recompute
// 1/length_scale and sigma^2 per call.
static void BM_Matern52_VectorXd_Symmetric(benchmark::State &state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto xs = make_vector_features(n, 8);
  albatross::Matern52<albatross::EuclideanDistance> cov;
  for (auto _ : state) {
    Eigen::MatrixXd K = cov(xs);
    benchmark::DoNotOptimize(K);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n) *
                          static_cast<int64_t>(n));
}
BENCHMARK(BM_Matern52_VectorXd_Symmetric)->Arg(256)->Arg(1024)->Arg(2048);

// Polynomial — Fix #8 hits the per-call map lookup + pow().
static void BM_Polynomial3_Scalar_Symmetric(benchmark::State &state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto xs = make_scalar_features(n);
  albatross::Polynomial<3> cov;
  for (auto _ : state) {
    Eigen::MatrixXd K = cov(xs);
    benchmark::DoNotOptimize(K);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n) *
                          static_cast<int64_t>(n));
}
BENCHMARK(BM_Polynomial3_Scalar_Symmetric)->Arg(256)->Arg(1024)->Arg(2048);

// Sum-of-kernels (SE + IndependentNoise) — common GP setup. Hits the diagonal
// short-circuit and the symmetric-build path together.
static void BM_SE_plus_Noise_VectorXd_Symmetric(benchmark::State &state) {
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto xs = make_vector_features(n, 8);
  albatross::SquaredExponential<albatross::EuclideanDistance> se;
  albatross::IndependentNoise<Eigen::VectorXd> noise;
  auto cov = se + noise;
  for (auto _ : state) {
    Eigen::MatrixXd K = cov(xs);
    benchmark::DoNotOptimize(K);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(n) *
                          static_cast<int64_t>(n));
}
BENCHMARK(BM_SE_plus_Noise_VectorXd_Symmetric)->Arg(256)->Arg(1024)->Arg(2048);
