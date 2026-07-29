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

/*
 * Covariance matrix assembly: per-pair scalar kernel calls versus the
 * opt-in matrix-level (_call_matrix_impl) evaluation.
 *
 * The "before" numbers are obtained by evaluating the exact same kernel
 * through a wrapper without a matrix implementation, which is fairer
 * than checking out the parent commit since both paths run in the same
 * binary with the same inlining decisions.
 */

#include <benchmark/benchmark.h>

#include "bench_utils.h"

namespace albatross {
namespace {

constexpr Eigen::Index cDimension = 3;
constexpr double cLengthScale = 2.;
constexpr double cSigma = 1.5;

// The exact same math as SquaredExponential<EuclideanDistance> but
// without the opt-in matrix implementation: this is the per-pair
// ("before") reference.
class PerPairSquaredExponential
    : public CovarianceFunction<PerPairSquaredExponential> {
public:
  std::string name() const { return "per_pair_squared_exponential"; }

  template <
      typename X,
      typename std::enable_if<
          has_call_operator<EuclideanDistance, X &, X &>::value, int>::type = 0>
  double _call_impl(const X &x, const X &y) const {
    return squared_exponential_covariance(distance_metric_(x, y), cLengthScale,
                                          cSigma);
  }

  EuclideanDistance distance_metric_;
};

std::vector<Eigen::VectorXd> random_vector_features(std::size_t n,
                                                    std::uint32_t seed) {
  std::mt19937 gen(seed);
  std::normal_distribution<double> dist(0., 1.);
  std::vector<Eigen::VectorXd> features(n);
  for (auto &f : features) {
    f.resize(cDimension);
    for (Eigen::Index i = 0; i < cDimension; ++i) {
      f[i] = dist(gen);
    }
  }
  return features;
}

void BM_assembly_per_pair(benchmark::State &state) {
  const PerPairSquaredExponential cov;
  const auto n = cast::to_size(state.range(0));
  const auto xs = random_vector_features(n, 0);
  const auto ys = random_vector_features(n, 1);
  for (auto _ : state) {
    Eigen::MatrixXd c = cov(xs, ys);
    benchmark::DoNotOptimize(c);
  }
}
BENCHMARK(BM_assembly_per_pair)->Arg(256)->Arg(512)->Arg(1024)->Arg(2048);

void BM_assembly_matrix(benchmark::State &state) {
  const SquaredExponential<EuclideanDistance> cov(cLengthScale, cSigma);
  const auto n = cast::to_size(state.range(0));
  const auto xs = random_vector_features(n, 0);
  const auto ys = random_vector_features(n, 1);
  for (auto _ : state) {
    Eigen::MatrixXd c = cov(xs, ys);
    benchmark::DoNotOptimize(c);
  }
}
BENCHMARK(BM_assembly_matrix)->Arg(256)->Arg(512)->Arg(1024)->Arg(2048);

// Small sizes to locate the crossover point.  Note that
// BM_assembly_matrix goes through operator() which falls back to the
// per-pair path below MIN_MATRIX_CALL_COEFFICIENTS, so the raw matrix
// implementation is also benchmarked directly.
void BM_assembly_matrix_forced(benchmark::State &state) {
  const SquaredExponential<EuclideanDistance> cov(cLengthScale, cSigma);
  const auto n = cast::to_size(state.range(0));
  const auto xs = random_vector_features(n, 0);
  const auto ys = random_vector_features(n, 1);
  for (auto _ : state) {
    Eigen::MatrixXd c = cov._call_matrix_impl(xs, ys);
    benchmark::DoNotOptimize(c);
  }
}
BENCHMARK(BM_assembly_per_pair)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(64);
BENCHMARK(BM_assembly_matrix)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(64);
BENCHMARK(BM_assembly_matrix_forced)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(64);

template <typename CovFunc>
RegressionDataset<Eigen::VectorXd>
vector_dataset(const CovFunc &, std::size_t n, std::uint32_t seed) {
  const auto features = random_vector_features(n, seed);
  Eigen::VectorXd targets(cast::to_index(n));
  for (std::size_t i = 0; i < n; ++i) {
    targets[cast::to_index(i)] = std::sin(features[i].sum());
  }
  const Eigen::VectorXd variance =
      Eigen::VectorXd::Constant(targets.size(), 0.01);
  return RegressionDataset<Eigen::VectorXd>(
      features, MarginalDistribution(targets, variance));
}

// End-to-end GP mean prediction (fit excluded from the timed loop).
template <typename CovFunc>
void gp_mean_prediction(benchmark::State &state, const CovFunc &cov) {
  auto model = gp_from_covariance(cov, "bench_gp");
  const auto n = cast::to_size(state.range(0));
  const auto dataset = vector_dataset(cov, n, 2);
  const auto test_features = random_vector_features(n, 3);
  const auto fit_model = model.fit(dataset);
  for (auto _ : state) {
    Eigen::VectorXd mean = fit_model.predict(test_features).mean();
    benchmark::DoNotOptimize(mean);
  }
}

void BM_gp_mean_predict_per_pair(benchmark::State &state) {
  gp_mean_prediction(state, PerPairSquaredExponential());
}
BENCHMARK(BM_gp_mean_predict_per_pair)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMillisecond);

void BM_gp_mean_predict_matrix(benchmark::State &state) {
  gp_mean_prediction(
      state, SquaredExponential<EuclideanDistance>(cLengthScale, cSigma));
}
BENCHMARK(BM_gp_mean_predict_matrix)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Arg(2048)
    ->Unit(benchmark::kMillisecond);

} // namespace
} // namespace albatross
