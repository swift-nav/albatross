/*
 * Copyright (C) 2026 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 */

#include <albatross/Core>
#include <albatross/Indexing>
#include <albatross/linalg/Block>
#include <albatross/src/eigen/serializable_ldlt.hpp>
#include <benchmark/benchmark.h>

#include <random>

namespace {

Eigen::MatrixXd make_spd(Eigen::Index n, double jitter = 1e-3) {
  std::mt19937 rng(7);
  std::normal_distribution<double> dist(0.0, 1.0);
  Eigen::MatrixXd A(n, n);
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index j = 0; j < n; ++j) {
      A(i, j) = dist(rng);
    }
  }
  Eigen::MatrixXd S = A.transpose() * A;
  S.diagonal().array() += jitter * static_cast<double>(n);
  return S;
}

}  // namespace

// Fix #4: SerializableLDLT::matrix() returns by value. Hot when callers compare
// or read the factor.
static void BM_LDLT_matrix_accessor(benchmark::State &state) {
  const auto n = static_cast<Eigen::Index>(state.range(0));
  const auto S = make_spd(n);
  Eigen::SerializableLDLT ldlt(S);
  for (auto _ : state) {
    auto m = ldlt.matrix();
    benchmark::DoNotOptimize(m);
  }
}
BENCHMARK(BM_LDLT_matrix_accessor)->Arg(256)->Arg(1024)->Arg(2048);

// Fix #5: inverse_diagonal currently calls inverse_blocks with N singletons.
static void BM_LDLT_inverse_diagonal(benchmark::State &state) {
  const auto n = static_cast<Eigen::Index>(state.range(0));
  const auto S = make_spd(n);
  Eigen::SerializableLDLT ldlt(S);
  for (auto _ : state) {
    Eigen::VectorXd d = ldlt.inverse_diagonal();
    benchmark::DoNotOptimize(d);
  }
}
BENCHMARK(BM_LDLT_inverse_diagonal)->Arg(128)->Arg(512)->Arg(1024);

// Fix #7: sqrt_transpose densifies L via toDenseMatrix().
static void BM_LDLT_sqrt_transpose(benchmark::State &state) {
  const auto n = static_cast<Eigen::Index>(state.range(0));
  const auto S = make_spd(n);
  Eigen::SerializableLDLT ldlt(S);
  for (auto _ : state) {
    Eigen::MatrixXd m = ldlt.sqrt_transpose();
    benchmark::DoNotOptimize(m);
  }
}
BENCHMARK(BM_LDLT_sqrt_transpose)->Arg(256)->Arg(1024)->Arg(2048);

// sqrt_solve — frequently called inside SparseGP predict.
static void BM_LDLT_sqrt_solve(benchmark::State &state) {
  const auto n = static_cast<Eigen::Index>(state.range(0));
  const auto k = static_cast<Eigen::Index>(state.range(1));
  const auto S = make_spd(n);
  Eigen::SerializableLDLT ldlt(S);
  std::mt19937 rng(11);
  std::normal_distribution<double> nd(0.0, 1.0);
  Eigen::MatrixXd rhs(n, k);
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index j = 0; j < k; ++j) {
      rhs(i, j) = nd(rng);
    }
  }
  for (auto _ : state) {
    Eigen::MatrixXd r = ldlt.sqrt_solve(rhs);
    benchmark::DoNotOptimize(r);
  }
}
BENCHMARK(BM_LDLT_sqrt_solve)->Args({1024, 16})->Args({2048, 32});

// Fix #2-3: BlockSymmetric ctor double-solves; BlockSymmetric::solve has
// duplicated S.solve(rhs_b) and eager rhs.topRows/bottomRows copies.
static void BM_BlockSymmetric_construct(benchmark::State &state) {
  const auto na = static_cast<Eigen::Index>(state.range(0));
  const auto nb = static_cast<Eigen::Index>(state.range(1));
  const auto A = make_spd(na);
  Eigen::SerializableLDLT A_ldlt(A);

  std::mt19937 rng(13);
  std::normal_distribution<double> nd(0.0, 1.0);
  Eigen::MatrixXd B(na, nb);
  for (Eigen::Index i = 0; i < na; ++i) {
    for (Eigen::Index j = 0; j < nb; ++j) {
      B(i, j) = nd(rng);
    }
  }
  Eigen::MatrixXd C = make_spd(nb);

  for (auto _ : state) {
    albatross::BlockSymmetric<Eigen::SerializableLDLT> bs(A_ldlt, B, C);
    benchmark::DoNotOptimize(bs);
  }
}
BENCHMARK(BM_BlockSymmetric_construct)
    ->Args({512, 64})
    ->Args({1024, 128});

static void BM_BlockSymmetric_solve(benchmark::State &state) {
  const auto na = static_cast<Eigen::Index>(state.range(0));
  const auto nb = static_cast<Eigen::Index>(state.range(1));
  const auto kr = static_cast<Eigen::Index>(state.range(2));
  const auto A = make_spd(na);
  Eigen::SerializableLDLT A_ldlt(A);
  std::mt19937 rng(17);
  std::normal_distribution<double> nd(0.0, 1.0);
  Eigen::MatrixXd B(na, nb);
  for (Eigen::Index i = 0; i < na; ++i) {
    for (Eigen::Index j = 0; j < nb; ++j) {
      B(i, j) = nd(rng);
    }
  }
  Eigen::MatrixXd C = make_spd(nb);
  albatross::BlockSymmetric<Eigen::SerializableLDLT> bs(A_ldlt, B, C);
  Eigen::MatrixXd rhs(na + nb, kr);
  for (Eigen::Index i = 0; i < rhs.rows(); ++i) {
    for (Eigen::Index j = 0; j < kr; ++j) {
      rhs(i, j) = nd(rng);
    }
  }
  for (auto _ : state) {
    Eigen::MatrixXd out = bs.solve(rhs);
    benchmark::DoNotOptimize(out);
  }
}
BENCHMARK(BM_BlockSymmetric_solve)
    ->Args({512, 64, 8})
    ->Args({1024, 128, 16});
