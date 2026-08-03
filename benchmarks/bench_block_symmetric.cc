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

// BlockSymmetric is built from a large pre-decomposed block A, the
// cross covariance B and the lower right block C.
struct BlockSymmetricInputs {
  explicit BlockSymmetricInputs(Eigen::Index total) {
    const Eigen::Index na = 3 * total / 4;
    const Eigen::Index nb = total - na;
    A = Eigen::SerializableLDLT(bench::random_psd_matrix(na, 11));
    B = bench::random_matrix(na, nb, 12);
    Eigen::MatrixXd C = bench::random_psd_matrix(nb, 13);
    // Make sure the Schur complement stays comfortably positive definite.
    C += 10. * B.transpose() * A.solve(B);
    this->C = C;
  }

  Eigen::SerializableLDLT A;
  Eigen::MatrixXd B;
  Eigen::MatrixXd C;
};

void BM_block_symmetric_construct(benchmark::State &state) {
  const BlockSymmetricInputs inputs(state.range(0));
  for (auto _ : state) {
    auto block = build_block_symmetric(inputs.A, inputs.B, inputs.C);
    benchmark::DoNotOptimize(block);
  }
}
BENCHMARK(BM_block_symmetric_construct)->Arg(256)->Arg(512);

void BM_block_symmetric_solve_vector(benchmark::State &state) {
  const BlockSymmetricInputs inputs(state.range(0));
  const auto block = build_block_symmetric(inputs.A, inputs.B, inputs.C);
  const Eigen::VectorXd rhs = bench::random_vector(block.rows(), 14);
  for (auto _ : state) {
    Eigen::VectorXd x = block.solve(rhs);
    benchmark::DoNotOptimize(x);
  }
}
BENCHMARK(BM_block_symmetric_solve_vector)->Arg(256)->Arg(512);

void BM_block_symmetric_solve_matrix(benchmark::State &state) {
  const BlockSymmetricInputs inputs(state.range(0));
  const auto block = build_block_symmetric(inputs.A, inputs.B, inputs.C);
  const Eigen::MatrixXd rhs = bench::random_matrix(block.rows(), 32, 15);
  for (auto _ : state) {
    Eigen::MatrixXd x = block.solve(rhs);
    benchmark::DoNotOptimize(x);
  }
}
BENCHMARK(BM_block_symmetric_solve_matrix)->Arg(256)->Arg(512);

} // namespace
} // namespace albatross
