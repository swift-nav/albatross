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

#include <gtest/gtest.h>

#include "test_models.h"

#include <albatross/CGGP>

namespace albatross {

static constexpr const std::size_t cDefaultSeed = 42;
static constexpr const double cSolveTolerance = 1.e-6;

namespace {

// Solve AX = B column-by-column using Eigen's CG with the given
// preconditioner, to produce a reference solution.
template <typename Preconditioner>
Eigen::MatrixXd eigen_cg_reference(const Eigen::MatrixXd &A,
                                   const Eigen::MatrixXd &B,
                                   Preconditioner &precond) {
  Eigen::MatrixXd X(B.rows(), B.cols());
  for (Eigen::Index j = 0; j < B.cols(); ++j) {
    Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower, Preconditioner>
        solver(A);
    solver.setTolerance(1.e-10);
    solver.setMaxIterations(500);
    X.col(j) = solver.solve(B.col(j));
    EXPECT_EQ(solver.info(), Eigen::Success)
        << "Eigen CG failed on column " << j;
  }
  return X;
}

} // namespace

TEST(BatchedPCG, SingleRhsIdentityPreconditioner) {
  std::default_random_engine gen{cDefaultSeed};
  constexpr Eigen::Index n = 30;
  const auto A = random_covariance_matrix(n, gen);
  const Eigen::VectorXd b{Eigen::VectorXd::NullaryExpr(
      n, [&gen]() { return std::normal_distribution<double>(0, 1)(gen); })};

  Eigen::IdentityPreconditioner precond;
  precond.compute(A);

  Eigen::BatchedPCGConfig config;
  config.tolerance = 1.e-10;
  config.max_iterations = 50;

  const Eigen::MatrixXd B = b;
  const auto result = Eigen::batched_preconditioned_conjugate_gradient(
      A, precond, B, config, Eigen::MatrixXd{Eigen::MatrixXd::Zero(n, 1)});

  // Reference: Eigen's own CG
  Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower,
                           Eigen::IdentityPreconditioner>
      ref_solver(A);
  ref_solver.setTolerance(1.e-10);
  const Eigen::VectorXd x_ref = ref_solver.solve(b);
  ASSERT_EQ(ref_solver.info(), Eigen::Success);

  EXPECT_LT((result.X - x_ref).norm(), cSolveTolerance)
      << "Batched PCG solution differs from Eigen CG reference; A:\n"
      << A << "\nb:\n\t" << b.transpose() << "\nmbpcg:\n\t"
      << result.X.transpose() << "\neigen:\n\t" << x_ref.transpose();
  EXPECT_LT((A * result.X - B).norm(), cSolveTolerance)
      << "Batched PCG residual too large";
}

TEST(BatchedPCG, MultipleRhsIdentityPreconditioner) {
  std::default_random_engine gen{cDefaultSeed};
  constexpr Eigen::Index n = 30;
  constexpr Eigen::Index num_rhs = 5;
  const auto A = random_covariance_matrix(n, gen);
  const Eigen::MatrixXd B{Eigen::MatrixXd::NullaryExpr(n, num_rhs, [&gen]() {
    return std::normal_distribution<double>(0, 1)(gen);
  })};

  Eigen::IdentityPreconditioner precond;
  precond.compute(A);

  Eigen::BatchedPCGConfig config;
  config.tolerance = 1.e-10;
  config.max_iterations = 500;

  const auto result = Eigen::batched_preconditioned_conjugate_gradient(
      A, precond, B, config,
      Eigen::MatrixXd{Eigen::MatrixXd::Zero(n, num_rhs)});

  Eigen::IdentityPreconditioner ref_precond;
  const Eigen::MatrixXd X_ref = eigen_cg_reference(A, B, ref_precond);

  EXPECT_LT((result.X - X_ref).norm(), cSolveTolerance)
      << "Batched PCG multi-RHS solution differs from column-by-column CG";
  EXPECT_LT((A * result.X - B).norm(), cSolveTolerance)
      << "Batched PCG multi-RHS residual too large";
}

TEST(BatchedPCG, MultipleRhsPartialCholeskyPreconditioner) {
  std::default_random_engine gen{cDefaultSeed};
  constexpr Eigen::Index n = 50;
  constexpr Eigen::Index num_rhs = 4;
  auto A = random_covariance_matrix(n, gen);
  A += Eigen::VectorXd::Constant(n, 1.e-5).asDiagonal();
  const Eigen::MatrixXd B{Eigen::MatrixXd::NullaryExpr(n, num_rhs, [&gen]() {
    return std::normal_distribution<double>(0, 1)(gen);
  })};

  Eigen::PartialCholesky<double> precond;
  precond.setOrder(n / 4);
  precond.compute(A);
  ASSERT_EQ(precond.info(), Eigen::Success);

  Eigen::BatchedPCGConfig config;
  config.tolerance = 1.e-10;
  config.max_iterations = 500;

  const auto result = Eigen::batched_preconditioned_conjugate_gradient(
      A, precond, B, config,
      Eigen::MatrixXd{Eigen::MatrixXd::Zero(n, num_rhs)});

  // Reference: column-by-column Eigen CG with the same preconditioner type
  Eigen::MatrixXd X_ref(n, num_rhs);
  for (Eigen::Index j = 0; j < num_rhs; ++j) {
    Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower,
                             Eigen::PartialCholesky<double>>
        solver(A);
    solver.setTolerance(1.e-10);
    solver.setMaxIterations(500);
    solver.preconditioner().setOrder(n / 4);
    X_ref.col(j) = solver.solve(B.col(j));
    ASSERT_EQ(solver.info(), Eigen::Success)
        << "Eigen CG with PartialCholesky failed on column " << j;
  }

  EXPECT_LT((A * result.X - B).norm(), cSolveTolerance)
      << "Batched PCG with PartialCholesky residual too large";

  // Both should solve the system; they may take different iteration
  // paths, but the solutions should be close.
  const double ref_residual = (A * X_ref - B).norm();
  const double batched_residual = (A * result.X - B).norm();
  EXPECT_LT(batched_residual, cSolveTolerance);
  EXPECT_LT(ref_residual, cSolveTolerance);
}

// Helper to convert a BidiagonalCholeskyFactor into a dense lower-triangular
// matrix.
Eigen::MatrixXd to_dense(const Eigen::BidiagonalCholeskyFactor &chol) {
  const Eigen::Index k = chol.rows();
  Eigen::MatrixXd L = Eigen::MatrixXd::Zero(k, k);
  L.diagonal() = chol.diagonal;
  if (k > 1) {
    L.diagonal(-1) = chol.subdiagonal;
  }
  return L;
}

// Helper to convert a SymmetricTridiagonalization into a dense matrix.
Eigen::MatrixXd to_dense(const Eigen::SymmetricTridiagonalization &trid) {
  const Eigen::Index k = trid.rows();
  Eigen::MatrixXd T = Eigen::MatrixXd::Zero(k, k);
  T.diagonal() = trid.diagonal;
  if (k > 1) {
    T.diagonal(-1) = trid.subdiagonal;
    T.diagonal(1) = trid.subdiagonal;
  }
  return T;
}

// This property doesn't hold in general:
//
//  - Lanczos is reorthogonalised at every step, whereas CG is not
//
//  - CG resets the residuals directly from A and B every few
//    iterations, which causes the sequence to diverge.  even
//    disabling this behaviour does not make them match.
//
// So you may not always get good results out of the CG-derived ones,
// but here at least we do a sanity check that for this particular
// problem, the results are the same (e.g. we haven't messed up our
// indexing or something blatant).
TEST(BatchedPCG, TridiagonalMatchesLanczos) {
  std::default_random_engine gen{cDefaultSeed};
  constexpr Eigen::Index n = 30;
  constexpr Eigen::Index num_rhs = 6;
  auto A = random_covariance_matrix(n, gen);
  A += Eigen::VectorXd::Constant(n, 1.e-5).asDiagonal();
  const Eigen::MatrixXd B{Eigen::MatrixXd::NullaryExpr(n, num_rhs, [&gen]() {
    return std::normal_distribution<double>(0, 1)(gen);
  })};

  Eigen::IdentityPreconditioner precond;
  precond.compute(A);

  // Run batched PCG requesting tridiagonal outputs for all columns.
  Eigen::BatchedPCGConfig config;
  config.tolerance = 1.e-10;
  config.max_iterations = n;
  config.compute_tridiagonals = num_rhs;

  const auto result = Eigen::batched_preconditioned_conjugate_gradient(
      A, precond, B, config,
      Eigen::MatrixXd{Eigen::MatrixXd::Zero(n, num_rhs)});

  // The solve itself should have succeeded.
  ASSERT_LT((A * result.X - B).norm(), cSolveTolerance)
      << "Batched PCG solve failed";
  ASSERT_EQ(static_cast<Eigen::Index>(result.trids.size()), num_rhs);

  // For each RHS column, run Lanczos on A with start vector = b_j.
  // With identity preconditioner, no pre-application is needed.
  for (Eigen::Index j = 0; j < num_rhs; ++j) {
    const Eigen::VectorXd &b_j = B.col(j);
    const Eigen::Index k = result.trids[j].rows();
    ASSERT_GT(k, 0) << "Tridiagonal " << j << " is empty";

    const auto lanczos = Eigen::reorthLanczosBasis(A, b_j, static_cast<int>(k));

    const Eigen::MatrixXd T_cg = to_dense(result.trids[j]);
    const Eigen::MatrixXd T_lanczos = to_dense(lanczos.T);

    ASSERT_EQ(T_cg.rows(), T_lanczos.rows())
        << "Tridiagonal size mismatch for column " << j;
    ASSERT_EQ(T_cg.cols(), T_lanczos.cols())
        << "Tridiagonal size mismatch for column " << j;

    EXPECT_LT((T_cg - T_lanczos).norm(), cSolveTolerance)
        << "Tridiagonal mismatch for column " << j << "\nCG T:\n"
        << T_cg << "\nLanczos T:\n"
        << T_lanczos;
  }
}

TEST(BatchedPCG, TridiagonalLLTMatchesEigen) {
  std::default_random_engine gen{cDefaultSeed};
  constexpr Eigen::Index n = 30;
  constexpr Eigen::Index num_rhs = 6;
  auto A = random_covariance_matrix(n, gen);
  A += Eigen::VectorXd::Constant(n, 1.e-5).asDiagonal();
  const Eigen::MatrixXd B{Eigen::MatrixXd::NullaryExpr(n, num_rhs, [&gen]() {
    return std::normal_distribution<double>(0, 1)(gen);
  })};

  Eigen::IdentityPreconditioner precond;
  precond.compute(A);

  Eigen::BatchedPCGConfig config;
  config.tolerance = 1.e-10;
  config.max_iterations = n;
  config.compute_tridiagonals = num_rhs;

  const auto result = Eigen::batched_preconditioned_conjugate_gradient(
      A, precond, B, config,
      Eigen::MatrixXd{Eigen::MatrixXd::Zero(n, num_rhs)});

  ASSERT_LT((A * result.X - B).norm(), cSolveTolerance)
      << "Batched PCG solve failed";
  ASSERT_EQ(static_cast<Eigen::Index>(result.trids.size()), num_rhs);

  for (Eigen::Index j = 0; j < num_rhs; ++j) {
    const auto &trid = result.trids[j];
    ASSERT_GT(trid.rows(), 0) << "Tridiagonal " << j << " is empty";

    // Our bidiagonal Cholesky factor
    const auto chol = Eigen::tridiagonal_llt(trid);
    const Eigen::MatrixXd L = to_dense(chol);

    // Eigen's LLT on the dense tridiagonal matrix
    const Eigen::MatrixXd T = to_dense(trid);
    Eigen::LLT<Eigen::MatrixXd> eigen_llt(T);
    ASSERT_EQ(eigen_llt.info(), Eigen::Success)
        << "Eigen LLT failed on tridiagonal " << j;
    const Eigen::MatrixXd L_ref = eigen_llt.matrixL();

    EXPECT_LT((L - L_ref).norm(), cSolveTolerance)
        << "tridiagonal_llt mismatch for column " << j << "\nOurs:\n"
        << L << "\nEigen:\n"
        << L_ref;

    // Also verify L L^T reconstructs T
    EXPECT_LT((L * L.transpose() - T).norm(), cSolveTolerance)
        << "L L^T reconstruction failed for column " << j;
  }
}

TEST(BatchedPCG, RandomProblems) {
  std::default_random_engine gen{cDefaultSeed};
  std::uniform_int_distribution<Eigen::Index> size_dist(10, 60);
  std::uniform_int_distribution<Eigen::Index> rhs_dist(1, 8);

  constexpr std::size_t num_problems = 100;

  for (std::size_t i = 0; i < num_problems; ++i) {
    const Eigen::Index n = size_dist(gen);
    const Eigen::Index num_rhs = rhs_dist(gen);

    auto A = random_covariance_matrix(n, gen);
    A += Eigen::VectorXd::Constant(n, 1.e-5).asDiagonal();

    const Eigen::MatrixXd B{Eigen::MatrixXd::NullaryExpr(n, num_rhs, [&gen]() {
      return std::normal_distribution<double>(0, 1)(gen);
    })};

    Eigen::IdentityPreconditioner precond;
    precond.compute(A);

    Eigen::BatchedPCGConfig config;
    config.tolerance = 1.e-10;
    config.max_iterations = 500;

    const auto result = Eigen::batched_preconditioned_conjugate_gradient(
        A, precond, B, config,
        Eigen::MatrixXd{Eigen::MatrixXd::Zero(n, num_rhs)});

    Eigen::IdentityPreconditioner ref_precond;
    const Eigen::MatrixXd X_ref = eigen_cg_reference(A, B, ref_precond);

    EXPECT_LT((A * result.X - B).norm(), cSolveTolerance)
        << "Problem " << i << " (n=" << n << ", rhs=" << num_rhs
        << "): batched PCG residual too large";
    EXPECT_LT((result.X - X_ref).norm(), cSolveTolerance)
        << "Problem " << i << " (n=" << n << ", rhs=" << num_rhs
        << "): batched PCG differs from Eigen CG reference";
  }
}

TEST(BatchedPCG, RandomTridiagonalLLTMatchesEigen) {
  std::default_random_engine gen{cDefaultSeed};
  std::uniform_int_distribution<Eigen::Index> size_dist(10, 60);
  std::uniform_int_distribution<Eigen::Index> rhs_dist(1, 8);

  constexpr std::size_t num_problems = 50;

  for (std::size_t i = 0; i < num_problems; ++i) {
    const Eigen::Index n = size_dist(gen);
    const Eigen::Index num_rhs = rhs_dist(gen);

    auto A = random_covariance_matrix(n, gen);
    A += Eigen::VectorXd::Constant(n, 1.e-5).asDiagonal();

    const Eigen::MatrixXd B{Eigen::MatrixXd::NullaryExpr(n, num_rhs, [&gen]() {
      return std::normal_distribution<double>(0, 1)(gen);
    })};

    Eigen::IdentityPreconditioner precond;
    precond.compute(A);

    Eigen::BatchedPCGConfig config;
    config.tolerance = 1.e-10;
    config.max_iterations = 500;
    config.compute_tridiagonals = num_rhs;

    const auto result = Eigen::batched_preconditioned_conjugate_gradient(
        A, precond, B, config,
        Eigen::MatrixXd{Eigen::MatrixXd::Zero(n, num_rhs)});

    ASSERT_LT((A * result.X - B).norm(), cSolveTolerance)
        << "Problem " << i << " (n=" << n << ", rhs=" << num_rhs
        << "): batched PCG solve failed";
    ASSERT_EQ(static_cast<Eigen::Index>(result.trids.size()), num_rhs);

    for (Eigen::Index j = 0; j < num_rhs; ++j) {
      const auto &trid = result.trids[j];
      ASSERT_GT(trid.rows(), 0)
          << "Problem " << i << ", column " << j << ": tridiagonal is empty";

      const auto chol = Eigen::tridiagonal_llt(trid);
      const Eigen::MatrixXd L = to_dense(chol);

      const Eigen::MatrixXd T = to_dense(trid);
      Eigen::LLT<Eigen::MatrixXd> eigen_llt(T);
      ASSERT_EQ(eigen_llt.info(), Eigen::Success)
          << "Problem " << i << ", column " << j
          << ": Eigen LLT failed on tridiagonal";
      const Eigen::MatrixXd L_ref = eigen_llt.matrixL();

      EXPECT_LT((L - L_ref).norm(), cSolveTolerance)
          << "Problem " << i << " (n=" << n << ", rhs=" << num_rhs
          << "), column " << j << ": tridiagonal_llt mismatch";

      EXPECT_LT((L * L.transpose() - T).norm(), cSolveTolerance)
          << "Problem " << i << " (n=" << n << ", rhs=" << num_rhs
          << "), column " << j << ": L L^T reconstruction failed";
    }
  }
}

} // namespace albatross
