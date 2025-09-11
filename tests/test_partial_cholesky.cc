/*
 * Copyright (C) 2025 Swift Navigation Inc.
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
#include <chrono>

#include <albatross/CGGP>

namespace albatross {

TEST(PartialCholesky, Construct) {
  Eigen::PartialCholesky<double> p;
  EXPECT_EQ(Eigen::Success, p.info());
}

static constexpr const std::size_t cDefaultSeed = 22;
static constexpr const Eigen::Index cExampleSize = 10;

static std::gamma_distribution<double> cEigenvalueDistribution(2, 2);

TEST(PartialCholesky, Compute) {
  Eigen::PartialCholesky<double> p;
  ASSERT_EQ(Eigen::Success, p.info());

  std::default_random_engine gen{cDefaultSeed};
  const auto m =
      random_covariance_matrix(cExampleSize, cEigenvalueDistribution, gen);

  p.compute(m);

  ASSERT_EQ(Eigen::Success, p.info());

  EXPECT_LT((p.reconstructedMatrix() - m).norm(), 1.e-9)
      << p.reconstructedMatrix() - m;
}

namespace {

Eigen::MatrixXd add_diagonal(const Eigen::MatrixXd &A, double sigma) {
  return A +
         Eigen::MatrixXd(
             Eigen::VectorXd::Constant(A.rows(), sigma * sigma).asDiagonal());
}

} // namespace

TEST(PartialCholesky, Solve) {
  Eigen::PartialCholesky<double> p;
  ASSERT_EQ(Eigen::Success, p.info());

  std::default_random_engine gen{cDefaultSeed};
  const auto m =
      random_covariance_matrix(cExampleSize, cEigenvalueDistribution, gen);
  const Eigen::VectorXd b{Eigen::VectorXd::NullaryExpr(cExampleSize, [&gen]() {
    return std::normal_distribution<double>(0, 1)(gen);
  })};

  p.compute(m);
  ASSERT_EQ(Eigen::Success, p.info());

  const Eigen::VectorXd x = p.solve(b);
  ASSERT_EQ(Eigen::Success, p.info());

  const Eigen::VectorXd x_llt = add_diagonal(m, p.nugget()).ldlt().solve(b);
  EXPECT_LT((x - x_llt).norm(), p.nugget());
}

TEST(PartialCholesky, SolveMatrix) {
  Eigen::PartialCholesky<double> p;
  ASSERT_EQ(Eigen::Success, p.info());

  std::default_random_engine gen{cDefaultSeed};
  const auto m =
      random_covariance_matrix(cExampleSize, cEigenvalueDistribution, gen);
  const Eigen::MatrixXd b{
      Eigen::MatrixXd::NullaryExpr(cExampleSize, cExampleSize, [&gen]() {
        return std::normal_distribution<double>(0, 1)(gen);
      })};

  p.compute(m);
  ASSERT_EQ(Eigen::Success, p.info());

  const Eigen::MatrixXd x = p.solve(b);
  ASSERT_EQ(Eigen::Success, p.info());

  const Eigen::MatrixXd x_llt = add_diagonal(m, p.nugget()).ldlt().solve(b);
  EXPECT_LT((x - x_llt).norm(), 1.e-9);
}

static constexpr const std::size_t cNumRandomExamples = 222;
static constexpr const std::size_t cMaxDecompositionOrder = 30;

TEST(PartialCholesky, RandomCompleteProblems) {
  std::default_random_engine gen{cDefaultSeed};
  std::uniform_int_distribution<Eigen::Index> decomp_order_dist(
      1, cMaxDecompositionOrder);

  for (std::size_t i = 0; i < cNumRandomExamples; ++i) {
    const Eigen::Index decomp_order{decomp_order_dist(gen)};
    // Generate matrices small enough that we always do a complete
    // Cholesky factorisation.
    std::uniform_int_distribution<Eigen::Index> problem_size_dist(1,
                                                                  decomp_order);
    const Eigen::Index problem_size{problem_size_dist(gen)};
    const auto m =
        random_covariance_matrix(problem_size, cEigenvalueDistribution, gen);
    const Eigen::VectorXd b{
        Eigen::VectorXd::NullaryExpr(problem_size, [&gen]() {
          return std::normal_distribution<double>(0, 1)(gen);
        })};

    Eigen::PartialCholesky<double> p;
    ASSERT_EQ(Eigen::Success, p.info());
    p.setOrder(decomp_order);

    p.compute(m);
    ASSERT_EQ(Eigen::Success, p.info());

    EXPECT_LT((p.reconstructedMatrix() - m).norm(), 1.e-9)
        << p.reconstructedMatrix() - m << "\neigs:\n"
        << Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(m)
               .eigenvalues()
               .transpose();

    const Eigen::VectorXd x = p.solve(b);
    ASSERT_EQ(Eigen::Success, p.info());

    const Eigen::VectorXd x_llt =
        (m + Eigen::MatrixXd(
                 Eigen::VectorXd::Constant(m.rows(), p.nugget() * p.nugget())
                     .asDiagonal()))
            .ldlt()
            .solve(b);
    // A cleverer implementation would derive the nugget size from the
    // eigenvalues of the original covariance rather than add a fudge
    // factor.
    EXPECT_LT((x - x_llt).norm(), 1.e-8)
        << "\neigs:\n"
        << Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(m)
               .eigenvalues()
               .transpose();
  }
}

namespace {

Eigen::VectorXd spd_eigs(const Eigen::MatrixXd &m) {
  return Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(m).eigenvalues();
}

double condition_number(const Eigen::MatrixXd &m) {
  Eigen::VectorXd eigs{spd_eigs(m)};
  return eigs.array().abs().maxCoeff() / eigs.array().abs().minCoeff();
}

} // namespace

TEST(PartialCholesky, IncreasingRank) {
  std::default_random_engine gen{cDefaultSeed};
  const Eigen::MatrixXd m = random_covariance_matrix(cExampleSize, gen);
  const Eigen::VectorXd b{Eigen::VectorXd::NullaryExpr(cExampleSize, [&gen]() {
    return std::normal_distribution<double>(0, 1)(gen);
  })};

  std::vector<Eigen::PartialCholesky<double>> decomps(cExampleSize);

  // Eigen::IOFormat high_prec_python(16, 0, ", ", "\n", "[", "]", "[", "]");
  // std::cout << m.format(high_prec_python) << std::endl;

  const Eigen::MatrixXd m_diag{add_diagonal(m, decomps[0].nugget())};
  // std::cout << m_diag.format(high_prec_python) << std::endl;
  const double mcond{condition_number(m_diag)};

  decomps[0].setOrder(1);
  // decomps[0].setNugget(std::sqrt(m.diagonal().minCoeff()));
  decomps[0].compute(m);
  ASSERT_EQ(decomps[0].info(), Eigen::Success);

  const auto triangle = [](Eigen::Index index,
                           const auto &a) -> Eigen::MatrixXd {
    return a.matrixL()
        .topLeftCorner(index, index)
        .template triangularView<Eigen::Lower>();
  };

  const auto triangle_compare =
      [&decomps, &triangle](Eigen::Index index) -> Eigen::MatrixXd {
    return triangle(index, decomps[index - 1]) -
           triangle(index, decomps[index]);
  };

  const auto bottom_left = [](Eigen::Index index,
                              const auto &a) -> Eigen::MatrixXd {
    return (a.permutationsP() * a.matrixL())
        .bottomLeftCorner(a.cols() - index, index);
  };

  const auto ordered_bottom_left_compare =
      [&decomps, &bottom_left](Eigen::Index index) -> Eigen::MatrixXd {
    return bottom_left(index, decomps[index - 1]) -
           bottom_left(index, decomps[index]);
  };

  const auto reconstruction_error = [&m](const auto &a) -> Eigen::MatrixXd {
    return a.reconstructedMatrix() - m;
  };

  const auto preconditioned_condition = [&m_diag](const auto &a) {
    return condition_number(a.solve(m_diag));
  };

  // const auto preconditioned_eigs = [&m_diag](const auto &a) {
  //   return spd_eigs(a.solve(m_diag));
  // };

  for (Eigen::Index k = 1; k < cExampleSize; ++k) {
    const Eigen::Index rank = k + 1;
    decomps[k].setOrder(rank);
    // decomps[k].setNugget(std::sqrt(m.diagonal().minCoeff()));
    decomps[k].compute(m);
    ASSERT_EQ(decomps[k].info(), Eigen::Success);
    // The top left triangle should be the same between successive
    // low-rank approximations
    EXPECT_LT(triangle_compare(k).norm(), 1.e-9)
        << "k = " << k << "; " << triangle(k, decomps[k - 1]) << "\n"
        << triangle(k, decomps[k]);
    // The bottom left block below the top triangle corresponds to the
    // pivot / non-pivot cross-correlation block and should be the
    // same up to a permutation.
    EXPECT_LT(ordered_bottom_left_compare(k).norm(), 1.e-9)
        << "k = " << k << "; " << bottom_left(k, decomps[k - 1]) << "\n"
        << bottom_left(k, decomps[k]);
    // The leftover Schur complement determinant should decrease with
    // increasing approximation rank
    EXPECT_LT(decomps[k].error(), decomps[k - 1].error()) << "k = " << k;
    // The reconstructed matrix should get closer and closer to the
    // true matrix as we increase the approximation rank
    EXPECT_LT(reconstruction_error(decomps[k]).norm(),
              reconstruction_error(decomps[k - 1]).norm())
        << "k = " << k;
    // The whole point of this decomposition is that preconditioning
    // by it should improve the condition number of the original
    // problem
    EXPECT_LT(preconditioned_condition(decomps[k]),
              preconditioned_condition(decomps[k - 1]))
        << "k = " << k << "  mcond: " << mcond
        << "  err: " << decomps[k].error();
    // << "\nm_diag eigs: " << spd_eigs(m_diag).transpose()
    // << "\nprecond eigs: "
    // << spd_eigs(
    //        decomps[k].solve(Eigen::MatrixXd::Identity(m.rows(), m.cols())))
    //        .transpose()
    // << "\nprecond eigs direct: "
    // << spd_eigs(decomps[k].direct_solve().solve(
    //                 Eigen::MatrixXd::Identity(m.rows(), m.cols())))
    //        .transpose()
    // << "\n    k eigs: " << preconditioned_eigs(decomps[k]).transpose()
    // << "\nk - 1 eigs: " << preconditioned_eigs(decomps[k - 1]).transpose();
    EXPECT_LT(preconditioned_condition(decomps[k]), mcond)
        << "k = " << k << "  err: " << decomps[k].error(); //  << "\nL:\n"
    //     << decomps[k].matrixL() << "\nP:\n"
    //     << Eigen::MatrixXi(decomps[k].permutationsP())
    //     << "\nm_diag eigs: " << spd_eigs(m_diag).transpose()
    //     << "\nprecond eigs: "
    //     << spd_eigs(
    //            decomps[k].solve(Eigen::MatrixXd::Identity(m.rows(),
    //            m.cols()))) .transpose()
    //     << "\nprecond eigs direct: "
    //     << spd_eigs(decomps[k].direct_solve().solve(
    //                     Eigen::MatrixXd::Identity(m.rows(), m.cols())))
    //            .transpose()
    //     << "\ndiag eigs: "
    //     << spd_eigs(m * m.diagonal().array().inverse().matrix().asDiagonal())
    //            .transpose()
    //     << "\n    k eigs: " << preconditioned_eigs(decomps[k]).transpose()
    //     << "\nk - 1 eigs: " << preconditioned_eigs(decomps[k -
    //     1]).transpose();
    // if (k == 2) {
    //   std::cout << "k = 2, rank 3.  precond(m_diag):\n"
    //             << decomps[k].solve(m_diag) << std::endl;
    // }
  }
}

// TEST(PartialCholesky, Approximate) {
//   std::default_random_engine gen{cDefaultSeed};
//   const Eigen::MatrixXd m =
//       random_covariance_matrix(cExampleSize, cEigenvalueDistribution, gen);
//   const Eigen::VectorXd b{Eigen::VectorXd::NullaryExpr(cExampleSize, [&gen]()
//   {
//     return std::normal_distribution<double>(0, 1)(gen);
//   })};

//   Eigen::PartialCholesky<double> p;
//   ASSERT_EQ(Eigen::Success, p.info());
//   p.setOrder(2);
//   p.compute(m);
//   ASSERT_EQ(Eigen::Success, p.info());
//   const Eigen::MatrixXd L{p.matrixL()};
//   for (Eigen::Index i = 1; i < L.cols(); ++i) {
//     EXPECT_LT(L(i, i), L(i - 1, i - 1)) << L.diagonal().transpose();
//   }
//   const Eigen::VectorXd x = p.solve(b);
//   ASSERT_EQ(Eigen::Success, p.info());
//   Eigen::MatrixXd preconditioned{p.solve(m)};
//   Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> preeigs(preconditioned);
//   std::cout << "A:\n" << m << std::endl;
//   std::cout << "reconstructed:\n" << p.reconstructedMatrix() << std::endl;
//   std::cout << "error: " << p.error() << std::endl;
//   std::cout << "diff:\n" << m - p.reconstructedMatrix() << std::endl;
//   std::cout << "L:\n" << p.matrixL() << std::endl;
//   std::cout << "P:\n" << Eigen::MatrixXi(p.permutationsP()) << std::endl;
//   std::cout << m * x - b << std::endl;
//   std::cout << m * (p.reconstructedMatrix().ldlt().solve(b)) - b <<
//   std::endl; std::cout << "Aapprox^-1 A:\n" << preconditioned << std::endl;
//   Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigs(m);
//   std::cout << "A eigs: " << eigs.eigenvalues().transpose() << std::endl;
//   std::cout << "Aapprox^-1 A eigs: " << preeigs.eigenvalues().transpose()
//             << std::endl;
//   Eigen::PartialCholesky<double> p3;
//   p3.setOrder(3);
//   p3.compute(m);
//   std::cout << "L:\n" << p3.matrixL() << std::endl;
//   std::cout << "P:\n" << Eigen::MatrixXi(p3.permutationsP()) << std::endl;
// }

TEST(PartialCholesky, Precondition) {
  std::default_random_engine gen{cDefaultSeed};
  const auto m =
      random_covariance_matrix(cExampleSize, cEigenvalueDistribution, gen);
  const Eigen::VectorXd b{Eigen::VectorXd::NullaryExpr(cExampleSize, [&gen]() {
    return std::normal_distribution<double>(0, 1)(gen);
  })};

  Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower,
                           Eigen::PartialCholesky<double>>
      solver(m);
  solver.preconditioner().setOrder(cExampleSize / 4);

  const Eigen::VectorXd x = solver.solve(b);
  EXPECT_EQ(solver.info(), Eigen::Success);
  EXPECT_LT((m * x - b).norm(), 1.e-9);

  Eigen::ConjugateGradient<Eigen::MatrixXd> solver0(m);

  const Eigen::VectorXd x0 = solver0.solve(b);
  EXPECT_EQ(solver0.info(), Eigen::Success);
  EXPECT_LT(solver.iterations(), solver0.iterations());
}

static constexpr const std::size_t cNumRandomProblems = 20;
static constexpr const Eigen::Index cMinProblemSize = 100;
static constexpr const Eigen::Index cMaxProblemSize = 300;

TEST(PartialCholesky, PreconditionRandomProblems) {
  std::default_random_engine gen{cDefaultSeed};
  std::uniform_int_distribution<Eigen::Index> problem_size_dist(
      cMinProblemSize, cMaxProblemSize);
  std::size_t fail = 0;
  std::size_t condfail = 0;
  double reduction = 0;
  double dcond = 0;
  for (std::size_t i = 0; i < cNumRandomProblems; ++i) {
    const Eigen::Index problem_size{problem_size_dist(gen)};
    const auto m = random_covariance_matrix(problem_size, gen);
    const Eigen::VectorXd b{
        Eigen::VectorXd::NullaryExpr(problem_size, [&gen]() {
          return std::normal_distribution<double>(0, 1)(gen);
        })};

    Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower,
                             Eigen::PartialCholesky<double>>
        solver(m);
    EXPECT_EQ(solver.info(), Eigen::Success)
        << "Preconditioned solver failed to decompose";
    solver.preconditioner().setOrder(
        std::max(problem_size / 20, Eigen::Index{20}));

    const Eigen::VectorXd x = solver.solve(b);
    EXPECT_EQ(solver.info(), Eigen::Success)
        << "Preconditioned solver failed to solve";
    EXPECT_LT((m * x - b).norm(), 1.e-9);

    Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower,
                             Eigen::IdentityPreconditioner>
        solver0(m);
    EXPECT_EQ(solver0.info(), Eigen::Success)
        << "Default solver failed to decompose";
    const Eigen::VectorXd x0 = solver0.solve(b);
    EXPECT_EQ(solver0.info(), Eigen::Success)
        << "Default solver failed to solve";
    const auto cond0 = condition_number(m);
    EXPECT_LE(solver.iterations(), solver0.iterations())
        << "Problem size: " << problem_size << "; cond " << cond0;

    const auto condp = condition_number(solver.preconditioner().solve(m));

    EXPECT_LE(condp, cond0)
        << "Problem size: " << problem_size
        << "; cond: " << std::setprecision(5) << cond0
        << "; preconditioned: " << condp << "; damage: " << condp / cond0;

    // std::cout << i << ": cond " << cond0 << " -> " << condp << " = "
    //           << condp / cond0 << std::endl;
    if (solver.iterations() > solver0.iterations()) {
      fail++;
    }
    if (condp > cond0) {
      condfail++;
    }
    dcond += condp / cond0;
    reduction += static_cast<double>(solver.iterations()) /
                 static_cast<double>(solver0.iterations());
  }

  std::cout << "More iterations on " << fail << " of " << cNumRandomProblems
            << " problems." << std::endl;
  std::cout << "Worse conditioning on " << condfail << " of "
            << cNumRandomProblems << " problems." << std::endl;
  reduction = std::pow(reduction,
                       2 / static_cast<double>(cNumRandomProblems - fail + 1));
  std::cout << "Geom mean iteration reduction: " << reduction << std::endl;
  dcond = std::pow(dcond, 2 / static_cast<double>(cNumRandomProblems + 1));
  std::cout << "Geom mean condition reduction: " << dcond << std::endl;
}

} // namespace albatross