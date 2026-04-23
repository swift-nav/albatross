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

#include <albatross/CholmodSupport>
#include <albatross/Indexing>
#include <albatross/Ransac>
#include <albatross/serialize/GP>
#include <albatross/utils/RandomUtils>

#include <gtest/gtest.h>

#include "test_models.h"
#include "test_utils.h"

namespace albatross {

using SparseDouble = Eigen::SparseMatrix<double>;
using SupernodalLLT = Eigen::SerializableCholmodSupernodalLLT<SparseDouble>;
using SimplicialLLT = Eigen::SerializableCholmodSimplicialLLT<SparseDouble>;
using SimplicialLDLT = Eigen::SerializableCholmodSimplicialLDLT<SparseDouble>;

template <typename Solver>
class CholmodCovarianceTest : public ::testing::Test {
public:
  CholmodCovarianceTest() : cov(), rhs() {
    const Eigen::Index n = 8;
    std::default_random_engine gen(1234);
    cov = random_covariance_matrix(n, gen);
    rhs = Eigen::MatrixXd::Random(n, 3);
  }
  Eigen::MatrixXd cov;
  Eigen::MatrixXd rhs;
};

using CholmodSolvers =
    ::testing::Types<SupernodalLLT, SimplicialLLT, SimplicialLDLT>;
TYPED_TEST_SUITE(CholmodCovarianceTest, CholmodSolvers);

TYPED_TEST(CholmodCovarianceTest, test_solve_matches_dense) {
  const CholmodCovariance<TypeParam> rep(this->cov);
  const Eigen::SerializableLDLT dense(this->cov);
  EXPECT_LE((rep.solve(this->rhs) - dense.solve(this->rhs)).norm(), 1e-10);
}

TYPED_TEST(CholmodCovarianceTest, test_log_determinant_matches_dense) {
  const CholmodCovariance<TypeParam> rep(this->cov);
  const Eigen::SerializableLDLT dense(this->cov);
  EXPECT_NEAR(rep.log_determinant(), dense.log_determinant(), 1e-10);
}

// The square root A^{1/2} is not unique (orthogonal rotation ambiguity), so
// compare via the quadratic form b^T A^{-1} b which IS unique.
TYPED_TEST(CholmodCovarianceTest, test_sqrt_solve_quadratic_form) {
  const CholmodCovariance<TypeParam> rep(this->cov);
  const Eigen::MatrixXd sqrt_b = rep.sqrt_solve(this->rhs);
  const Eigen::MatrixXd quad_sqrt = sqrt_b.transpose() * sqrt_b;
  const Eigen::MatrixXd quad_direct =
      this->rhs.transpose() * rep.solve(this->rhs);
  EXPECT_LE((quad_sqrt - quad_direct).norm(), 1e-10);
}

// With the SerializableLDLT convention S = D^{-1/2} L^{-1} P, sqrt_solve
// applies S and sqrt_transpose_solve applies S^T, with S^T S = A^{-1}.
// Composing in order -- first sqrt_solve, then sqrt_transpose_solve --
// yields exactly A^{-1} b.  (The other order gives S S^T, which is NOT
// A^{-1}.)
TYPED_TEST(CholmodCovarianceTest, test_sqrt_transpose_solve_composes) {
  const CholmodCovariance<TypeParam> rep(this->cov);
  const Eigen::MatrixXd sqrt_b = rep.sqrt_solve(this->rhs);
  const Eigen::MatrixXd composed = rep.sqrt_transpose_solve(sqrt_b);
  const Eigen::MatrixXd direct = rep.solve(this->rhs);
  EXPECT_LE((composed - direct).norm(), 1e-10);
}

// sqrt_solve and sqrt_transpose_solve are dual decompositions of A^{-1},
// so sqrt_transpose_solve(sqrt_solve(x)^T)^T should reconstruct A^{-1} x
// only up to the A^{1/2} ambiguity -- but the composition
// sqrt_solve(A) -> then take (sqrt_solve)^T * (sqrt_solve) must equal
// A^{-1}.  Validate that on a random full matrix.
TYPED_TEST(CholmodCovarianceTest, test_sqrt_solve_reconstructs_inverse) {
  const CholmodCovariance<TypeParam> rep(this->cov);
  const Eigen::MatrixXd n_by_n =
      Eigen::MatrixXd::Identity(this->cov.rows(), this->cov.cols());
  const Eigen::MatrixXd sqrt_I = rep.sqrt_solve(n_by_n);
  const Eigen::MatrixXd A_inv_reconstructed = sqrt_I.transpose() * sqrt_I;
  const Eigen::MatrixXd A_inv_direct =
      rep.solve(Eigen::MatrixXd::Identity(this->cov.rows(), this->cov.cols()));
  EXPECT_LE((A_inv_reconstructed - A_inv_direct).norm(), 1e-8);
}

TYPED_TEST(CholmodCovarianceTest, test_inverse_diagonal_matches_dense) {
  const CholmodCovariance<TypeParam> rep(this->cov);
  const Eigen::SerializableLDLT dense(this->cov);
  const Eigen::VectorXd cholmod_diag = rep.inverse_diagonal();
  const Eigen::VectorXd dense_diag = dense.inverse_diagonal();
  EXPECT_LE((cholmod_diag - dense_diag).norm(), 1e-10);
}

TYPED_TEST(CholmodCovarianceTest, test_inverse_blocks_matches_dense) {
  const CholmodCovariance<TypeParam> rep(this->cov);
  const Eigen::SerializableLDLT dense(this->cov);
  // A mix of contiguous, singleton, and multi-index blocks.
  const std::vector<std::vector<std::size_t>> blocks = {
      {0, 1}, {2}, {3, 4, 5}, {6, 7}};
  const auto cholmod_blocks = rep.inverse_blocks(blocks);
  const auto dense_blocks = dense.inverse_blocks(blocks);
  ASSERT_EQ(cholmod_blocks.size(), dense_blocks.size());
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    EXPECT_LE((cholmod_blocks[i] - dense_blocks[i]).norm(), 1e-10)
        << "block " << i;
  }
}

TYPED_TEST(CholmodCovarianceTest, test_is_positive_definite) {
  const CholmodCovariance<TypeParam> rep(this->cov);
  EXPECT_TRUE(rep.is_positive_definite());
}

TYPED_TEST(CholmodCovarianceTest, test_rows_cols) {
  const CholmodCovariance<TypeParam> rep(this->cov);
  EXPECT_EQ(rep.rows(), this->cov.rows());
  EXPECT_EQ(rep.cols(), this->cov.cols());
}

// --- Dense-GP prediction parity via manual Fit<GPFit<...>> construction.
// This exercises the CovarianceRepresentation duck-typed interface through
// the same gp_marginal_prediction / gp_joint_prediction helpers the dense
// GP code uses.
TYPED_TEST(CholmodCovarianceTest, test_gp_marginal_prediction_matches_dense) {
  const Eigen::Index n_train = this->cov.rows();
  MarginalDistribution targets(Eigen::VectorXd::Random(n_train));
  targets.covariance = Eigen::VectorXd::Constant(n_train, 1e-6).asDiagonal();

  std::vector<double> train_features(cast::to_size(n_train));
  for (Eigen::Index i = 0; i < n_train; ++i) {
    train_features[cast::to_size(i)] = cast::to_double(i);
  }

  using CholFit = Fit<GPFit<CholmodCovariance<TypeParam>, double>>;
  using DenseFit = Fit<GPFit<Eigen::SerializableLDLT, double>>;
  const CholFit chol_fit(train_features, this->cov, targets);
  const DenseFit dense_fit(train_features, this->cov, targets);

  // Build a cross-covariance to pretend-predict with.
  const Eigen::Index n_test = 4;
  const Eigen::MatrixXd cross = Eigen::MatrixXd::Random(n_train, n_test);
  const Eigen::VectorXd prior_var = Eigen::VectorXd::Ones(n_test);
  const Eigen::MatrixXd prior_cov =
      Eigen::MatrixXd::Identity(n_test, n_test) +
      Eigen::MatrixXd::Constant(n_test, n_test, 0.1);

  const auto chol_marginal = gp_marginal_prediction(
      cross, prior_var, chol_fit.information, chol_fit.train_covariance);
  const auto dense_marginal = gp_marginal_prediction(
      cross, prior_var, dense_fit.information, dense_fit.train_covariance);

  EXPECT_LE((chol_marginal.mean - dense_marginal.mean).norm(), 1e-10);
  EXPECT_LE((Eigen::VectorXd(chol_marginal.covariance.diagonal()) -
             Eigen::VectorXd(dense_marginal.covariance.diagonal()))
                .norm(),
            1e-10);

  const auto chol_joint = gp_joint_prediction(
      cross, prior_cov, chol_fit.information, chol_fit.train_covariance);
  const auto dense_joint = gp_joint_prediction(
      cross, prior_cov, dense_fit.information, dense_fit.train_covariance);

  EXPECT_LE((chol_joint.mean - dense_joint.mean).norm(), 1e-10);
  EXPECT_LE((chol_joint.covariance - dense_joint.covariance).norm(), 1e-8);
}

// Sparsity smoke test: give the representation a dense MatrixXd that
// happens to have a tridiagonal structure (many exact zeros) and verify
// that the kept SparseMatrix<double> has the expected sparsity.  This is
// the mechanism by which compactly-supported covariance kernels will get
// real speedups out of the Cholmod backend.
TEST(CholmodCovarianceSparsity, preserves_exact_zeros) {
  const Eigen::Index n = 20;
  Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(n, n);
  for (Eigen::Index i = 0; i < n; ++i) {
    dense(i, i) = 4.;
    if (i > 0) {
      dense(i, i - 1) = dense(i - 1, i) = 1.;
    }
  }
  // 3n - 2 nonzeros in a tridiagonal n x n matrix.
  const auto expected_nnz = 3 * n - 2;
  const CholmodCovariance<SupernodalLLT> rep(dense);
  EXPECT_EQ(rep.sparse_A_.nonZeros(), expected_nnz);
  EXPECT_LT(rep.sparse_A_.nonZeros(), n * n);
}

// End-to-end serialization of a Fit<GPFit<CholmodCovariance<...>, double>>.
// Round-trips through JSON and PortableBinary and verifies predictions
// stay identical before and after.
template <typename Solver>
class CholmodCovarianceFitSerializeTest : public ::testing::Test {};
TYPED_TEST_SUITE(CholmodCovarianceFitSerializeTest, CholmodSolvers);

TYPED_TEST(CholmodCovarianceFitSerializeTest, test_fit_roundtrip_json) {
  const Eigen::Index n = 6;
  std::default_random_engine gen(9999);
  const Eigen::MatrixXd cov = random_covariance_matrix(n, gen);
  MarginalDistribution targets(Eigen::VectorXd::Random(n));
  targets.covariance = Eigen::VectorXd::Constant(n, 1e-8).asDiagonal();
  std::vector<double> features(cast::to_size(n));
  for (Eigen::Index i = 0; i < n; ++i) {
    features[cast::to_size(i)] = cast::to_double(i);
  }

  using FitType = Fit<GPFit<CholmodCovariance<TypeParam>, double>>;
  const FitType fit_in(features, cov, targets);

  std::ostringstream os;
  {
    cereal::JSONOutputArchive oarchive(os);
    oarchive(fit_in);
  }
  FitType fit_out;
  {
    std::istringstream is(os.str());
    cereal::JSONInputArchive iarchive(is);
    iarchive(fit_out);
  }

  EXPECT_EQ(fit_in.train_features, fit_out.train_features);
  EXPECT_EQ(fit_in.information, fit_out.information);

  const Eigen::VectorXd probe = Eigen::VectorXd::Ones(n);
  EXPECT_LE((fit_in.train_covariance.solve(probe) -
             fit_out.train_covariance.solve(probe))
                .norm(),
            1e-12);
}

// --- CV / RANSAC parity vs the dense SerializableLDLT-backed GP. --------
//
// CholmodGaussianProcess mirrors the AdaptedGaussianProcess pattern in
// test_models.h:186-229: override _fit_impl to return a Fit whose
// CovarianceRepresentation is CholmodCovariance<Solver>; inherit
// _predict_impl and route cross_validated_predictions through the generic
// gp_cross_validated_predictions helper.  Combined with the
// cross_validation_utils.hpp change that templatizes held_out_predictions
// on the covariance type, this plugs the Cholmod backend into the
// public cross_validate() and ransac() APIs.
template <typename CovFunc, typename Solver>
class CholmodGaussianProcess
    : public GaussianProcessBase<CovFunc, ZeroMean,
                                 CholmodGaussianProcess<CovFunc, Solver>> {
public:
  using Base = GaussianProcessBase<CovFunc, ZeroMean,
                                   CholmodGaussianProcess<CovFunc, Solver>>;
  using Base::_predict_impl;
  using Base::Base;

  template <typename FeatureType>
  auto _fit_impl(const std::vector<FeatureType> &features,
                 const MarginalDistribution &targets) const {
    const auto measurement_features = as_measurements(features);
    Eigen::MatrixXd cov =
        this->covariance_function_(measurement_features, Base::threads_.get());
    MarginalDistribution zero_mean_targets(targets);
    this->mean_function_.remove_from(measurement_features,
                                     &zero_mean_targets.mean);
    using FitType = Fit<GPFit<CholmodCovariance<Solver>, FeatureType>>;
    return FitType(features, cov, zero_mean_targets);
  }

  template <typename FeatureType, typename PredictType, typename GroupKey>
  std::map<GroupKey, PredictType>
  cross_validated_predictions(const RegressionDataset<FeatureType> &dataset,
                              const GroupIndexer<GroupKey> &group_indexer,
                              PredictTypeIdentity<PredictType> identity) const {
    return gp_cross_validated_predictions(dataset, group_indexer, *this,
                                          identity);
  }
};

template <typename Solver, typename CovFunc>
auto cholmod_gp_from_covariance(CovFunc &&covariance) {
  return CholmodGaussianProcess<std::decay_t<CovFunc>, Solver>(
      std::forward<CovFunc>(covariance));
}

// Typed test fixture: one instantiation per Cholmod solver variant.
template <typename Solver>
class CholmodCVRansacTest : public ::testing::Test {};
TYPED_TEST_SUITE(CholmodCVRansacTest, CholmodSolvers);

namespace {

// Tight tolerance on the means; slightly looser on the marginal
// variances / covariances because they involve an additional solve on
// the group-block level.  For the identical dense inputs and solves we
// hit ~1e-10 comfortably.
constexpr double kCVMeanTol = 1e-10;
constexpr double kCVVarianceTol = 1e-10;

template <typename Pred1, typename Pred2>
void expect_groupwise_marginal_match(const Pred1 &a, const Pred2 &b,
                                     double mean_tol, double var_tol) {
  ASSERT_EQ(a.size(), b.size());
  for (const auto &[key, a_pred] : a) {
    const auto it = b.find(key);
    ASSERT_TRUE(it != b.end()) << "missing group key in second prediction";
    const auto &b_pred = it->second;
    EXPECT_LE((a_pred.mean - b_pred.mean).norm(), mean_tol) << "group " << key;
    EXPECT_LE((Eigen::VectorXd(a_pred.covariance.diagonal()) -
               Eigen::VectorXd(b_pred.covariance.diagonal()))
                  .norm(),
              var_tol)
        << "group " << key;
  }
}

} // namespace

TYPED_TEST(CholmodCVRansacTest, test_loo_cv_matches_dense) {
  const auto dataset = make_toy_linear_data();
  const auto covariance = make_simple_covariance_function();

  const auto dense_gp = gp_from_covariance(covariance);
  const auto cholmod_gp = cholmod_gp_from_covariance<TypeParam>(covariance);

  LeaveOneOutGrouper loo;
  const auto dense_cv =
      dense_gp.cross_validate().predict(dataset, loo).marginals();
  const auto cholmod_cv =
      cholmod_gp.cross_validate().predict(dataset, loo).marginals();

  expect_groupwise_marginal_match(dense_cv, cholmod_cv, kCVMeanTol,
                                  kCVVarianceTol);
}

TYPED_TEST(CholmodCVRansacTest, test_logo_cv_matches_dense) {
  const auto dataset = make_toy_linear_data();
  const auto covariance = make_simple_covariance_function();

  const auto dense_gp = gp_from_covariance(covariance);
  const auto cholmod_gp = cholmod_gp_from_covariance<TypeParam>(covariance);

  // Two-groups: even- vs odd-indexed feature values.  Plenty for LOGO to
  // exercise the inverse_blocks path.
  const auto grouper = [](const double &x) { return static_cast<int>(x) % 2; };

  const auto dense_cv =
      dense_gp.cross_validate().predict(dataset, grouper).marginals();
  const auto cholmod_cv =
      cholmod_gp.cross_validate().predict(dataset, grouper).marginals();

  expect_groupwise_marginal_match(dense_cv, cholmod_cv, kCVMeanTol,
                                  kCVVarianceTol);
}

TYPED_TEST(CholmodCVRansacTest, test_ransac_matches_dense) {
  // Inject two outliers (matching test_ransac.cc:59-109's pattern) so
  // RANSAC has something to reject.
  auto dataset = make_toy_linear_data();
  const std::vector<std::size_t> bad_inds = {3, 5};
  for (const auto &i : bad_inds) {
    dataset.targets.mean[cast::to_index(i)] = std::pow(-1, i) * 400.;
  }

  const auto covariance = make_simple_covariance_function();

  RansacConfig config;
  config.inlier_threshold = 1.;
  config.random_sample_size = 3;
  config.min_consensus_size = 3;
  config.max_iterations = 20;
  config.max_failed_candidates = 20;

  DefaultGPRansacStrategy ransac_strategy;

  // Same random seed consumed by both RANSACs: they'll sample the same
  // candidates and converge on the same consensus set, so the resulting
  // fits are on identical inlier subsets and their predictions should
  // agree to within the dense/cholmod solve tolerance.
  const auto dense_ransac =
      gp_from_covariance(covariance).ransac(ransac_strategy, config);
  const auto cholmod_ransac = cholmod_gp_from_covariance<TypeParam>(covariance)
                                  .ransac(ransac_strategy, config);

  const auto dense_fit = dense_ransac.fit(dataset);
  const auto cholmod_fit = cholmod_ransac.fit(dataset);

  const auto dense_pred = dense_fit.predict(dataset.features).marginal();
  const auto cholmod_pred = cholmod_fit.predict(dataset.features).marginal();

  EXPECT_LE((dense_pred.mean - cholmod_pred.mean).norm(), 1e-8);
  EXPECT_LE((Eigen::VectorXd(dense_pred.covariance.diagonal()) -
             Eigen::VectorXd(cholmod_pred.covariance.diagonal()))
                .norm(),
            1e-8);
}

} // namespace albatross
