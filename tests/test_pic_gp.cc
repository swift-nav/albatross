/*
 * Copyright (C) 2024 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <fstream>
#include <gtest/gtest.h>

// #include "albatross/src/eigen/serializable_ldlt.hpp"
// #include "albatross/src/linalg/block_diagonal.hpp"
#include "test_models.h"
#include <albatross/Common>
#include <chrono>

#include <albatross/SparseGP>
#include <random>

namespace albatross {

struct LeaveOneIntervalOut {

  LeaveOneIntervalOut(){};
  explicit LeaveOneIntervalOut(double group_domain_size_)
      : group_domain_size(group_domain_size_){};

  int operator()(const double &f) const {
    return static_cast<int>(floor(f / group_domain_size));
  }

  int operator()(const Measurement<double> &f) const {
    return static_cast<int>(floor(f.value / group_domain_size));
  }

  double group_domain_size = 5.;
};

TEST(TestPicGP, TestPredictionExists) {
  const auto dataset = make_toy_linear_data();
  UniformlySpacedInducingPoints strategy(8);
  LeaveOneIntervalOut grouper;
  const auto pic =
      pic_gp_from_covariance(make_simple_covariance_function(), grouper,
                             strategy, "pic", DenseQRImplementation{});

  const auto pic_fit = pic.fit(dataset);

  const auto test_features = linspace(0.1, 9.9, 11);

  const auto pic_pred =
      pic_fit.predict_with_measurement_noise(test_features).joint();

  EXPECT_GT(pic_pred.mean.size(), 0);
}


TEST(TestPicGP, ScalarEquivalence) {
  static constexpr std::size_t kNumTrainPoints = 3;
  static constexpr std::size_t kNumTestPoints = 1;
  static constexpr std::size_t kNumInducingPoints = 3;

  UniformlySpacedInducingPoints strategy(kNumInducingPoints);
  // Set the grouper so that all the simple test data falls within one
  // group.
  LeaveOneIntervalOut grouper(10);
  auto direct = gp_from_covariance(make_simple_covariance_function(), "direct");
  auto pic = pic_gp_from_covariance(make_simple_covariance_function(), grouper,
                                    strategy, "pic", DenseQRImplementation{});

  auto dataset = make_toy_linear_data(5, 1, 0.1, kNumTrainPoints);

  auto direct_fit = direct.fit(dataset);
  auto pic_fit = pic.fit(dataset);

  auto test_features = linspace(0.1, 9.9, kNumTestPoints);
  auto direct_pred =
      direct_fit.predict_with_measurement_noise(test_features).joint();
  auto pic_pred = pic_fit.predict_with_measurement_noise(test_features).joint();
  EXPECT_LT(distance::wasserstein_2(direct_pred, pic_pred), 1e-12);
}

TEST(TestPicGP, SingleBlockEquivalence) {
  static constexpr std::size_t kNumTrainPoints = 10;
  static constexpr std::size_t kNumTestPoints = 10;
  static constexpr std::size_t kNumInducingPoints = 4;

  UniformlySpacedInducingPoints strategy(kNumInducingPoints);
  // Set the grouper so that all the simple test data falls within one
  // group.
  LeaveOneIntervalOut grouper(10);
  auto direct = gp_from_covariance(make_simple_covariance_function(), "direct");
  auto pic = pic_gp_from_covariance(make_simple_covariance_function(), grouper,
                                    strategy, "pic", DenseQRImplementation{});

  auto dataset = make_toy_linear_data(5, 1, 0.1, kNumTrainPoints);

  auto direct_fit = direct.fit(dataset);
  auto pic_fit = pic.fit(dataset);

  auto test_features = linspace(0.1, 9.9, kNumTestPoints);
  auto direct_pred =
      direct_fit.predict_with_measurement_noise(test_features).joint();
  auto pic_pred = pic_fit.predict_with_measurement_noise(test_features).joint();
  EXPECT_LT(distance::wasserstein_2(direct_pred, pic_pred), 1e-10);
}

template <typename CovarianceType, typename FeatureType,
          typename InducingStrategy, typename GrouperFunction>
JointDistribution test_pic(const RegressionDataset<FeatureType> &data,
                           const std::vector<FeatureType> &predict,
                           CovarianceType &&cov, InducingStrategy &&inducing,
                           GrouperFunction &&grouper) {
  const auto u = inducing(cov, data.features);

  auto bfp_cov = make_brute_force_pic_covariance(inducing(cov, data.features),
                                                 cov, grouper);

  BlockDiagonal K_ff;
  const auto indexers = group_by(data.features, grouper).indexers();
  std::vector<std::size_t> reordered_inds;
  for (const auto &pair : indexers) {
    reordered_inds.insert(reordered_inds.end(), pair.second.begin(),
                          pair.second.end());
    K_ff.blocks.emplace_back(cov(subset(data.features, pair.second)));
    K_ff.blocks.back().diagonal() +=
        subset(data.targets.covariance.diagonal(), pair.second);
  }
  const std::vector<FeatureType> ordered_features =
      subset(data.features, reordered_inds);

  Eigen::MatrixXd K_uu = cov(u, u);
  K_uu.diagonal() += Eigen::VectorXd::Constant(K_uu.rows(), 1e-8);
  const Eigen::MatrixXd K_fu = cov(ordered_features, u);

  const Eigen::SerializableLDLT K_uu_ldlt = K_uu.ldlt();
  const Eigen::MatrixXd P = K_uu_ldlt.sqrt_solve(K_fu.transpose());
  const Eigen::MatrixXd Q_ff = P.transpose() * P;
  BlockDiagonal Q_ff_diag;
  Eigen::Index i = 0;
  for (const auto &pair : indexers) {
    const Eigen::Index cols = cast::to_index(pair.second.size());
    auto P_cols = P.block(0, i, P.rows(), cols);
    Q_ff_diag.blocks.emplace_back(P_cols.transpose() * P_cols);
    i += cols;
  }
  // auto Lambda = K_ff - Q_ff_diag;

  Eigen::MatrixXd Q_ff_lambda = Q_ff;
  Eigen::Index k = 0;
  for (const auto &block : K_ff.blocks) {
    Q_ff_lambda.block(k, k, block.rows(), block.rows()) = block;
    Q_ff_lambda.block(k, k, block.rows(), block.rows()).diagonal() +=
        Eigen::VectorXd::Constant(block.rows(), 1e-2);
    k += block.rows();
  }

  const auto print_matrix = [](const Eigen::MatrixXd &m, std::string &&label) {
    std::cout << label << " (" << m.rows() << "x" << m.cols() << "):\n"
              << m.format(Eigen::FullPrecision) << std::endl;
  };

  const auto print_vector = [](const Eigen::VectorXd &v, std::string &&label) {
    std::cout << label << " (" << v.size()
              << "): " << v.transpose().format(Eigen::FullPrecision)
              << std::endl;
  };

  print_matrix(Q_ff_lambda, "Q_ff + Lambda");
  // const BlockDiagonalLDLT Lambda_ldlt = Lambda.ldlt();

  print_matrix(bfp_cov(ordered_features), "BFP: Q_ff + Lambda");

  print_matrix(Q_ff_lambda - bfp_cov(ordered_features),
               "Q_ff_lambda difference");

  const Eigen::SerializableLDLT S = Q_ff_lambda.ldlt();

  print_matrix(S.matrixL(), "PITC L");
  print_vector(S.vectorD(), "PITC D");

  Eigen::MatrixXd K_PIC(ordered_features.size(), predict.size());
  Eigen::MatrixXd K_PITC(ordered_features.size(), predict.size());

  Eigen::MatrixXd V_PIC =
      Eigen::MatrixXd::Zero(ordered_features.size(), predict.size());

  Eigen::MatrixXd Y = K_uu_ldlt.solve(K_fu.transpose());
  print_matrix(Y, "Y");

  Eigen::MatrixXd W = K_uu_ldlt.solve(S.solve(K_fu).transpose());
  print_matrix(W, "W");

  for (Eigen::Index f = 0; f < cast::to_index(ordered_features.size()); ++f) {
    for (Eigen::Index p = 0; p < cast::to_index(predict.size()); ++p) {
      const std::vector<FeatureType> pv{predict[p]};
      K_PITC(f, p) = K_fu.row(f).dot(K_uu_ldlt.solve(cov(u, pv)).col(0));
      if (grouper(predict[p]) == grouper(ordered_features[f])) {
        K_PIC(f, p) = cov(ordered_features[f], predict[p]);
        V_PIC(f, p) =
            K_PIC(f, p) - K_fu.row(f).dot(K_uu_ldlt.solve(cov(u, pv)).col(0));
      } else {
        K_PIC(f, p) = K_fu.row(f).dot(K_uu_ldlt.solve(cov(u, pv)).col(0));
      }
    }
  }

  const Eigen::MatrixXd K_pu = cov(predict, u);
  // Eigen::Index j = 0;
  // Eigen::Index blk = 0;
  // std::vector<Eigen::MatrixXd> WW;

  const Eigen::MatrixXd WBW = K_uu_ldlt.solve(S.solve(K_fu).transpose());
  print_matrix(K_PIC, "K_PIC");
  print_matrix(bfp_cov(ordered_features, predict), "BFP: K_PIC");

  print_matrix(K_PIC - bfp_cov(ordered_features, predict), "K_PIC error");
  print_matrix(K_PITC, "K_PITC");

  print_matrix(V_PIC, "V_PIC");

  print_matrix(WBW, "W");
  const Eigen::MatrixXd U = K_pu * WBW * V_PIC;
  print_matrix(U, "U");

  auto SV = S.sqrt_solve(V_PIC);
  const Eigen::MatrixXd VSV =
      V_PIC.transpose() * S.solve(V_PIC); // SV.transpose() * SV;
  print_matrix(VSV, "VSV");

  const Eigen::MatrixXd predict_prior = cov(predict);

  print_matrix(predict_prior, "prior");

  print_matrix(bfp_cov(predict), "BFP: prior");

  print_matrix(predict_prior - bfp_cov(predict), "prior error");

  // auto KK = K_PITC_ldlt.sqrt_solve(K_PIC);
  // const Eigen::MatrixXd explained_cov = KK.transpose() * KK;
  const Eigen::MatrixXd explained_cov = K_PIC.transpose() * S.solve(K_PIC);
  print_matrix(explained_cov, "explained");

  const Eigen::VectorXd PIC_mean =
      K_PIC.transpose() * S.solve(data.targets.mean);
  print_vector(PIC_mean, "PIC mean");
  const Eigen::MatrixXd predict_cov = predict_prior - explained_cov;

  print_matrix(predict_cov, "K_**");

  const Eigen::MatrixXd PITC_cov =
      predict_prior - K_PITC.transpose() * S.solve(K_PITC);

  const Eigen::MatrixXd PIC_cov = PITC_cov - U - U.transpose() - VSV;
  print_matrix(PITC_cov, "PTIC cov");
  print_matrix(PIC_cov, "PIC cov");

  print_matrix(predict_cov - PITC_cov, "PIC - PITC");
  print_matrix(predict_cov - PIC_cov, "PIC - factored");

  return JointDistribution{PIC_mean, PIC_cov};
}

TEST(TestPicGP, BruteForceEquivalenceOneBlock) {
  static constexpr std::size_t kNumTrainPoints = 10;
  static constexpr std::size_t kNumTestPoints = 10;
  static constexpr std::size_t kNumInducingPoints = 5;

  UniformlySpacedInducingPoints strategy(kNumInducingPoints);
  LeaveOneIntervalOut grouper(10);
  auto pic = pic_gp_from_covariance(make_simple_covariance_function(), grouper,
                                    strategy, "pic", DenseQRImplementation{});

  auto dataset = make_toy_linear_data(5, 1, 0.1, kNumTrainPoints);
  auto bfp_cov = make_brute_force_pic_covariance(
      strategy(make_simple_covariance_function(), dataset.features),
      make_simple_covariance_function(), grouper);
  auto bfp = gp_from_covariance(bfp_cov);

  auto bfp_fit = bfp.fit(dataset);
  auto pic_fit = pic.fit(dataset);

  auto test_features = linspace(0.1, 9.9, kNumTestPoints);
  auto bfp_pred = bfp_fit.predict_with_measurement_noise(test_features).joint();
  auto pic_pred = pic_fit.predict_with_measurement_noise(test_features).joint();
  EXPECT_LT(distance::wasserstein_2(bfp_pred, pic_pred), 1e-11);
}

TEST(TestPicGP, BruteForceEquivalenceMultipleBlocks) {
  static constexpr std::size_t kNumTrainPoints = 10;
  static constexpr std::size_t kNumTestPoints = 10;
  static constexpr std::size_t kNumInducingPoints = 5;

  UniformlySpacedInducingPoints strategy(kNumInducingPoints);
  LeaveOneIntervalOut grouper(2);
  auto pic = pic_gp_from_covariance(make_simple_covariance_function(), grouper,
                                    strategy, "pic", DenseQRImplementation{});

  auto dataset = make_toy_linear_data(5, 1, 0.1, kNumTrainPoints);
  auto bfp_cov = make_brute_force_pic_covariance(
      strategy(make_simple_covariance_function(), dataset.features),
      make_simple_covariance_function(), grouper);
  auto bfp = gp_from_covariance(bfp_cov);

  auto bfp_fit = bfp.fit(dataset);
  auto pic_fit = pic.fit(dataset);

  auto test_features = linspace(0.1, 9.9, kNumTestPoints);
  auto bfp_pred = bfp_fit.predict_with_measurement_noise(test_features).joint();
  auto pic_pred = pic_fit.predict_with_measurement_noise(test_features).joint();

  EXPECT_LT(distance::wasserstein_2(bfp_pred, pic_pred), 1e-12);
}

TEST(TestPicGP, PITCEquivalenceOutOfTraining) {
  static constexpr std::size_t kNumTrainPoints = 10;
  static constexpr std::size_t kNumTestPoints = 10;
  static constexpr std::size_t kNumInducingPoints = 5;

  UniformlySpacedInducingPoints strategy(kNumInducingPoints);
  LeaveOneIntervalOut grouper(2);
  auto pic = pic_gp_from_covariance(make_simple_covariance_function(), grouper,
                                    strategy, "pic", DenseQRImplementation{});

  auto dataset = make_toy_linear_data(5, 1, 0.1, kNumTrainPoints);
  auto pitc =
      sparse_gp_from_covariance(make_simple_covariance_function(), grouper,
                                strategy, "pitc", DenseQRImplementation{});

  auto pic_fit = pic.fit(dataset);
  auto pitc_fit = pitc.fit(dataset);

  auto test_features = linspace(10.1, 19.9, kNumTestPoints);
  auto pic_pred = pic_fit.predict_with_measurement_noise(test_features).joint();
  auto pitc_pred = pitc_fit.predict_with_measurement_noise(test_features).joint();

  EXPECT_LT(distance::wasserstein_2(pic_pred, pitc_pred), 1e-12);
}

TEST(TestPicGP, PredictMeanEquivalent) {
  static constexpr std::size_t kNumTrainPoints = 10;
  static constexpr std::size_t kNumTestPoints = 10;
  static constexpr std::size_t kNumInducingPoints = 5;

  UniformlySpacedInducingPoints strategy(kNumInducingPoints);
  LeaveOneIntervalOut grouper(2);
  auto pic = pic_gp_from_covariance(make_simple_covariance_function(), grouper,
                                    strategy, "pic", DenseQRImplementation{});

  auto dataset = make_toy_linear_data(5, 1, 0.1, kNumTrainPoints);

  auto pic_fit = pic.fit(dataset);

  auto test_features = linspace(0.1, 9.9, kNumTestPoints);
  auto pic_pred = pic_fit.predict_with_measurement_noise(test_features).mean();
  auto pic_joint_pred = pic_fit.predict_with_measurement_noise(test_features).joint();

  const double pic_mean_error = (pic_pred - pic_joint_pred.mean).norm();
  EXPECT_LT(pic_mean_error, 1e-12);
}

TEST(TestPicGP, PredictMarginalEquivalent) {
  static constexpr std::size_t kNumTrainPoints = 10;
  static constexpr std::size_t kNumTestPoints = 10;
  static constexpr std::size_t kNumInducingPoints = 5;

  UniformlySpacedInducingPoints strategy(kNumInducingPoints);
  LeaveOneIntervalOut grouper(2);
  auto pic = pic_gp_from_covariance(make_simple_covariance_function(), grouper,
                                    strategy, "pic", DenseQRImplementation{});

  auto dataset = make_toy_linear_data(5, 1, 0.1, kNumTrainPoints);

  auto pic_fit = pic.fit(dataset);

  auto test_features = linspace(0.1, 9.9, kNumTestPoints);
  auto pic_pred = pic_fit.predict_with_measurement_noise(test_features).marginal();
  auto pic_joint_pred = pic_fit.predict_with_measurement_noise(test_features).joint();

  const double pic_marginal_error =
      (pic_pred.mean - pic_joint_pred.mean).norm();
  EXPECT_LT(pic_marginal_error, 1e-12);
  const double pic_marginal_cov_error =
      (Eigen::VectorXd(pic_pred.covariance.diagonal()) -
       Eigen::VectorXd(pic_joint_pred.covariance.diagonal()))
          .norm();
  EXPECT_LT(pic_marginal_cov_error, 1e-12)
      << "\nmarginal: "
      << Eigen::VectorXd(pic_pred.covariance)
             .transpose()
             .format(Eigen::FullPrecision)
      << "\njoint diag: "
      << Eigen::VectorXd(pic_joint_pred.covariance.diagonal())
             .transpose()
             .format(Eigen::FullPrecision);
}


} // namespace albatross
