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
/*
static constexpr std::size_t kNumRandomFits = 1000;

static constexpr std::size_t kMaxSize = 20000;

TEST(TestPicGP, ComparePrediction) {
  std::default_random_engine gen(22);
  std::uniform_int_distribution<std::size_t> data_sizes{1, kMaxSize};
  std::uniform_int_distribution<std::size_t> predict_sizes{1, 200};
  //   std::uniform_real_distribution<double> group_domain_sizes{1.e-3, 7.};

  auto workers = make_shared_thread_pool(8);

  std::ofstream csv_out(
      "/home/peddie/orion-engine/third_party/albatross/times.csv");

  csv_out << "idx,n_train,n_inducing,n_blocks,blocks_used,n_predict,dense_fit_"
             "time_ns,dense_pred_"
             "time_ns,pic_fit_time_ns,pic_pred_time_ns,mean_err_norm,cov_err_"
             "norm\n";

  for (std::size_t i = 0; i < kNumRandomFits; ++i) {
    // static constexpr double kLargestTrainPoint =
    //   static_cast<double>(kNumTrainPoints) - 1.;
    // static constexpr double kSpatialBuffer = 0;
    // static constexpr std::size_t kNumBlocks = 5;

    // UniformlySpacedInducingPoints strategy(kNumInducingPoints);
    // LeaveOneIntervalOut grouper(
    //     kLargestTrainPoint / static_cast<double>(kNumBlocks) + 1e-8);
    const std::size_t n_train = data_sizes(gen);
    const auto n_train_fraction = [&n_train](double div) {
      return std::max(std::size_t{1}, static_cast<std::size_t>(ceil(
                                          static_cast<double>(n_train) / div)));
    };
    std::uniform_int_distribution<std::size_t> inducing_sizes{
        n_train_fraction(30.), n_train_fraction(5.)};
    const std::size_t n_inducing_points = inducing_sizes(gen);
    const double largest_training_point = static_cast<double>(n_train) - 1.;
    std::uniform_int_distribution<std::size_t> num_blocks_dist{2, 50};
    const std::size_t n_blocks = num_blocks_dist(gen);

    const std::size_t n_predict = predict_sizes(gen);

    UniformlySpacedInducingPoints strategy(n_inducing_points);
    LeaveOneIntervalOut grouper(largest_training_point /
                                static_cast<double>(n_blocks));
    auto direct =
        gp_from_covariance(make_simple_covariance_function(), "direct");
    auto pic =
        pic_gp_from_covariance(make_simple_covariance_function(), grouper,
                               strategy, "pic", DenseQRImplementation{});
    pic.set_thread_pool(workers);

    auto dataset = make_toy_linear_data(5, 1, 0.1, n_train);
    const auto begin_direct_fit = std::chrono::steady_clock::now();
    auto direct_fit = direct.fit(dataset);
    const auto end_direct_fit = std::chrono::steady_clock::now();
    const auto begin_pic_fit = std::chrono::steady_clock::now();
    auto pic_fit = pic.fit(dataset);
    const auto end_pic_fit = std::chrono::steady_clock::now();

    std::set<
        std::result_of_t<LeaveOneIntervalOut(decltype(dataset.features[0]))>>
        train_indices;
    std::transform(dataset.features.begin(), dataset.features.end(),
                   std::inserter(train_indices, train_indices.begin()),
                   grouper);
    auto test_features = linspace(0, largest_training_point, n_predict);
    // const std::size_t test_features_before = test_features.size();
    test_features.erase(
        std::remove_if(test_features.begin(), test_features.end(),
                       [&train_indices, &grouper](const auto &f) {
                         // std::cout << "f: " << f << "; group: "
                         // << grouper(f)
                         //           << std::endl;
                         return train_indices.count(grouper(f)) == 0;
                       }),
        test_features.end());

    if (test_features.empty()) {
      continue;
    }

    // const std::size_t test_features_after = test_features.size();

    // std::cout << "train_indices (" << train_indices.size() << "): ";
    // for (const auto &index : train_indices) {
    //   std::cout << index << ", ";
    // }
    // std::cout << std::endl;

    // std::cout << "test_features, " << test_features_before << " -> "
    //           << test_features_after << ": ";
    // for (const auto &feature : test_features) {
    //   std::cout << feature << ", ";
    // }
    // std::cout << std::endl;
    const auto begin_direct_pred = std::chrono::steady_clock::now();
    auto direct_pred =
        direct_fit.predict_with_measurement_noise(test_features).joint();
    const auto end_direct_pred = std::chrono::steady_clock::now();
    auto pic_pred =
        pic_fit.predict_with_measurement_noise(test_features).joint();
    const auto end_pic_pred = std::chrono::steady_clock::now();

    csv_out << i << ',' << n_train << ',' << n_inducing_points << ','
            << n_blocks << ',' << train_indices.size() << ',' << n_predict
            << ','
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                   end_direct_fit - begin_direct_fit)
                   .count()
            << ','
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                   end_direct_pred - begin_direct_pred)
                   .count()
            << ','
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                   end_pic_fit - begin_pic_fit)
                   .count()
            << ','
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                   end_pic_pred - end_direct_pred)
                   .count()
            << ',' << (pic_pred.mean - direct_pred.mean).norm() << ','
            << (pic_pred.covariance - direct_pred.covariance).norm() << '\n';

    const double pic_error = (pic_pred.mean - direct_pred.mean).norm();
    EXPECT_LT(pic_error, 1e-4);
    // << "|u|: " << n_inducing_points << "; |f|: " << dataset.size()
    // << "; |p|: " << test_features.size()
    // << "; B width: " << grouper.group_domain_size
    // << "; n_B: " << train_indices.size();

    // const double pic_cov_error =
    //     (pic_pred.covariance - direct_pred.covariance).norm();
    // EXPECT_LT(pic_cov_error, 1e-6)
    //     << "|u|: " << n_inducing_points << "; |f|: " << dataset.size()
    //     << "; |p|: " << test_features.size()
    //     << "; B width: " << grouper.group_domain_size
    //     << "; n_B: " << train_indices.size();

    if (i % 10 == 0) {
      csv_out.flush();
      std::cout << i << std::endl;
    }
  }
  csv_out << std::endl;
  csv_out.close();
}
*/
template <typename CovarianceType, typename InducingFeatureType,
          typename GrouperFunction>
class BruteForcePIC
    : public CovarianceFunction<
          BruteForcePIC<CovarianceType, InducingFeatureType, GrouperFunction>> {
public:
  CovarianceType cov_;
  std::vector<InducingFeatureType> inducing_points_;
  GrouperFunction grouper_;
  Eigen::LDLT<Eigen::MatrixXd> K_uu_ldlt_;

  BruteForcePIC(const std::vector<InducingFeatureType> &inducing_points,
                CovarianceType &&cov, GrouperFunction &&grouper)
      : cov_{cov}, inducing_points_{inducing_points}, grouper_{grouper},
        K_uu_ldlt_{cov_(inducing_points_, inducing_points_).ldlt()} {}

  template <typename X> double _call_impl(const X &x, const X &y) const {
    if (grouper_(x) == grouper_(y)) {
      return cov_(x, y);
    }

    Eigen::VectorXd K_xu(inducing_points_.size());
    Eigen::VectorXd K_uy(inducing_points_.size());
    for (Eigen::Index i = 0;
         i < static_cast<Eigen::Index>(inducing_points_.size()); ++i) {
      K_xu[i] = cov_(x, inducing_points_[cast::to_size(i)]);
      K_uy[i] = cov_(inducing_points_[cast::to_size(i)], y);
    }
    // const Eigen::VectorXd K_uy = cov_(inducing_points_, y);
    return K_xu.dot(K_uu_ldlt_.solve(K_uy));
  }
};

template <typename CovarianceType, typename InducingFeatureType,
          typename GrouperFunction>
auto make_brute_force_pic_covariance(
    const std::vector<InducingFeatureType> &inducing_points,
    CovarianceType &&cov, GrouperFunction &&grouper) {
  return BruteForcePIC<CovarianceType, InducingFeatureType, GrouperFunction>(
      inducing_points, std::forward<CovarianceType>(cov),
      std::forward<GrouperFunction>(grouper));
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
  const double pic_error = (pic_pred.mean - direct_pred.mean).norm();
  EXPECT_LT(pic_error, 1e-7)
      << "|u|: " << kNumInducingPoints << "; |f|: " << dataset.size()
      << "; |p|: " << test_features.size()
      << "; B width: " << grouper.group_domain_size;

  const double pic_cov_error =
      (pic_pred.covariance - direct_pred.covariance).norm();
  EXPECT_LT(pic_cov_error, 1e-7)
      << "|u|: " << kNumInducingPoints << "; |f|: " << dataset.size()
      << "; |p|: " << test_features.size()
      << "; B width: " << grouper.group_domain_size
      << "; n_B: " << kNumTrainPoints << "\n"
      << pic_pred.covariance << "\n"
      << direct_pred.covariance;
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
  const double pic_error = (pic_pred.mean - direct_pred.mean).norm();
  EXPECT_LT(pic_error, 2e-7)
      << "|u|: " << kNumInducingPoints << "; |f|: " << dataset.size()
      << "; |p|: " << test_features.size()
      << "; B width: " << grouper.group_domain_size << "\n"
      << pic_pred.mean.transpose() << "\n"
      << direct_pred.mean.transpose();

  const double pic_cov_error =
      (pic_pred.covariance - direct_pred.covariance).norm();
  EXPECT_LT(pic_cov_error, 2e-7)
      << "|u|: " << kNumInducingPoints << "; |f|: " << dataset.size()
      << "; |p|: " << test_features.size()
      << "; B width: " << grouper.group_domain_size
      << "; n_B: " << kNumTrainPoints << "\n"
      << pic_pred.covariance << "\n"
      << direct_pred.covariance;
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
  // Eigen::MatrixXd WBW =
  //     Eigen::MatrixXd::Zero(u.size(), ordered_features.size());

  // Eigen::MatrixXd K_Su = S.solve(K_fu);
  // std::stringstream Ksus;
  // Ksus << "K_Su[" << blk << "]";
  // print_matrix(K_Su, Ksus.str());

  // for (const auto &block : K_ff.blocks) {
  //   // Eigen::MatrixXd K_Au = K_fu;
  //   // K_Au.middleRows(j, block.cols()) =
  //   //     Eigen::MatrixXd::Zero(block.cols(), K_Au.cols());
  //   // std::stringstream Kaus;
  //   // std::cout << "Block " << blk << " j = " << j << " size = " <<
  //   block.cols()
  //   //           << std::endl;
  //   // Kaus << "K_Au[" << blk << "]";
  //   // print_matrix(K_Au, Kaus.str());

  //   Eigen::MatrixXd WB = K_uu_ldlt.solve(K_Su.transpose());
  //   std::stringstream WBs;
  //   WBs << "W_B[" << blk << "]";
  //   print_matrix(WB, WBs.str());
  //   WB.leftCols(j) = Eigen::MatrixXd::Zero(K_fu.cols(), j);
  //   WB.rightCols(ordered_features.size() - j - block.cols()) =
  //       Eigen::MatrixXd::Zero(K_fu.cols(),
  //                             ordered_features.size() - j - block.cols());
  //   print_matrix(WB, WBs.str());

  //   print_matrix(WB * V_PIC, "WB * V");

  //   WW.push_back(WB);
  //   WBW += WB;

  //   blk++;
  //   j += block.cols();
  // }

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
  std::cout << "BFP mean (" << bfp_pred.mean.size()
            << "): " << bfp_pred.mean.transpose().format(Eigen::FullPrecision)
            << std::endl;
  const double pic_error = (pic_pred.mean - bfp_pred.mean).norm();
  // const auto test_result =
  //     test_pic(dataset, test_features, make_simple_covariance_function(),
  //              strategy, grouper);

  EXPECT_LT(pic_error, 1e-8);
  // EXPECT_LT((pic_pred.mean - test_result.mean).norm(), 1e-8);

  const double pic_cov_error =
      (pic_pred.covariance - bfp_pred.covariance).norm();
  EXPECT_LT(pic_cov_error, 1e-7);
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
  // std::cout << "BFP mean (" << bfp_pred.mean.size()
  //           << "): " <<
  //           bfp_pred.mean.transpose().format(Eigen::FullPrecision)
  //           << std::endl;
  const double pic_error = (pic_pred.mean - bfp_pred.mean).norm();
  // const auto test_result =
  //     test_pic(dataset, test_features, make_simple_covariance_function(),
  //              strategy, grouper);

  EXPECT_LT(pic_error, 1e-7);
  // EXPECT_LT((pic_pred.mean - test_result.mean).norm(), 1e-8);

  const double pic_cov_error =
      (pic_pred.covariance - bfp_pred.covariance).norm();
  EXPECT_LT(pic_cov_error, 1e-7);
}

/*
TEST(TestPicGP, EmitCSV) {
  static constexpr std::size_t kNumTrainPoints = 80;
  static constexpr std::size_t kNumTestPoints = 300;
  static constexpr std::size_t kNumInducingPoints = 20;

  static constexpr double kLargestTrainPoint =
      static_cast<double>(kNumTrainPoints) - 1.;
  static constexpr double kSpatialBuffer = 0;
  static constexpr std::size_t kNumBlocks = 5;

  UniformlySpacedInducingPoints strategy(kNumInducingPoints);
  LeaveOneIntervalOut grouper(kLargestTrainPoint /
                              static_cast<double>(kNumBlocks) + 1e-8);
  // LeaveOneIntervalOut grouper(10);

  auto direct = gp_from_covariance(make_simple_covariance_function(), "direct");

  auto pic = pic_gp_from_covariance(make_simple_covariance_function(), grouper,
                                    strategy, "pic", DenseQRImplementation{});
  auto pitc =
      sparse_gp_from_covariance(make_simple_covariance_function(), grouper,
                                strategy, "sparse", DenseQRImplementation{});

  auto dataset =
      make_toy_linear_data(-kLargestTrainPoint / 2., 1, 0.1, kNumTrainPoints);
  // auto dataset = make_toy_linear_data(5, 1, 0.1, kNumTrainPoints);

  auto bfp_cov = make_brute_force_pic_covariance(
      strategy(make_simple_covariance_function(), dataset.features),
      make_simple_covariance_function(), grouper);
  auto bfp = gp_from_covariance(bfp_cov);

  auto test_features = linspace(
      kSpatialBuffer, kLargestTrainPoint - kSpatialBuffer, kNumTestPoints);
  // auto test_features = linspace(0.1, 9.9, kNumTestPoints);
  auto direct_fit = direct.fit(dataset);
  auto pic_fit = pic.fit(dataset);
  auto pitc_fit = pitc.fit(dataset);
  // std::cout << "BFP: ";
  auto bfp_fit = bfp.fit(dataset);

  auto direct_pred =
      direct_fit.predict_with_measurement_noise(test_features).joint();
  auto pic_pred = pic_fit.predict_with_measurement_noise(test_features).joint();
  auto pitc_pred =
      pitc_fit.predict_with_measurement_noise(test_features).joint();
  // std::cout << "BFP: ";
  auto bfp_pred = bfp_fit.predict_with_measurement_noise(test_features).joint();

  std::ofstream csv_out(
      "/home/peddie/orion-engine/third_party/albatross/example.csv");

  csv_out << "type,idx,x,mean,marginal,group\n";
  for (std::size_t pred = 0; pred < direct_pred.size(); ++pred) {
    csv_out << "dense," << std::setprecision(16) << pred << ','
            << test_features[pred] << ',' <<
direct_pred.mean[cast::to_index(pred)] << ','
            << direct_pred.covariance(cast::to_index(pred),
cast::to_index(pred)) << ','
            << grouper(test_features[pred]) << '\n';
  }

  for (std::size_t pred = 0; pred < pitc_pred.size(); ++pred) {
    csv_out << "pitc," << std::setprecision(16) << pred << ','
            << test_features[pred] << ',' <<
pitc_pred.mean[cast::to_index(pred)] << ','
            << pitc_pred.covariance(cast::to_index(pred), cast::to_index(pred))
<< ','
            << grouper(test_features[pred]) << '\n';
  }

  for (std::size_t pred = 0; pred < pic_pred.size(); ++pred) {
    csv_out << "pic," << std::setprecision(16) << pred << ','
            << test_features[pred] << ',' << pic_pred.mean[cast::to_index(pred)]
<< ','
            << pic_pred.covariance(cast::to_index(pred), cast::to_index(pred))
<< ','
            << grouper(test_features[pred]) << '\n';
  }

  for (std::size_t pred = 0; pred < bfp_pred.size(); ++pred) {
    csv_out << "bfp," << std::setprecision(16) << pred << ','
            << test_features[pred] << ',' << bfp_pred.mean[cast::to_index(pred)]
<< ','
            << bfp_pred.covariance(cast::to_index(pred), cast::to_index(pred))
<< ','
            << grouper(test_features[pred]) << '\n';
  }

  csv_out << std::endl;
  csv_out.close();

  std::ofstream points_out("/home/peddie/orion-engine/third_party/"
                           "albatross/example_points.csv");

  points_out << "type,x,y\n";
  for (std::size_t i = 0; i < dataset.size(); ++i) {
    points_out << "train," << dataset.features[i] << ','
               << dataset.targets.mean[cast::to_index(i)] << '\n';
  }

  for (const auto &f : pic_fit.get_fit().inducing_features) {
    points_out << "inducing," << f << ",0\n";
  }
  points_out << std::endl;
  points_out.close();

  // std::cout << "BFP L: "
  //           << Eigen::MatrixXd(bfp_fit.get_fit().train_covariance.matrixL())
  //                  .format(Eigen::FullPrecision)
  //           << std::endl;
  // std::cout << "BFP D: "
  //           <<
  //           bfp_fit.get_fit().train_covariance.vectorD().transpose().format(
  //                  Eigen::FullPrecision)
  //           << std::endl;

  // const auto test_result =
  //     test_pic(dataset, test_features, make_simple_covariance_function(),
  //              strategy, grouper);

  // std::cout << "bfp_pred.covariance (" << bfp_pred.covariance.rows() << "x"
  //           << bfp_pred.covariance.cols() << "):\n"
  //           << bfp_pred.covariance << std::endl;

  const double pic_error = (pic_pred.mean - direct_pred.mean).norm();
  EXPECT_LT(pic_error, 5e-7);
  // EXPECT_LT((pic_pred.mean - test_result.mean).norm(), 1e-7);
  // << "|u|: " << kNumInducingPoints << "; |f|: " << dataset.size()
  // << "; |p|: " << test_features.size()
  // << "; B width: " << grouper.group_domain_size << "\n"
  // << pic_pred.mean.transpose() << "\n"
  // << direct_pred.mean.transpose();

  const double pic_cov_error =
      (pic_pred.covariance - direct_pred.covariance).norm();
  EXPECT_LT(pic_cov_error, 5e-7);
  // << "|u|: " << kNumInducingPoints << "; |f|: " << dataset.size()
  // << "; |p|: " << test_features.size()
  // << "; B width: " << grouper.group_domain_size
  // << "; n_B: " << kNumTrainPoints << "\n"
  // << pic_pred.covariance << "\n"
  // << direct_pred.covariance;
}
*/

TEST(TestPicGP, EmitCSVExtended) {
  static constexpr double xMin = 0.;
  static constexpr double xMax = 300.;
  static constexpr std::size_t kNumTrainPoints = 80;
  static constexpr std::size_t kNumTestPoints =
      300; // even number since right-most boundary shall not get predicted. it
           // would fall into a group without observtaions
  static constexpr std::size_t kNumInducingPoints = 20;

  // static constexpr double kLargestTrainPoint =
  //     static_cast<double>(kNumTrainPoints) - 1.;
  // static constexpr double kSpatialBuffer = 0;
  static constexpr std::size_t kNumBlocks = 4;

  UniformlySpacedInducingPoints strategy(kNumInducingPoints);
  LeaveOneIntervalOut grouper(kNumTestPoints / static_cast<double>(kNumBlocks));
  // LeaveOneIntervalOut grouper(10);

  auto direct = gp_from_covariance(make_simple_covariance_function(), "direct");

  auto pic = pic_gp_from_covariance(make_simple_covariance_function(), grouper,
                                    strategy, "pic", DenseQRImplementation{});
  auto pitc =
      sparse_gp_from_covariance(make_simple_covariance_function(), grouper,
                                strategy, "sparse", DenseQRImplementation{});

  /* auto dataset =
      make_toy_linear_data(-kLargestTrainPoint / 2., 1, 0.1, kNumTrainPoints);
   */
  auto dataset =
      make_toy_bellshaped_data(xMin, xMax, 1., 0.05, kNumTrainPoints);
  // auto dataset = make_toy_linear_data(5, 1, 0.1, kNumTrainPoints);

  auto bfp_cov = make_brute_force_pic_covariance(
      strategy(make_simple_covariance_function(), dataset.features),
      make_simple_covariance_function(), grouper);
  auto bfp = gp_from_covariance(bfp_cov);

  auto test_features =
      linspace(xMin, xMax - (xMax - xMin) / kNumTestPoints, kNumTestPoints);
  // auto test_features = linspace(0.1, 9.9, kNumTestPoints);
  auto direct_fit = direct.fit(dataset);
  auto pic_fit = pic.fit(dataset);
  auto pitc_fit = pitc.fit(dataset);
  // std::cout << "BFP: ";
  auto bfp_fit = bfp.fit(dataset);

  auto direct_pred =
      direct_fit.predict_with_measurement_noise(test_features).joint();
  auto pic_pred = pic_fit.predict_with_measurement_noise(test_features).joint();
  auto pitc_pred =
      pitc_fit.predict_with_measurement_noise(test_features).joint();
  // std::cout << "BFP: ";
  auto bfp_pred = bfp_fit.predict_with_measurement_noise(test_features).joint();

  std::ofstream csv_out("/Users/wolfganglanghans/Code/orion/third_party/"
                        "orion-engine/third_party/albatross/example.csv");

  csv_out << "type,idx,x,mean,marginal,group\n";
  for (std::size_t pred = 0; pred < direct_pred.size(); ++pred) {
    csv_out << "dense," << std::setprecision(16) << pred << ','
            << test_features[pred] << ','
            << direct_pred.mean[cast::to_index(pred)] << ','
            << direct_pred.covariance(cast::to_index(pred),
                                      cast::to_index(pred))
            << ',' << grouper(test_features[pred]) << '\n';
  }

  /*for (std::size_t pred = 0; pred < pitc_pred.size(); ++pred) {
     csv_out << "pitc," << std::setprecision(16) << pred << ','
            << test_features[pred] << ',' <<
  pitc_pred.mean[cast::to_index(pred)] << ','
            << pitc_pred.covariance(cast::to_index(pred), cast::to_index(pred))
  << ','
            << grouper(test_features[pred]) << '\n';
  } */

  for (std::size_t pred = 0; pred < pic_pred.size(); ++pred) {
    csv_out << "pic," << std::setprecision(16) << pred << ','
            << test_features[pred] << ',' << pic_pred.mean[cast::to_index(pred)]
            << ','
            << pic_pred.covariance(cast::to_index(pred), cast::to_index(pred))
            << ',' << grouper(test_features[pred]) << '\n';
  }

  for (std::size_t pred = 0; pred < bfp_pred.size(); ++pred) {
    csv_out << "bfp," << std::setprecision(16) << pred << ','
            << test_features[pred] << ',' << bfp_pred.mean[cast::to_index(pred)]
            << ','
            << bfp_pred.covariance(cast::to_index(pred), cast::to_index(pred))
            << ',' << grouper(test_features[pred]) << '\n';
  }

  csv_out << std::endl;
  csv_out.close();

  std::ofstream points_out(
      "/Users/wolfganglanghans/Code/orion/third_party/orion-engine/third_party/"
      "albatross/example_points.csv");

  points_out << "type,x,y\n";
  for (std::size_t i = 0; i < dataset.size(); ++i) {
    points_out << "train," << dataset.features[i] << ','
               << dataset.targets.mean[cast::to_index(i)] << '\n';
  }

  for (const auto &f : pic_fit.get_fit().inducing_features) {
    points_out << "inducing," << f << ",0\n";
  }
  points_out << std::endl;
  points_out.close();

  // std::cout << "BFP L: "
  //           << Eigen::MatrixXd(bfp_fit.get_fit().train_covariance.matrixL())
  //                  .format(Eigen::FullPrecision)
  //           << std::endl;
  // std::cout << "BFP D: "
  //           <<
  //           bfp_fit.get_fit().train_covariance.vectorD().transpose().format(
  //                  Eigen::FullPrecision)
  //           << std::endl;

  // const auto test_result =
  //     test_pic(dataset, test_features, make_simple_covariance_function(),
  //              strategy, grouper);

  // std::cout << "bfp_pred.covariance (" << bfp_pred.covariance.rows() << "x"
  //           << bfp_pred.covariance.cols() << "):\n"
  //           << bfp_pred.covariance << std::endl;

  const double pic_error = (pic_pred.mean - direct_pred.mean).norm();
  EXPECT_LT(pic_error, 1e-5);
  // EXPECT_LT((pic_pred.mean - test_result.mean).norm(), 1e-7);
  // << "|u|: " << kNumInducingPoints << "; |f|: " << dataset.size()
  // << "; |p|: " << test_features.size()
  // << "; B width: " << grouper.group_domain_size << "\n"
  // << pic_pred.mean.transpose() << "\n"
  // << direct_pred.mean.transpose();

  const double pic_cov_error =
      (pic_pred.covariance - direct_pred.covariance).norm();
  EXPECT_LT(pic_cov_error, 5e-5);
  // << "|u|: " << kNumInducingPoints << "; |f|: " << dataset.size()
  // << "; |p|: " << test_features.size()
  // << "; B width: " << grouper.group_domain_size
  // << "; n_B: " << kNumTrainPoints << "\n"
  // << pic_pred.covariance << "\n"
  // << direct_pred.covariance;
}

} // namespace albatross