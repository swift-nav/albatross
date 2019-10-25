/*
 * Copyright (C) 2019 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <albatross/GP>
#include <albatross/LeastSquares>
#include <albatross/NullModel>
#include <albatross/Ransac>
#include <gtest/gtest.h>

#include "test_utils.h"

namespace albatross {

inline auto make_simple_covariance_function() {
  SquaredExponential<EuclideanDistance> squared_exponential(100., 100.);
  IndependentNoise<double> noise(0.1);
  return squared_exponential + measurement_only(noise);
}

class MakeGaussianProcess {
public:
  MakeGaussianProcess(){};

  auto get_model() const {
    auto covariance = make_simple_covariance_function();
    return gp_from_covariance(covariance);
  }

  auto get_dataset() const { return make_toy_linear_data(); }
};

class MakeRansacGaussianProcess {
public:
  auto get_model() const {
    auto covariance = make_simple_covariance_function();

    RansacConfig config;
    config.inlier_threshold = 1.;
    config.random_sample_size = 3;
    config.min_consensus_size = 3;
    config.max_iterations = 20;
    config.max_failed_candidates = 20;

    const auto gp = gp_from_covariance(covariance);

    DefaultGPRansacStrategy ransac_strategy;
    return gp.ransac(ransac_strategy, config);
  }

  auto get_dataset() const { return make_toy_linear_data(); }
};

class MakeRansacChiSquaredGaussianProcess {
public:
  auto get_model() const {
    auto covariance = make_simple_covariance_function();

    RansacConfig config;
    config.inlier_threshold = 1.;
    config.random_sample_size = 3;
    config.min_consensus_size = 3;
    config.max_iterations = 20;
    config.max_failed_candidates = 20;

    const auto gp = gp_from_covariance(covariance);

    GaussianProcessRansacStrategy<ChiSquaredCdf, ChiSquaredConsensusMetric,
                                  LeaveOneOutGrouper>
        ransac_strategy;

    return gp.ransac(ransac_strategy, config);
  }

  auto get_dataset() const { return make_toy_linear_data(); }
};

namespace adapted {

inline std::vector<double>
convert_features(const std::vector<AdaptedFeature> &features) {
  std::vector<double> converted;
  for (const auto &f : features) {
    converted.push_back(f.value);
  }
  return converted;
}

} // namespace adapted

template <typename CovarianceFunc>
class AdaptedGaussianProcess
    : public GaussianProcessBase<CovarianceFunc,
                                 AdaptedGaussianProcess<CovarianceFunc>> {
public:
  using Base = GaussianProcessBase<CovarianceFunc,
                                   AdaptedGaussianProcess<CovarianceFunc>>;

  using Base::_fit_impl;
  using Base::_predict_impl;
  using Base::Base;
  using Base::compute_covariance;

  using FitType = Fit<GPFit<Eigen::SerializableLDLT, double>>;

  template <typename FeatureType, typename PredictType, typename GroupKey>
  std::map<GroupKey, PredictType>
  cross_validated_predictions(const RegressionDataset<FeatureType> &dataset,
                              const GroupIndexer<GroupKey> &group_indexer,
                              PredictTypeIdentity<PredictType> identity) const {
    return gp_cross_validated_predictions(dataset, group_indexer, *this,
                                          identity);
  }

  auto _fit_impl(const std::vector<AdaptedFeature> &features,
                 const MarginalDistribution &targets) const {
    const auto converted = adapted::convert_features(features);
    return Base::_fit_impl(converted, targets);
  }

  template <typename PredictType>
  PredictType _predict_impl(const std::vector<AdaptedFeature> &features,
                            const FitType &adapted_gp_fit,
                            PredictTypeIdentity<PredictType> &&) const {
    return Base::_predict_impl(adapted::convert_features(features),
                               adapted_gp_fit,
                               PredictTypeIdentity<PredictType>());
  }

  Eigen::MatrixXd
  compute_covariance(const std::vector<AdaptedFeature> &features) const {
    return this->covariance_function_(adapted::convert_features(features));
  }
};

class MakeAdaptedGaussianProcess {
public:
  auto get_model() const {
    auto covariance = make_simple_covariance_function();
    AdaptedGaussianProcess<decltype(covariance)> gp(covariance);
    return gp;
  }

  auto get_dataset() const { return make_adapted_toy_linear_data(); }
};

struct AdaptedRansacStrategy : public GaussianProcessRansacStrategy<
                                   NegativeLogLikelihood<JointDistribution>,
                                   FeatureCountConsensusMetric, LeaveOneOut> {

  template <typename ModelType>
  RansacFunctions<FitAndIndices<ModelType, double>>
  operator()(const ModelType &model,
             const RegressionDataset<AdaptedFeature> &dataset) const {
    const RegressionDataset<double> converted(
        adapted::convert_features(dataset.features), dataset.targets);
    const auto indexer = get_indexer(converted);
    const FeatureCountConsensusMetric consensus_metric;
    const AlwaysAcceptCandidateMetric always_accept;
    return get_gp_ransac_functions(model, converted, indexer,
                                   this->inlier_metric_, consensus_metric,
                                   always_accept);
  }
};

class MakeRansacAdaptedGaussianProcess {
public:
  auto get_model() const {
    auto covariance = make_simple_covariance_function();

    RansacConfig config;
    config.inlier_threshold = 1.;
    config.random_sample_size = 3;
    config.min_consensus_size = 3;
    config.max_iterations = 20;
    config.max_failed_candidates = 20;

    AdaptedGaussianProcess<decltype(covariance)> gp(covariance);

    AdaptedRansacStrategy ransac_strategy;
    return gp.ransac(ransac_strategy, config);
  }

  auto get_dataset() const { return make_adapted_toy_linear_data(); }
};

class MakeLinearRegression {
public:
  LinearRegression get_model() const { return LinearRegression(); }

  RegressionDataset<double> get_dataset() const {
    return make_toy_linear_data();
  }
};

class MakeNullModel {
public:
  NullModel get_model() const { return NullModel(); }

  RegressionDataset<double> get_dataset() const {
    return make_toy_linear_data();
  }
};

template <typename ModelTestCase>
class RegressionModelTester : public ::testing::Test {
public:
  ModelTestCase test_case;
};

typedef ::testing::Types<MakeLinearRegression, MakeGaussianProcess,
                         MakeAdaptedGaussianProcess, MakeRansacGaussianProcess,
                         MakeRansacChiSquaredGaussianProcess,
                         MakeRansacAdaptedGaussianProcess, MakeNullModel>
    ExampleModels;

TYPED_TEST_CASE_P(RegressionModelTester);

enum PredictLevel { MEAN, MARGINAL, JOINT };

/*
 * This TestPredictVariants struct provides different levels of
 * testing depending on what sort of predictions are available.
 */
template <typename PredictionType, typename = void> struct TestPredictVariants {
  PredictLevel test(const PredictionType &pred) const {
    const Eigen::VectorXd pred_mean = pred.mean();
    const auto get_mean = pred.template get<Eigen::VectorXd>();
    EXPECT_EQ(get_mean, pred_mean);
    EXPECT_GT(pred_mean.size(), 0);
    return PredictLevel::MEAN;
  }

  PredictLevel xfail(const PredictionType &pred) const {
    const Eigen::VectorXd pred_mean = pred.mean();
    EXPECT_EQ(pred_mean.size(), 0);
    return PredictLevel::MEAN;
  }
};

template <typename PredictionType>
struct TestPredictVariants<
    PredictionType, std::enable_if_t<has_marginal<PredictionType>::value &&
                                     !has_joint<PredictionType>::value>> {
  PredictLevel test(const PredictionType &pred) const {
    const Eigen::VectorXd pred_mean = pred.mean();
    const MarginalDistribution marginal = pred.marginal();
    const auto get_marginal = pred.template get<MarginalDistribution>();
    EXPECT_EQ(get_marginal, marginal);
    EXPECT_LE((pred_mean - marginal.mean).norm(), 1e-8);
    return PredictLevel::MARGINAL;
  }

  PredictLevel xfail(const PredictionType &pred) const {
    const Eigen::VectorXd pred_mean = pred.mean();
    const MarginalDistribution marginal = pred.marginal();
    EXPECT_GT((pred_mean - marginal.mean).norm(), 1e-8);
    return PredictLevel::MARGINAL;
  }
};

template <typename PredictionType>
struct TestPredictVariants<PredictionType,
                           std::enable_if_t<has_joint<PredictionType>::value>> {
  PredictLevel test(const PredictionType &pred) const {
    const Eigen::VectorXd pred_mean = pred.mean();
    const MarginalDistribution marginal = pred.marginal();
    EXPECT_LE((pred_mean - marginal.mean).norm(), 1e-8);
    const JointDistribution joint = pred.joint();

    const auto get_joint = pred.template get<JointDistribution>();
    EXPECT_EQ(get_joint, joint);

    EXPECT_LE((pred_mean - joint.mean).norm(), 1e-8);
    EXPECT_LE(
        (marginal.covariance.diagonal() - joint.covariance.diagonal()).norm(),
        1e-8);
    return PredictLevel::JOINT;
  }

  PredictLevel xfail(const PredictionType &pred) const {
    const Eigen::VectorXd pred_mean = pred.mean();
    const MarginalDistribution marginal = pred.marginal();
    const JointDistribution joint = pred.joint();
    bool mean_is_close = (pred_mean - joint.mean).norm() <= 1e-8;
    bool variance_is_close =
        (marginal.covariance.diagonal() - joint.covariance.diagonal()).norm() <=
        1e-8;
    EXPECT_FALSE(mean_is_close && variance_is_close);
    return PredictLevel::JOINT;
  }
};

template <typename PredictionType>
void expect_predict_variants_consistent(const PredictionType &pred) {
  TestPredictVariants<PredictionType> tester;
  const auto level = tester.test(pred);
  // Just in case the traits above don't work.
  if (has_mean<PredictionType>::value) {
    EXPECT_GE(level, PredictLevel::MEAN);
  }

  if (has_marginal<PredictionType>::value) {
    EXPECT_GE(level, PredictLevel::MARGINAL);
  }

  if (has_joint<PredictionType>::value) {
    EXPECT_GE(level, PredictLevel::JOINT);
  }
}

template <typename PredictionType>
void expect_predict_variants_inconsistent(const PredictionType &pred) {
  TestPredictVariants<PredictionType> tester;
  const auto level = tester.xfail(pred);
  // Just in case the traits above don't work.
  if (has_mean<PredictionType>::value) {
    EXPECT_GE(level, PredictLevel::MEAN);
  }

  if (has_marginal<PredictionType>::value) {
    EXPECT_GE(level, PredictLevel::MARGINAL);
  }

  if (has_joint<PredictionType>::value) {
    EXPECT_GE(level, PredictLevel::JOINT);
  }
}

} // namespace albatross
