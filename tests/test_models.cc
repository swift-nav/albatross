/*
 * Copyright (C) 2018 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "covariance_functions/covariance_functions.h"
#include "evaluate.h"
#include "models/gp.h"
#include "models/least_squares.h"
#include "test_utils.h"
#include <cereal/archives/json.hpp>
#include <gtest/gtest.h>

namespace albatross {

class AbstractTestModel {
public:
  virtual ~AbstractTestModel(){};
  virtual std::unique_ptr<RegressionModel<double>> create() const = 0;
};

class MakeGaussianProcess : public AbstractTestModel {
public:
  std::unique_ptr<RegressionModel<double>> create() const override {
    using SqrExp = SquaredExponential<EuclideanDistance>;
    using Noise = IndependentNoise<double>;
    CovarianceFunction<SqrExp> squared_exponential = {SqrExp(100., 100.)};
    CovarianceFunction<Noise> noise = {Noise(0.1)};
    auto covariance = squared_exponential + noise;
    return gp_pointer_from_covariance<double>(covariance);
  }
};

class MakeLinearRegression : public AbstractTestModel {
public:
  std::unique_ptr<RegressionModel<double>> create() const override {
    return std::make_unique<LinearRegression>();
  }
};

template <typename ModelCreator>
class RegressionModelTester : public ::testing::Test {
public:
  ModelCreator creator;
};

typedef ::testing::Types<MakeLinearRegression, MakeGaussianProcess>
    ModelCreators;
TYPED_TEST_CASE(RegressionModelTester, ModelCreators);

TYPED_TEST(RegressionModelTester, performs_reasonably_on_linear_data) {
  auto dataset = make_toy_linear_data();
  auto folds = leave_one_out(dataset);
  auto model = this->creator.create();
  EvaluationMetric<Eigen::VectorXd> rmse =
      evaluation_metrics::root_mean_square_error;
  auto cv_scores = cross_validated_scores(rmse, folds, model.get());
  // Here we make sure the cross validated mean absolute error is reasonable.
  // Note that because we are running leave one out cross validation, the
  // RMSE for each fold is just the absolute value of the error.
  EXPECT_LE(cv_scores.mean(), 0.1);
}

/*
 * Here we build two different datasets.  Each dataset consists of targets which
 * have been distorted by non-constant noise (heteroscedastic), we then perform
 * cross-validated evaluation of a GaussianProcess which takes that noise into
 * account, and one which is agnostic of the added noise and assert that taking
 * noise into account improves the model.
 */
TEST(test_models, test_with_target_distribution) {
  auto dataset = make_heteroscedastic_toy_linear_data();

  auto folds = leave_one_out(dataset);
  auto model = MakeGaussianProcess().create();
  EvaluationMetric<Eigen::VectorXd> rmse =
      evaluation_metrics::root_mean_square_error;
  auto scores = cross_validated_scores(rmse, folds, model.get());
  RegressionDataset<double> dataset_without_variance(dataset.features,
                                                     dataset.targets.mean);
  auto folds_without_variance = leave_one_out(dataset_without_variance);

  auto scores_without_variance =
      cross_validated_scores(rmse, folds_without_variance, model.get());

  EXPECT_LE(scores.mean(), scores_without_variance.mean());
}

TEST(test_models, test_predict_variants) {
  auto dataset = make_heteroscedastic_toy_linear_data();

  auto model = MakeGaussianProcess().create();
  model->fit(dataset);

  const auto joint_predictions = model->predict(dataset.features);
  const auto marginal_predictions =
      model->predict<MarginalDistribution>(dataset.features);
  const auto mean_predictions =
      model->predict<Eigen::VectorXd>(dataset.features);

  const auto single_pred_joint =
      model->predict<JointDistribution>(dataset.features[0]);
  EXPECT_NEAR(single_pred_joint.mean[0], mean_predictions[0], 1e-6);
  EXPECT_NEAR(single_pred_joint.get_diagonal(0),
              joint_predictions.get_diagonal(0), 1e-6);

  const auto single_pred_marginal =
      model->predict<MarginalDistribution>(dataset.features[0]);
  EXPECT_NEAR(single_pred_marginal.mean[0], mean_predictions[0], 1e-6);
  EXPECT_NEAR(single_pred_marginal.get_diagonal(0),
              joint_predictions.get_diagonal(0), 1e-6);

  const auto single_pred_mean =
      model->predict<Eigen::VectorXd>(dataset.features[0]);
  EXPECT_NEAR(single_pred_mean[0], mean_predictions[0], 1e-6);

  for (Eigen::Index i = 0; i < joint_predictions.mean.size(); i++) {
    EXPECT_NEAR(joint_predictions.mean[i], mean_predictions[i], 1e-6);
    EXPECT_NEAR(joint_predictions.mean[i], marginal_predictions.mean[i], 1e-6);
    EXPECT_NEAR(joint_predictions.covariance(i, i),
                marginal_predictions.covariance.diagonal()[i], 1e-6);
  }
}

} // namespace albatross
