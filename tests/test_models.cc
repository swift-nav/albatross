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
  auto cv_scores = cross_validated_scores(
      evaluation_metrics::root_mean_square_error, folds, model.get());
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
  auto scores = cross_validated_scores(
      evaluation_metrics::root_mean_square_error, folds, model.get());
  RegressionDataset<double> dataset_without_variance(dataset.features,
                                                     dataset.targets.mean);
  auto folds_without_variance = leave_one_out(dataset_without_variance);
  auto scores_without_variance =
      cross_validated_scores(evaluation_metrics::root_mean_square_error,
                             folds_without_variance, model.get());

  EXPECT_LE(scores.mean(), scores_without_variance.mean());
}
} // namespace albatross
