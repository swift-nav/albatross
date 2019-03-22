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

#include <chrono>
#include <gtest/gtest.h>

#include "test_models.h"

#include "Evaluation"

namespace albatross {

// Group values by interval, but return keys that once sorted won't be
// in order
std::string group_by_interval(const double &x) {
  if (x <= 3) {
    return "2";
  } else if (x <= 6) {
    return "3";
  } else {
    return "1";
  }
}

std::string group_by_interval(const AdaptedFeature &x) {
  return group_by_interval(x.value);
}

bool is_monotonic_increasing(const Eigen::VectorXd &x) {
  for (Eigen::Index i = 0; i < x.size() - 1; i++) {
    if (x[i + 1] - x[i] <= 0.) {
      return false;
    }
  }
  return true;
}

TYPED_TEST_P(RegressionModelTester, test_logo_predict_variants) {
  auto dataset = this->test_case.get_dataset();
  auto model = this->test_case.get_model();

  // Here we assume that the test case is linear, then split
  // it using a group function which will not preserve order
  // and make sure that cross validation properly reassembles
  // the predictions
  LeaveOneGroupOut<typename decltype(dataset)::Feature> logo(group_by_interval);
  const auto prediction = model.cross_validate().predict(dataset, logo);

  EXPECT_TRUE(is_monotonic_increasing(prediction.mean()));

  expect_predict_variants_consistent(prediction);
}

TYPED_TEST_P(RegressionModelTester, test_loo_predict_variants) {
  auto dataset = this->test_case.get_dataset();
  auto model = this->test_case.get_model();

  LeaveOneOut leave_one_out;
  const auto prediction =
      model.cross_validate().predict(dataset, leave_one_out);

  expect_predict_variants_consistent(prediction);
}

TYPED_TEST_P(RegressionModelTester, test_loo_get_predictions) {
  auto dataset = this->test_case.get_dataset();
  auto model = this->test_case.get_model();

  LeaveOneOut leave_one_out;
  const auto predictions =
      model.cross_validate().get_predictions(dataset, leave_one_out);

  for (const auto &pred : predictions) {
    expect_predict_variants_consistent(pred);
  }
}

TYPED_TEST_P(RegressionModelTester, test_score_variants) {
  auto dataset = this->test_case.get_dataset();
  auto model = this->test_case.get_model();

  LeaveOneOut leave_one_out;
  const auto indexer = leave_one_out(dataset);
  const auto folds = folds_from_fold_indexer(dataset, indexer);

  const RootMeanSquareError rmse;

  auto cv_scores = model.cross_validate().scores(rmse, folds);

  auto cv_fast_scores = model.cross_validate().scores(rmse, dataset, indexer);
  auto cv_fast_scores_alternate =
      model.cross_validate().scores(rmse, dataset, leave_one_out);

  EXPECT_LE((cv_fast_scores - cv_fast_scores_alternate).norm(), 1e-8);
  EXPECT_LE((cv_scores - cv_fast_scores).norm(), 1e-8);
  // Here we make sure the cross validated mean absolute error is reasonable.
  // Note that because we are running leave one out cross validation, the
  // RMSE for each fold is just the absolute value of the error.
  EXPECT_LE(cv_scores.mean(), 0.1);
}

REGISTER_TYPED_TEST_CASE_P(RegressionModelTester, test_loo_predict_variants,
                           test_logo_predict_variants, test_loo_get_predictions,
                           test_score_variants);

INSTANTIATE_TYPED_TEST_CASE_P(test_cross_validation, RegressionModelTester,
                              ExampleModels);

/*
 * Here we build two different datasets.  Each dataset consists of targets
 * which have been distorted by non-constant noise (heteroscedastic), we then
 * perform cross-validated evaluation of a GaussianProcess which takes that
 * noise into account, and one which is agnostic of the added noise and assert
 * that taking noise into account improves the model.
 */
TEST(test_crossvalidation, test_heteroscedastic) {
  const auto dataset = make_heteroscedastic_toy_linear_data();

  auto model = MakeGaussianProcess().get_model();

  LeaveOneOut loo;
  const RootMeanSquareError rmse;
  const auto scores = model.cross_validate().scores(rmse, dataset, loo);

  RegressionDataset<double> dataset_without_variance(dataset.features,
                                                     dataset.targets.mean);

  const auto scores_without_variance =
      model.cross_validate().scores(rmse, dataset_without_variance, loo);

  EXPECT_LE(scores.mean(), scores_without_variance.mean());
}

class MakeLargeGaussianProcess {
public:
  auto get_model() const {
    auto covariance = make_simple_covariance_function();
    return gp_from_covariance(covariance);
  }

  RegressionDataset<double> get_dataset() const {
    return make_toy_linear_data(5., 1., 0.1, 100);
  }
};

class MakeLargeAdaptedGaussianProcess {
public:
  auto get_model() const {
    auto covariance = make_simple_covariance_function();
    AdaptedGaussianProcess<decltype(covariance)> gp(covariance);
    return gp;
  }

  auto get_dataset() const {
    return make_adapted_toy_linear_data(5., 1., 0.1, 100);
  }
};

template <typename ModelTestCase>
class SpecializedCrossValidationTester : public ::testing::Test {
public:
  ModelTestCase test_case;
};

typedef ::testing::Types<MakeLargeGaussianProcess,
                         MakeLargeAdaptedGaussianProcess>
    SpecializedModels;
TYPED_TEST_CASE(SpecializedCrossValidationTester, SpecializedModels);

TYPED_TEST(SpecializedCrossValidationTester,
           test_uses_specialized_cross_validation_functions) {
  auto dataset = this->test_case.get_dataset();
  auto model = this->test_case.get_model();

  LeaveOneOut leave_one_out;
  // time the computation of RMSE using the fast LOO variant.
  using namespace std::chrono;
  high_resolution_clock::time_point start = high_resolution_clock::now();
  const auto cv_fast_scores = model.cross_validate().scores(
      RootMeanSquareError(), dataset, leave_one_out);
  high_resolution_clock::time_point end = high_resolution_clock::now();
  auto fast_duration = duration_cast<microseconds>(end - start).count();

  // time RMSE using the default method.
  const auto loo_indexer = leave_one_out(dataset);
  const auto folds = folds_from_fold_indexer(dataset, loo_indexer);
  start = high_resolution_clock::now();
  const auto cv_slow_scores =
      model.cross_validate().scores(RootMeanSquareError(), folds);
  end = high_resolution_clock::now();
  auto slow_duration = duration_cast<microseconds>(end - start).count();

  // Make sure the faster variant is actually faster and that the results
  // are the same.
  EXPECT_LT(fast_duration, 0.5 * slow_duration);
  EXPECT_NEAR((cv_fast_scores - cv_slow_scores).norm(), 0., 1e-8);
}

} // namespace albatross
