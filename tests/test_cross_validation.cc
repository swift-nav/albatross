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

#include <albatross/Evaluation>
#include <chrono>
#include <gtest/gtest.h>

#include "test_models.h"

namespace albatross {

template <typename FeatureType>
std::string group_by_interval(const FeatureType &);

// Group values by interval, but return keys that once sorted won't be
// in order
template <> std::string group_by_interval(const double &x) {
  if (x <= 3) {
    return "2";
  } else if (x <= 6) {
    return "3";
  } else {
    return "1";
  }
}

template <> std::string group_by_interval(const AdaptedFeature &x) {
  return group_by_interval<double>(x.value);
}

TEST(test_cross_validation, test_fold_creation) {
  const auto dataset = make_toy_linear_data();
  const auto folds = folds_from_grouper(dataset, group_by_interval<double>);
  EXPECT_EQ(folds.size(), 3);
}

bool is_monotonic_increasing(const Eigen::VectorXd &x) {
  for (Eigen::Index i = 0; i < x.size() - 1; i++) {
    if (x[i + 1] - x[i] < 0.) {
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

  using FeatureType = typename decltype(dataset)::Feature;

  const auto prediction =
      model.cross_validate().predict(dataset, &group_by_interval<FeatureType>);

  EXPECT_TRUE(is_monotonic_increasing(prediction.mean()));

  expect_predict_variants_consistent(prediction);
}

TYPED_TEST_P(RegressionModelTester, test_loo_predict_variants) {
  auto dataset = this->test_case.get_dataset();
  auto model = this->test_case.get_model();

  LeaveOneOutGrouper leave_one_out;
  const auto prediction =
      model.cross_validate().predict(dataset, leave_one_out);

  expect_predict_variants_consistent(prediction);
}

TYPED_TEST_P(RegressionModelTester, test_loo_get_predictions) {
  auto dataset = this->test_case.get_dataset();
  auto model = this->test_case.get_model();

  LeaveOneOutGrouper leave_one_out;
  const auto predictions =
      model.cross_validate().predictions(dataset, leave_one_out);

  for (const auto &pred : predictions) {
    expect_predict_variants_consistent(pred.second);
  }
}

TYPED_TEST_P(RegressionModelTester, test_score_variants) {
  auto dataset = this->test_case.get_dataset();
  auto model = this->test_case.get_model();

  LeaveOneOutGrouper leave_one_out;
  const auto indexer = group_by(dataset, leave_one_out).indexers();
  const auto folds = folds_from_group_indexer(dataset, indexer);

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
  if (!std::is_same<decltype(model), NullModel>::value) {
    EXPECT_LE(cv_scores.mean(), 0.1);
  }
}

REGISTER_TYPED_TEST_SUITE_P(RegressionModelTester, test_loo_predict_variants,
                            test_logo_predict_variants,
                            test_loo_get_predictions, test_score_variants);

INSTANTIATE_TYPED_TEST_SUITE_P(test_cross_validation, RegressionModelTester,
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

  LeaveOneOutGrouper loo;
  const RootMeanSquareError rmse;
  const auto scores = model.cross_validate().scores(rmse, dataset, loo);

  RegressionDataset<double> dataset_without_variance(dataset.features,
                                                     dataset.targets.mean);

  const auto scores_without_variance =
      model.cross_validate().scores(rmse, dataset_without_variance, loo);

  EXPECT_LE(scores.mean(), scores_without_variance.mean());
}

TEST(test_crossvalidation, test_leave_one_out_conditional_variance) {
  const auto dataset = make_toy_linear_data();

  auto model = MakeGaussianProcess().get_model();

  LeaveOneOutGrouper loo;
  const auto loo_marginal =
      model.cross_validate().predict(dataset, loo).marginal();

  const auto meas = as_measurements(dataset.features);
  Eigen::MatrixXd cov = model.get_covariance()(meas, meas);
  cov.diagonal() = cov.diagonal() + dataset.targets.covariance.diagonal();
  const Eigen::VectorXd loo_variance = leave_one_out_conditional_variance(cov);
  EXPECT_LE((loo_marginal.covariance.diagonal() - loo_variance).norm(), 1e-8);
}

TEST(test_crossvalidation, test_leave_one_out_conditional) {
  const auto dataset = make_toy_linear_data();

  auto model = MakeGaussianProcess().get_model();

  LeaveOneOutGrouper loo;
  const auto loo_marginal =
      model.cross_validate().predict(dataset, loo).marginal();

  const auto meas = as_measurements(dataset.features);
  Eigen::MatrixXd cov = model.get_covariance()(meas, meas);
  JointDistribution prior(Eigen::VectorXd::Zero(cov.rows()), cov);
  const auto actual = leave_one_out_conditional(prior, dataset.targets);

  EXPECT_LE((loo_marginal.mean - actual.mean).norm(), 1e-6);
  EXPECT_LE((loo_marginal.covariance.diagonal() - actual.covariance.diagonal())
                .norm(),
            1e-6);

  // With a new dataset with a perturbed observation, the leave one out
  // prediction of that perturbed element should not have changed, while all the
  // other predictions will have changed.
  RegressionDataset<double> perturbed_dataset(dataset);
  perturbed_dataset.targets.mean[0] += 10;
  const auto perturbed_conditional =
      leave_one_out_conditional(prior, perturbed_dataset.targets);
  EXPECT_NEAR(perturbed_conditional.mean[0], actual.mean[0], 1e-6);
  EXPECT_GT((perturbed_conditional.mean - actual.mean).norm(), 1.);
}

TEST(test_crossvalidation, test_leave_one_out_equivalences) {
  // Make sure that brute force leave one group out computations
  // match both the model based, model.cross_validate(), approach
  // and the leave_one_group_out_conditional_* free functions

  const auto dataset = make_toy_linear_data();
  auto model = MakeGaussianProcess().get_model();

  const auto indexers = dataset.group_by(group_by_interval<double>).indexers();
  const auto prior = model.prior(dataset.features);

  ConditionalGaussian conditional_model(prior, dataset.targets);
  auto brute_force_loo = [&](const auto &group_inds) {
    const auto train_inds = indices_complement(group_inds, prior.size());
    return conditional_model.fit(train_inds).predict(group_inds).joint();
  };

  const auto expected_joints = indexers.apply(brute_force_loo);
  const auto cv_means = model.cross_validate()
                            .predict(dataset, group_by_interval<double>)
                            .means();
  const auto cv_marginals = model.cross_validate()
                                .predict(dataset, group_by_interval<double>)
                                .marginals();
  const auto cv_joints = model.cross_validate()
                             .predict(dataset, group_by_interval<double>)
                             .joints();

  const auto loo_means =
      leave_one_group_out_conditional_means(prior, dataset.targets, indexers);
  const auto loo_marginals = leave_one_group_out_conditional_marginals(
      prior, dataset.targets, indexers);
  const auto loo_joints =
      leave_one_group_out_conditional_joints(prior, dataset.targets, indexers);

  auto mean_near = [](const auto &x, const auto &y) {
    EXPECT_LE((x - y).norm(), 1e-6);
  };

  auto marginal_near = [](const auto &x, const auto &y) {
    EXPECT_LE((x.mean - y.mean).norm(), 1e-6);
    EXPECT_LE((x.covariance.diagonal() - y.covariance.diagonal()).norm(), 1e-6);
  };

  auto joint_near = [](const auto &x, const auto &y) {
    EXPECT_LE((x.mean - y.mean).norm(), 1e-6);
    EXPECT_LE((x.covariance - y.covariance).norm(), 1e-6);
  };

  for (const auto &pair : expected_joints) {
    const auto &key = pair.first;
    const auto &expected = pair.second;
    mean_near(expected.mean, cv_means.at(key));
    mean_near(expected.mean, loo_means.at(key));

    marginal_near(expected.marginal(), cv_marginals.at(key));
    marginal_near(expected.marginal(), loo_marginals.at(key));

    joint_near(expected, cv_joints.at(key));
    joint_near(expected, loo_joints.at(key));
  }
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
TYPED_TEST_SUITE(SpecializedCrossValidationTester, SpecializedModels);

TYPED_TEST(SpecializedCrossValidationTester,
           test_uses_specialized_cross_validation_functions) {
  auto dataset = this->test_case.get_dataset();
  auto model = this->test_case.get_model();

  LeaveOneOutGrouper leave_one_out;
  // time the computation of RMSE using the fast LOO variant.
  using namespace std::chrono;
  high_resolution_clock::time_point start = high_resolution_clock::now();
  const auto cv_fast_scores = model.cross_validate().scores(
      RootMeanSquareError(), dataset, leave_one_out);
  high_resolution_clock::time_point end = high_resolution_clock::now();
  auto fast_duration = duration_cast<microseconds>(end - start).count();

  // time RMSE using the default method.
  const auto folds = folds_from_grouper(dataset, LeaveOneOutGrouper());
  start = high_resolution_clock::now();
  const auto cv_slow_scores =
      model.cross_validate().scores(RootMeanSquareError(), folds);
  end = high_resolution_clock::now();
  auto slow_duration = duration_cast<microseconds>(end - start).count();

  // Make sure the faster variant is actually faster and that the results
  // are the same.
  EXPECT_LT(static_cast<double>(fast_duration),
            0.5 * static_cast<double>(slow_duration));
  EXPECT_NEAR((cv_fast_scores - cv_slow_scores).norm(), 0., 1e-8);
}

} // namespace albatross
