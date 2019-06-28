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

#include "test_models.h"

namespace albatross {

TYPED_TEST_P(RegressionModelTester, test_performs_reasonably_on_linear_data) {
  auto dataset = this->test_case.get_dataset();
  auto model = this->test_case.get_model();

  const auto fit_model = model.fit(dataset.features, dataset.targets);
  const auto pred = fit_model.predict(dataset.features);
  const auto pred_mean = pred.mean();

  double rmse = sqrt((pred_mean - dataset.targets.mean).norm());
  EXPECT_LE(rmse, 0.5);
}

Eigen::Index silly_function_to_increment_stack_pointer() {
  Eigen::VectorXd x(10);
  return x.size();
}

TYPED_TEST_P(RegressionModelTester, test_predict_variants) {
  auto dataset = this->test_case.get_dataset();
  auto model = this->test_case.get_model();

  const auto fit_model = model.fit(dataset.features, dataset.targets);
  silly_function_to_increment_stack_pointer();
  const auto pred = fit_model.predict(dataset.features);
  silly_function_to_increment_stack_pointer();

  expect_predict_variants_consistent(pred);
}

TYPED_TEST_P(RegressionModelTester, test_currect_derived_type) {
  const auto model = this->test_case.get_model();
  const auto derived = model.derived();
  bool same = std::is_same<decltype(model), decltype(derived)>::value;
  EXPECT_TRUE(same);
}

REGISTER_TYPED_TEST_CASE_P(RegressionModelTester,
                           test_performs_reasonably_on_linear_data,
                           test_predict_variants, test_currect_derived_type);

INSTANTIATE_TYPED_TEST_CASE_P(test_models, RegressionModelTester,
                              ExampleModels);

class BadModel : public ModelBase<BadModel> {
public:
  Fit<BadModel> _fit_impl(const std::vector<double> &,
                          const MarginalDistribution &) const {
    return {};
  }

  Eigen::VectorXd
  _predict_impl(const std::vector<double> &features, const Fit<BadModel> &,
                const PredictTypeIdentity<Eigen::VectorXd>) const {
    return Eigen::VectorXd::Ones(static_cast<Eigen::Index>(features.size()));
  }

  MarginalDistribution
  _predict_impl(const std::vector<double> &features, const Fit<BadModel> &,
                const PredictTypeIdentity<MarginalDistribution>) const {
    const auto zeros =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(features.size()));
    return MarginalDistribution(zeros, zeros.asDiagonal());
  }
};

TEST(test_models, test_expect_predict_variants_consistent_fails) {
  const auto dataset = make_toy_linear_data();
  BadModel m;
  const auto pred = m.fit(dataset).predict(dataset.features);
  expect_predict_variants_inconsistent(pred);
}

TEST(test_models, test_model_from_prediction) {
  MakeGaussianProcess test_case;
  auto dataset = test_case.get_dataset();
  auto model = test_case.get_model();
  std::vector<double> test_features = {0.1, 1.1, 2.2};
  auto joint_prediction = model.fit(dataset).predict(test_features).joint();
  auto joint_prediction_from_prediction =
      model.fit_from_prediction(test_features, joint_prediction)
          .predict(test_features)
          .joint();
  EXPECT_TRUE(joint_prediction_from_prediction.mean.isApprox(
      joint_prediction.mean, 1e-12));
  EXPECT_TRUE(joint_prediction_from_prediction.covariance.isApprox(
      joint_prediction.covariance, 1e-8));
}

/*
 * In what follows we create a small problem which contains unobservable
 * components.  The model consists of a constant term for each nearest
 * integer, and another constant term for all values.  Ie, measurements
 * from the same integer wide bin will share a bias and all measurements
 * will share another bias.  The result is that the model can't
 * differentiate between the global bias and the average of all integer
 * biases (if you add 1 to the global bias and subtract 1 from all interval
 * biases you end up with the same measurements).  This is handled
 * properly by the direct Gaussian process, but if you first make a
 * prediction of each of the biases, then try to use that prediction to
 * make a new model you end up dealing with a low rank system of
 * equations which if not handled properly can lead to very large
 * errors.  This simply makes sure those errors are properly dealth with.
 */

enum InducingFeatureType { ConstantEverywhereType, ConstantPerIntervalType };

struct InducingFeature {
  InducingFeatureType type;
  long location;
};

std::vector<InducingFeature>
create_inducing_points(const std::vector<double> &features) {

  std::vector<InducingFeature> inducing_points;
  double min = *std::min_element(features.begin(), features.end());
  double max = *std::max_element(features.begin(), features.end());

  InducingFeature everywhere = {ConstantEverywhereType, 0};
  inducing_points.push_back(everywhere);

  InducingFeature interval_feature = {ConstantPerIntervalType, 0};
  long interval = lround(min);
  while (interval <= lround(max)) {
    interval_feature.location = interval;
    inducing_points.push_back(interval_feature);
    interval += 1;
  }

  return inducing_points;
}

class ConstantEverywhere : public CovarianceFunction<ConstantEverywhere> {
public:
  ConstantEverywhere(){};
  ~ConstantEverywhere(){};

  double variance = 10.;

  /*
   * This will create a covariance matrix that looks like,
   *     sigma_mean^2 * ones(m, n)
   * which is saying all observations are perfectly correlated,
   * so you can move one if you move the rest the same amount.
   */
  double _call_impl(const double &x, const double &y) const { return variance; }

  double _call_impl(const InducingFeature &x, const double &y) const {
    if (x.type == ConstantEverywhereType) {
      return variance;
    } else {
      return 0.;
    }
  }

  double _call_impl(const InducingFeature &x, const InducingFeature &y) const {
    if (x.type == ConstantEverywhereType && y.type == ConstantEverywhereType) {
      return variance;
    } else {
      return 0.;
    }
  }
};

class ConstantPerInterval : public CovarianceFunction<ConstantPerInterval> {
public:
  ConstantPerInterval(){};
  ~ConstantPerInterval(){};

  double variance = 5.;

  /*
   * This will create a covariance matrix that looks like,
   *     sigma_mean^2 * ones(m, n)
   * which is saying all observations are perfectly correlated,
   * so you can move one if you move the rest the same amount.
   */
  double _call_impl(const double &x, const double &y) const {
    if (lround(x) == lround(y)) {
      return variance;
    } else {
      return 0.;
    }
  }

  double _call_impl(const InducingFeature &x, const double &y) const {
    if (x.type == ConstantPerIntervalType && x.location == lround(y)) {
      return variance;
    } else {
      return 0.;
    }
  }

  double _call_impl(const InducingFeature &x, const InducingFeature &y) const {
    if (x.type == ConstantPerIntervalType &&
        y.type == ConstantPerIntervalType && x.location == y.location) {
      return variance;
    } else {
      return 0.;
    }
  }
};

TEST(test_models, test_model_from_prediction_low_rank) {
  MakeGaussianProcess test_case;

  Eigen::Index k = 10;
  Eigen::VectorXd mean = 3.14159 * Eigen::VectorXd::Ones(k);
  Eigen::VectorXd variance = 0.1 * Eigen::VectorXd::Ones(k);
  MarginalDistribution targets(mean, variance.asDiagonal());

  std::vector<double> train_features;
  for (Eigen::Index i = 0; i < k; ++i) {
    train_features.push_back(static_cast<double>(i) * 0.3);
  }

  ConstantEverywhere constant;
  ConstantPerInterval per_interval;

  auto model = gp_from_covariance(constant + per_interval, "unobservable");
  const auto fit_model = model.fit(train_features, targets);

  const auto inducing_points = create_inducing_points(train_features);

  auto joint_prediction = fit_model.predict(inducing_points).joint();

  std::vector<double> perturbed_features = {50.01, 51.01, 52.01};

  const auto model_pred = fit_model.predict(perturbed_features).joint();

  auto joint_prediction_from_prediction =
      model.fit_from_prediction(inducing_points, joint_prediction)
          .predict(perturbed_features)
          .joint();

  EXPECT_TRUE(
      joint_prediction_from_prediction.mean.isApprox(model_pred.mean, 1e-12));
  EXPECT_TRUE(joint_prediction_from_prediction.covariance.isApprox(
      model_pred.covariance, 1e-8));
}

TEST(test_models, test_model_from_different_datasets) {
  Eigen::Index k = 10;
  Eigen::VectorXd mean = 3.14159 * Eigen::VectorXd::Ones(k);
  Eigen::VectorXd variance = 0.1 * Eigen::VectorXd::Ones(k);
  MarginalDistribution targets(mean, variance.asDiagonal());

  std::vector<double> train_features;
  for (Eigen::Index i = 0; i < k; ++i) {
    train_features.push_back(static_cast<double>(i) * 0.3);
  }

  ConstantEverywhere constant;
  ConstantPerInterval per_interval;

  // First we fit a model directly to the training data and use
  // that to get a prediction of the inducing points.
  auto model = gp_from_covariance(constant + per_interval, "unobservable");
  const auto fit_model = model.fit(train_features, targets);
  const auto inducing_points = create_inducing_points(train_features);
  MarginalDistribution inducing_prediction =
      fit_model.predict(inducing_points).marginal();

  // Then we create a new model in which the inducing points are
  // constrained to be the same as the previous prediction.
  inducing_prediction.covariance =
      1e-12 * Eigen::VectorXd::Ones(inducing_prediction.size()).asDiagonal();
  RegressionDataset<double> dataset(train_features, targets);
  RegressionDataset<InducingFeature> inducing_dataset(inducing_points,
                                                      inducing_prediction);
  const auto fit_again = model.fit(dataset, inducing_dataset);

  // Then we can make sure that the subsequent constrained predictions are
  // consistent
  const auto pred = fit_again.predict(inducing_points).joint();
  EXPECT_TRUE(inducing_prediction.mean.isApprox(pred.mean));

  const auto train_pred = fit_model.predict(train_features).joint();
  const auto train_pred_again = fit_again.predict(train_features).joint();
  EXPECT_TRUE(train_pred.mean.isApprox(train_pred_again.mean));

  // Now constrain the inducing points to be zero and make sure that
  // messes things up.
  inducing_dataset.targets.mean.fill(0.);
  const auto fit_zero = model.fit(dataset, inducing_dataset);
  const auto pred_zero = fit_zero.predict(inducing_points).joint();

  EXPECT_FALSE(inducing_dataset.targets.mean.isApprox(pred.mean));
  EXPECT_LT((inducing_dataset.targets.mean - pred_zero.mean).norm(), 1e-6);
}

} // namespace albatross
