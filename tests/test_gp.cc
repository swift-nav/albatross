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

#include <gtest/gtest.h>

#include "test_utils.h"

namespace albatross {

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
 * errors.  This simply makes sure those errors are properly dealt with.
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

RegressionDataset<double> test_unobservable_dataset() {
  Eigen::Index k = 10;
  Eigen::VectorXd mean = 3.14159 * Eigen::VectorXd::Ones(k);
  Eigen::VectorXd variance = 0.1 * Eigen::VectorXd::Ones(k);
  MarginalDistribution targets(mean, variance.asDiagonal());

  std::vector<double> train_features;
  for (Eigen::Index i = 0; i < k; ++i) {
    train_features.push_back(static_cast<double>(i) * 0.3);
  }

  RegressionDataset<double> dataset(train_features, targets);
  return dataset;
}

auto test_unobservable_model() {
  ConstantEverywhere constant;
  ConstantPerInterval per_interval;
  // First we fit a model directly to the training data and use
  // that to get a prediction of the inducing points.
  auto model = gp_from_covariance(constant + per_interval, "unobservable");
  return model;
}

TEST(test_gp, test_update_model_trait) {
  const auto dataset = test_unobservable_dataset();

  auto model = test_unobservable_model();

  using FitModelType = typename fit_model_type<decltype(model), double>::type;

  using UpdatedFitType = typename updated_fit_type<FitModelType, int>::type;
  using ExpectedType =
      FitModel<decltype(model),
               Fit<GPFit<BlockSymmetric<Eigen::SerializableLDLT>,
                         variant<double, int>>>>;

  EXPECT_TRUE(bool(std::is_same<UpdatedFitType, ExpectedType>::value));
}

TEST(test_gp, test_update_model_same_types) {
  const auto dataset = test_unobservable_dataset();

  std::vector<std::size_t> train_inds = {0, 1, 3, 4, 6, 7, 8, 9};
  std::vector<std::size_t> test_inds = {2, 5};

  const auto train = albatross::subset(dataset, train_inds);
  const auto test = albatross::subset(dataset, test_inds);

  std::vector<std::size_t> first_inds = {0, 1, 2, 3, 5, 7};
  std::vector<std::size_t> second_inds = {4, 6};
  const auto first = albatross::subset(train, first_inds);
  const auto second = albatross::subset(train, second_inds);

  const auto model = test_unobservable_model();

  const auto full_model = model.fit(train);
  const auto full_pred = full_model.predict(test.features).joint();

  const auto first_model = model.fit(first);
  const auto split_model = update(first_model, second);
  const auto split_pred = split_model.predict(test.features).joint();

  // Make sure the fit feature type is a double
  const auto split_fit = split_model.get_fit();
  bool is_double =
      std::is_same<typename decltype(split_fit)::Feature, double>::value;
  EXPECT_TRUE(is_double);

  // Make sure a partial fit, followed by update is the same as a full fit
  EXPECT_TRUE(split_pred.mean.isApprox(full_pred.mean));
  EXPECT_LE((split_pred.covariance - full_pred.covariance).norm(), 1e-6);

  // Make sure a partial fit is not the same as a full fit
  const auto first_pred = first_model.predict(test.features).joint();
  EXPECT_FALSE(split_pred.mean.isApprox(first_pred.mean));
  EXPECT_GE((split_pred.covariance - first_pred.covariance).norm(), 1e-6);
}

TEST(test_gp, test_update_model_different_types) {
  const auto dataset = test_unobservable_dataset();

  const auto model = test_unobservable_model();
  const auto fit_model = model.fit(dataset);

  const auto inducing_points = create_inducing_points(dataset.features);
  MarginalDistribution inducing_prediction =
      fit_model.predict(inducing_points).marginal();

  inducing_prediction.covariance =
      (1e-4 * Eigen::VectorXd::Ones(inducing_prediction.mean.size()))
          .asDiagonal();

  RegressionDataset<InducingFeature> inducing_dataset(inducing_points,
                                                      inducing_prediction);
  const auto new_fit_model = update(fit_model, inducing_dataset);

  // Make sure the new fit with constrained inducing points reproduces
  // the prediction of the constraint
  const auto new_pred = new_fit_model.predict(inducing_points).joint();
  EXPECT_LE((new_pred.mean - inducing_prediction.mean).norm(), 0.01);
  // Without changing the prediction of the training features much
  const auto train_pred = new_fit_model.predict(dataset.features).marginal();
  EXPECT_LE((train_pred.mean - dataset.targets.mean).norm(), 0.1);

  MarginalDistribution perturbed_inducing_targets(inducing_prediction);
  perturbed_inducing_targets.mean +=
      Eigen::VectorXd::Random(perturbed_inducing_targets.mean.size());

  RegressionDataset<InducingFeature> perturbed_dataset(
      inducing_points, perturbed_inducing_targets);
  const auto new_perturbed_model = update(fit_model, perturbed_dataset);
  const auto perturbed_inducing_pred =
      new_perturbed_model.predict(inducing_points).marginal();
  const auto perturbed_train_pred =
      new_perturbed_model.predict(dataset.features).marginal();

  // Make sure constraining to a different value changes the results.
  EXPECT_GE((perturbed_inducing_pred.mean - new_pred.mean).norm(), 1.);
  EXPECT_GE((perturbed_train_pred.mean - train_pred.mean).norm(), 1.);
}

TEST(test_gp, test_model_from_different_datasets) {
  Eigen::Index k = 10;
  Eigen::VectorXd mean = 3.14159 * Eigen::VectorXd::Ones(k);
  Eigen::VectorXd variance = 0.1 * Eigen::VectorXd::Ones(k);
  MarginalDistribution targets(mean, variance.asDiagonal());

  std::vector<double> train_features;
  for (Eigen::Index i = 0; i < k; ++i) {
    train_features.push_back(static_cast<double>(i) * 0.3);
  }

  const auto model = test_unobservable_model();

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

TEST(test_gp, test_model_from_prediction_low_rank) {
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

} // namespace albatross
