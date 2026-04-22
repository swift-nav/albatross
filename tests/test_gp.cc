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

#include "test_models.h"

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

struct ConstantEverywhereFeature {};

struct ConstantPerIntervalFeature {
  ConstantPerIntervalFeature() : location() {};
  explicit ConstantPerIntervalFeature(const long &location_)
      : location(location_) {};
  long location;
};

using InducingFeature =
    variant<ConstantEverywhereFeature, ConstantPerIntervalFeature>;

std::vector<InducingFeature>
create_inducing_points(const std::vector<double> &features) {

  std::vector<InducingFeature> inducing_points;
  double min = *std::min_element(features.begin(), features.end());
  double max = *std::max_element(features.begin(), features.end());

  ConstantEverywhereFeature everywhere;
  inducing_points.emplace_back(everywhere);

  long interval = lround(min);
  while (interval <= lround(max)) {
    inducing_points.emplace_back(ConstantPerIntervalFeature(interval));
    interval += 1;
  }

  return inducing_points;
}

class ConstantEverywhere : public CovarianceFunction<ConstantEverywhere> {
public:
  ConstantEverywhere() {};
  ~ConstantEverywhere() {};

  double variance = 10.;

  /*
   * This will create a covariance matrix that looks like,
   *     sigma_mean^2 * ones(m, n)
   * which is saying all observations are perfectly correlated,
   * so you can move one if you move the rest the same amount.
   */
  double _call_impl(const double &x ALBATROSS_UNUSED,
                    const double &y ALBATROSS_UNUSED) const {
    return variance;
  }

  double _call_impl(const ConstantEverywhereFeature &x ALBATROSS_UNUSED,
                    const double &y ALBATROSS_UNUSED) const {
    return variance;
  }

  double _call_impl(const ConstantEverywhereFeature &x ALBATROSS_UNUSED,
                    const ConstantEverywhereFeature &y ALBATROSS_UNUSED) const {
    return variance;
  }
};

class ConstantPerInterval : public CovarianceFunction<ConstantPerInterval> {
public:
  ConstantPerInterval() {};
  ~ConstantPerInterval() {};

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

  double _call_impl(const ConstantPerIntervalFeature &x,
                    const double &y) const {
    if (x.location == lround(y)) {
      return variance;
    } else {
      return 0.;
    }
  }

  double _call_impl(const ConstantPerIntervalFeature &x,
                    const ConstantPerIntervalFeature &y) const {
    if (x.location == y.location) {
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
    train_features.push_back(cast::to_double(i) * 0.3);
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
  auto fit_model = model.fit(dataset);

  std::vector<int> int_features;
  for (int i = 0; i < static_cast<int>(dataset.features.size()); ++i) {
    int_features.push_back(i);
  }
  RegressionDataset<int> int_dataset(int_features, dataset.targets);

  auto updated_fit = fit_model.update(int_dataset);

  using UpdatedFitType = decltype(updated_fit);
  using ExpectedType =
      typename updated_fit_model_type<decltype(fit_model), int>::type;

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

TEST(test_gp, test_update_then_prune_same_as_first) {
  const auto dataset = test_unobservable_dataset();

  // Deliberately pick |test|=3, |train|=7, |first|=5, |second|=2 so every
  // size is distinct.  A bug in the post-prune correction path that
  // silently transposes the (|test| x |second|) matrix - or
  // mis-parenthesizes a cwiseProduct against the (|second| x |second|)
  // conditional covariance - would only compile when those dimensions
  // coincide, so keeping them apart turns the bug into a build-time or
  // assertion failure instead of wrong numerics.
  std::vector<std::size_t> train_inds = {0, 1, 3, 4, 6, 7, 9};
  std::vector<std::size_t> test_inds = {2, 5, 8};

  const auto train = albatross::subset(dataset, train_inds);
  const auto test = albatross::subset(dataset, test_inds);

  std::vector<std::size_t> first_inds = {0, 2, 4, 5, 6};
  std::vector<std::size_t> second_inds = {1, 3};
  const auto first = albatross::subset(train, first_inds);
  const auto second = albatross::subset(train, second_inds);

  const auto model = test_unobservable_model();

  const auto first_model = model.fit(first);
  const auto first_pred = first_model.predict(test.features).joint();

  const auto updated = update(first_model, second);
  // After an update, `second` lives at the tail of the concatenated
  // training features; demonstrate composing match_features with the
  // index-based prune instead of open-coding the positions.
  const auto round_tripped = prune(
      updated, match_features(updated.get_fit().train_features, second.features));
  const auto round_tripped_pred = round_tripped.predict(test.features).joint();

  EXPECT_TRUE(round_tripped_pred.mean.isApprox(first_pred.mean));
  EXPECT_LE((round_tripped_pred.covariance - first_pred.covariance).norm(),
            1e-6);
}

TEST(test_gp, test_prune_then_update_same_as_full) {
  const auto dataset = test_unobservable_dataset();

  std::vector<std::size_t> train_inds = {0, 1, 3, 4, 6, 7, 9};
  std::vector<std::size_t> test_inds = {2, 5, 8};

  const auto train = albatross::subset(dataset, train_inds);
  const auto test = albatross::subset(dataset, test_inds);

  std::vector<std::size_t> second_inds = {1, 3};
  const auto second = albatross::subset(train, second_inds);

  const auto model = test_unobservable_model();

  const auto full_model = model.fit(train);
  const auto full_pred = full_model.predict(test.features).joint();

  const auto pruned = prune(full_model, second_inds);
  const auto round_tripped = update(pruned, second);
  const auto round_tripped_pred = round_tripped.predict(test.features).joint();

  EXPECT_TRUE(round_tripped_pred.mean.isApprox(full_pred.mean));
  EXPECT_LE((round_tripped_pred.covariance - full_pred.covariance).norm(),
            1e-6);
}

TEST(test_gp, test_partition_update_against_removed_no_cache) {
  const std::vector<double> train = {0.0, 1.0, 2.0, 3.0};
  const std::vector<std::size_t> remove = {};
  const std::vector<double> update = {1.0, 4.0};

  const auto p = partition_update_against_removed(train, remove, update);
  EXPECT_TRUE(p.reduced_remove_indices.empty());
  EXPECT_EQ(p.truly_new_update_indices, (std::vector<std::size_t>{0, 1}));
}

TEST(test_gp, test_partition_update_against_removed_full_overlap) {
  const std::vector<double> train = {0.0, 1.0, 2.0, 3.0};
  const std::vector<std::size_t> remove = {1, 3};
  const std::vector<double> update = {1.0, 3.0};

  const auto p = partition_update_against_removed(train, remove, update);
  EXPECT_TRUE(p.reduced_remove_indices.empty());
  EXPECT_TRUE(p.truly_new_update_indices.empty());
}

TEST(test_gp, test_partition_update_against_removed_partial_overlap) {
  const std::vector<double> train = {0.0, 1.0, 2.0, 3.0};
  const std::vector<std::size_t> remove = {1, 3};
  const std::vector<double> update = {1.0, 4.0, 5.0};

  const auto p = partition_update_against_removed(train, remove, update);
  EXPECT_EQ(p.reduced_remove_indices, (std::vector<std::size_t>{3}));
  EXPECT_EQ(p.truly_new_update_indices, (std::vector<std::size_t>{1, 2}));
}

TEST(test_gp, test_partition_update_against_removed_different_types) {
  const std::vector<double> train = {0.0, 1.0, 2.0, 3.0};
  const std::vector<std::size_t> remove = {1, 3};
  const std::vector<int> update = {1, 4, 5};

  const auto p = partition_update_against_removed(train, remove, update);
  EXPECT_EQ(p.reduced_remove_indices, remove);
  EXPECT_EQ(p.truly_new_update_indices, (std::vector<std::size_t>{0, 1, 2}));
}

// Round-trip test for a genuine partial-overlap update: some incoming
// features resurrect previously-pruned rows while others are truly new
// and must be folded into the fit.  The post-round-trip predictions
// must match a fresh fit whose training set is exactly
// (original_train \ still_pruned) ∪ truly_new.
TEST(test_gp, test_update_partial_overlap_matches_expected_fit) {
  const auto dataset = test_unobservable_dataset();

  // |train| = 7, |test| = 3.
  const std::vector<std::size_t> train_inds = {0, 1, 3, 4, 6, 7, 9};
  const std::vector<std::size_t> test_inds = {2, 5, 8};

  const auto train = albatross::subset(dataset, train_inds);
  const auto test = albatross::subset(dataset, test_inds);

  // Prune two rows from train.  These are positions 1 and 3 within
  // train, i.e. dataset indices {1, 4}.
  const std::vector<std::size_t> prune_positions = {1, 3};

  // Update with 3 features: dataset[1] overlaps the remove-cache and
  // should be resurrected; dataset[5] and dataset[8] are outside
  // train and need to be folded in.
  const std::vector<std::size_t> update_inds = {1, 5, 8};
  const auto update_data = albatross::subset(dataset, update_inds);

  const auto model = test_unobservable_model();

  const auto trained = model.fit(train);
  const auto pruned_fit = prune(trained, prune_positions);
  const auto round_tripped = update(pruned_fit, update_data);
  const auto round_tripped_pred = round_tripped.predict(test.features).joint();

  // Expected: fit on (train \ {dataset[4]}) ∪ {dataset[5], dataset[8]}
  // = dataset rows {0, 1, 3, 5, 6, 7, 8, 9}.
  const std::vector<std::size_t> expected_inds = {0, 1, 3, 5, 6, 7, 8, 9};
  const auto expected_data = albatross::subset(dataset, expected_inds);
  const auto expected_fit = model.fit(expected_data);
  const auto expected_pred = expected_fit.predict(test.features).joint();

  EXPECT_TRUE(round_tripped_pred.mean.isApprox(expected_pred.mean));
  EXPECT_LE((round_tripped_pred.covariance - expected_pred.covariance).norm(),
            1e-6);
}

// Each of mean / marginal / joint prediction must apply the
// removal correction.  Compare the pruned fit's predictions against
// a fresh fit on the kept subset of the training data.  Sizes are
// intentionally distinct (|test|=3, |train|=7, |second|=2) so that
// any transpose / cwiseProduct shape bug in the correction path
// manifests as an assertion rather than garbage numerics.
namespace {
struct PrunedPredictionFixture {
  RegressionDataset<double> train;
  RegressionDataset<double> test;
  RegressionDataset<double> kept;
  std::vector<std::size_t> second_inds;
};

PrunedPredictionFixture make_pruned_prediction_fixture() {
  const auto dataset = test_unobservable_dataset();
  const std::vector<std::size_t> train_inds = {0, 1, 3, 4, 6, 7, 9};
  const std::vector<std::size_t> test_inds = {2, 5, 8};
  const auto train = albatross::subset(dataset, train_inds);
  const auto test = albatross::subset(dataset, test_inds);
  const std::vector<std::size_t> second_inds = {1, 3};
  const std::vector<std::size_t> kept_positions = {0, 2, 4, 5, 6};
  const auto kept = albatross::subset(train, kept_positions);
  return {train, test, kept, second_inds};
}
} // namespace

TEST(test_gp, test_pruned_mean_matches_refit) {
  const auto f = make_pruned_prediction_fixture();
  const auto model = test_unobservable_model();
  const auto pruned_fit = prune(model.fit(f.train), f.second_inds);
  const auto kept_fit = model.fit(f.kept);

  const Eigen::VectorXd pruned_mean =
      pruned_fit.predict(f.test.features).mean();
  const Eigen::VectorXd kept_mean = kept_fit.predict(f.test.features).mean();

  EXPECT_TRUE(pruned_mean.isApprox(kept_mean));
}

TEST(test_gp, test_pruned_marginal_matches_refit) {
  const auto f = make_pruned_prediction_fixture();
  const auto model = test_unobservable_model();
  const auto pruned_fit = prune(model.fit(f.train), f.second_inds);
  const auto kept_fit = model.fit(f.kept);

  const auto pruned_marg = pruned_fit.predict(f.test.features).marginal();
  const auto kept_marg = kept_fit.predict(f.test.features).marginal();

  EXPECT_TRUE(pruned_marg.mean.isApprox(kept_marg.mean));
  EXPECT_LE(
      (pruned_marg.covariance.diagonal() - kept_marg.covariance.diagonal())
          .norm(),
      1e-6);
}

TEST(test_gp, test_pruned_joint_matches_refit) {
  const auto f = make_pruned_prediction_fixture();
  const auto model = test_unobservable_model();
  const auto pruned_fit = prune(model.fit(f.train), f.second_inds);
  const auto kept_fit = model.fit(f.kept);

  const auto pruned_joint = pruned_fit.predict(f.test.features).joint();
  const auto kept_joint = kept_fit.predict(f.test.features).joint();

  EXPECT_TRUE(pruned_joint.mean.isApprox(kept_joint.mean));
  EXPECT_LE((pruned_joint.covariance - kept_joint.covariance).norm(), 1e-6);
}

// Pruning every training feature should leave the posterior equal
// to the GP's prior at the query locations: the correction exactly
// cancels the "explained" components of mean and covariance, so the
// mean drops to mean_function_(test) (zero here) and the covariance
// reduces to covariance_function_(test).  Covers mean / marginal /
// joint predictions.
TEST(test_gp, test_prune_all_features_gives_prior) {
  const auto dataset = test_unobservable_dataset();
  const std::vector<double> test_features = {0.15, 0.9, 1.65, 2.4};

  const auto model = test_unobservable_model();
  const auto full_fit = model.fit(dataset);
  std::vector<std::size_t> all_inds(dataset.features.size());
  std::iota(all_inds.begin(), all_inds.end(), std::size_t{0});
  const auto all_pruned = prune(full_fit, all_inds);
  ASSERT_EQ(all_pruned.get_fit().removed.indices.size(),
            dataset.features.size());

  // The mean function is ZeroMean, so the expected posterior mean
  // is exactly zero; use an absolute tolerance since isApprox's
  // relative tolerance degenerates when the reference is zero.
  const Eigen::MatrixXd expected_cov = model.compute_covariance(test_features);

  const auto joint = all_pruned.predict(test_features).joint();
  EXPECT_TRUE(joint.mean.isZero(1e-8));
  EXPECT_TRUE(joint.covariance.isApprox(expected_cov));

  const auto marginal = all_pruned.predict(test_features).marginal();
  EXPECT_TRUE(marginal.mean.isZero(1e-8));
  EXPECT_TRUE(marginal.covariance.diagonal().isApprox(expected_cov.diagonal()));

  const Eigen::VectorXd mean_pred = all_pruned.predict(test_features).mean();
  EXPECT_TRUE(mean_pred.isZero(1e-8));
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

  ConstantPerInterval cov;

  // Make sure the new fit with constrained inducing points reproduces
  // the prediction of the constraint
  const auto new_pred = new_fit_model.predict(inducing_points).joint();
  EXPECT_LE((new_pred.mean - inducing_prediction.mean).norm(), 0.01);
  // Without changing the prediction of the training features much
  const auto train_pred = new_fit_model.predict(dataset.features).marginal();
  EXPECT_LE((train_pred.mean - dataset.targets.mean).norm(), 0.1);

  MarginalDistribution perturbed_inducing_targets(inducing_prediction);
  perturbed_inducing_targets.mean +=
      Eigen::VectorXd::Ones(perturbed_inducing_targets.mean.size());

  RegressionDataset<InducingFeature> perturbed_dataset(
      inducing_points, perturbed_inducing_targets);
  const auto new_perturbed_model = update(fit_model, perturbed_dataset);
  const auto perturbed_inducing_pred =
      new_perturbed_model.predict(inducing_points).marginal();
  const auto perturbed_train_pred =
      new_perturbed_model.predict(dataset.features).marginal();

  // Make sure constraining to a different value changes the results.
  EXPECT_GE((perturbed_inducing_pred.mean - new_pred.mean).norm(), 0.5);
  EXPECT_GE((perturbed_train_pred.mean - train_pred.mean).norm(), 0.5);
}

TEST(test_gp, test_model_from_different_datasets) {
  const auto unobservable_dataset = test_unobservable_dataset();

  const auto model = test_unobservable_model();

  const auto fit_model = model.fit(unobservable_dataset);
  const auto inducing_points =
      create_inducing_points(unobservable_dataset.features);
  MarginalDistribution inducing_prediction =
      fit_model.predict(inducing_points).marginal();

  // Then we create a new model in which the inducing points are
  // constrained to be the same as the previous prediction.
  inducing_prediction.covariance =
      1e-12 * Eigen::VectorXd::Ones(cast::to_index(inducing_prediction.size()))
                  .asDiagonal();
  RegressionDataset<double> dataset(unobservable_dataset);
  RegressionDataset<InducingFeature> inducing_dataset(inducing_points,
                                                      inducing_prediction);
  const auto fit_again = model.fit(dataset, inducing_dataset);

  // Then we can make sure that the subsequent constrained predictions are
  // consistent
  const auto pred = fit_again.predict(inducing_points).joint();
  EXPECT_TRUE(inducing_prediction.mean.isApprox(pred.mean));

  const auto train_pred =
      fit_model.predict(unobservable_dataset.features).joint();
  const auto train_pred_again =
      fit_again.predict(unobservable_dataset.features).joint();
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
    train_features.push_back(cast::to_double(i) * 0.3);
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

TEST(test_gp, test_unobservablemodel_with_sum_constraint) {

  const auto dataset = test_unobservable_dataset();
  const auto model = test_unobservable_model();

  const auto inducing_points = create_inducing_points(dataset.features);

  std::vector<ConstantPerIntervalFeature> interval_features;
  for (const auto &f : inducing_points) {
    if (f.is<ConstantPerIntervalFeature>()) {
      interval_features.emplace_back(f.get<ConstantPerIntervalFeature>());
    }
  }

  LinearCombination<ConstantPerIntervalFeature> sums(interval_features);

  Eigen::VectorXd mean = Eigen::VectorXd::Zero(1);
  Eigen::VectorXd variance = 1e-5 * Eigen::VectorXd::Ones(1);
  MarginalDistribution targets(mean, variance.asDiagonal());
  RegressionDataset<LinearCombination<ConstantPerIntervalFeature>> sum_dataset(
      {sums}, targets);

  const auto both = concatenate_datasets(dataset, sum_dataset);

  const auto fit_model = model.fit(both);

  const auto pred = fit_model.predict(interval_features).joint();

  const auto ones = Eigen::VectorXd::Ones(pred.mean.size());
  EXPECT_NEAR(ones.dot(pred.mean), 0., 1e-6);
  EXPECT_NEAR(ones.dot(pred.covariance * ones), 0., 1e-5);
}

TEST(test_gp, test_unobservablemodel_with_diff_constraint) {

  const auto dataset = test_unobservable_dataset();
  const auto model = test_unobservable_model();

  const auto inducing_points = create_inducing_points(dataset.features);

  std::vector<ConstantPerIntervalFeature> interval_features;
  for (const auto &f : inducing_points) {
    if (f.is<ConstantPerIntervalFeature>()) {
      interval_features.emplace_back(f.get<ConstantPerIntervalFeature>());
    }
  }

  std::vector<ConstantPerIntervalFeature> diff_features = {
      interval_features[0], interval_features[1]};

  Eigen::Vector2d diff_coefs;
  diff_coefs << 1, -1;

  LinearCombination<ConstantPerIntervalFeature> difference(diff_features,
                                                           diff_coefs);

  Eigen::VectorXd mean = Eigen::VectorXd::Zero(1);
  Eigen::VectorXd variance = 1e-5 * Eigen::VectorXd::Ones(1);
  MarginalDistribution targets(mean, variance.asDiagonal());
  RegressionDataset<LinearCombination<ConstantPerIntervalFeature>> diff_dataset(
      {difference}, targets);

  const auto both = concatenate_datasets(dataset, diff_dataset);

  const auto fit_model = model.fit(both);

  const auto pred = fit_model.predict(diff_features).joint();

  EXPECT_NEAR(diff_coefs.dot(pred.mean), 0., 1e-6);
  EXPECT_NEAR(diff_coefs.dot(pred.covariance * diff_coefs), 0., 1e-5);
}

TEST(test_gp, test_nonzero_mean) {

  MakeGaussianProcessWithMean gp_with_mean_case;

  const auto dataset = gp_with_mean_case.get_dataset();
  const auto model = gp_with_mean_case.get_model();

  const auto fit_model = model.fit(dataset);

  const auto covariance_function = model.get_covariance();
  const auto measurement_features = as_measurements(dataset.features);
  const auto train_cov = covariance_function(measurement_features);
  const auto train_ldlt = train_cov.ldlt();
  const Eigen::VectorXd information = train_ldlt.solve(dataset.targets.mean);

  const std::vector<double> prediction_features = {-20., 0.01};

  const auto cross = covariance_function(dataset.features, prediction_features);
  const auto prior = covariance_function(prediction_features);
  const auto pred_without_mean =
      gp_joint_prediction(cross, prior, information, train_ldlt);

  const auto actual = fit_model.predict(prediction_features).joint();

  // The prediction without using the mean should be very different.
  EXPECT_GT((pred_without_mean.mean - actual.mean).norm(), 1.);
}

TEST(test_gp, test_get_prior) {

  MakeGaussianProcessWithMean gp_with_mean_case;

  const auto dataset = gp_with_mean_case.get_dataset();
  const auto model = gp_with_mean_case.get_model();

  const auto prior = model.prior(dataset.features);

  const auto cov = model.get_covariance();
  EXPECT_EQ(prior.covariance, cov(as_measurements(dataset.features)));

  const auto mean = model.get_mean()(as_measurements(dataset.features));
  EXPECT_EQ(prior.mean, mean);
}

TEST(test_gp, test_predict_with_complicated_feature) {

  MakeGaussianProcessWithMean gp_with_mean_case;

  const auto dataset = gp_with_mean_case.get_dataset();
  const auto model = gp_with_mean_case.get_model();

  using ComplexType =
      variant<double, Eigen::VectorXd,
              LinearCombination<variant<double, Eigen::VectorXd>>>;

  const auto prior = model.prior(dataset.features);

  std::vector<variant<double, Eigen::VectorXd>> vec = {1.0, 2.0};
  LinearCombination<variant<double, Eigen::VectorXd>> sum(vec);
  std::vector<ComplexType> features = {1.0, 2.0, sum};

  const auto pred = model.fit(dataset).predict(features).joint();

  Eigen::Vector3d for_sum;
  for_sum << 1., 1., 0;

  const double expected_mean = for_sum.transpose() * pred.mean;
  const double expected_variance =
      for_sum.transpose() * pred.covariance * for_sum;

  EXPECT_NEAR(expected_mean, pred.mean[2], 1e-6);
  EXPECT_NEAR(expected_variance, pred.covariance(2, 2), 1e-6);
}

} // namespace albatross
