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

#ifndef ALBATROSS_EVALUATE_H
#define ALBATROSS_EVALUATE_H

#include "core/model.h"
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <functional>
#include <map>
#include <math.h>
#include <memory>

namespace albatross {

/*
 * An evaluation metric is a function that takes a prediction distribution and
 * corresponding targets and returns a single real value that summarizes
 * the quality of the prediction.
 */
using EvaluationMetric = std::function<double(
    const PredictDistribution &prediction, const TargetDistribution &targets)>;

/*
 * Computes the negative log likelihood under the assumption that the predcitve
 * distribution is multivariate normal.
 */
static inline double
negative_log_likelihood(const PredictDistribution &prediction,
                        const TargetDistribution &truth) {

  const auto mean = prediction.mean;
  Eigen::MatrixXd covariance(prediction.covariance);

  if (truth.has_covariance()) {
    covariance += truth.covariance;
  }

  auto llt = covariance.llt();
  auto cholesky = llt.matrixL();
  double det = cholesky.determinant();
  double log_det = log(det);
  Eigen::VectorXd normalized_residuals(truth.size());
  normalized_residuals = cholesky.solve(mean - truth.mean);
  double residuals = normalized_residuals.dot(normalized_residuals);
  return 0.5 *
         (log_det + residuals + static_cast<double>(truth.size()) * 2 * M_PI);
}

static inline double
root_mean_square_error(const PredictDistribution &prediction,
                       const TargetDistribution &truth) {
  const Eigen::VectorXd error = prediction.mean - truth.mean;
  double mse = error.dot(error) / static_cast<double>(error.size());
  return sqrt(mse);
}

/*
 * Takes output from a model (PredictionDistribution)
 * and the corresponding truth and uses them to compute the stddev.
 */
static inline double standard_deviation(const PredictDistribution &prediction,
                                        const TargetDistribution &truth) {
  Eigen::VectorXd error = prediction.mean - truth.mean;
  const auto n_elements = static_cast<double>(error.size());
  const double mean_error = error.sum() / n_elements;
  error.array() -= mean_error;
  return std::sqrt(error.dot(error) / (n_elements - 1));
}

inline FoldIndices get_train_indices(const FoldIndices &test_indices,
                                     const int n) {
  const s32 k = static_cast<s32>(test_indices.size());
  // The train indices are all the indices that are not test indices.
  FoldIndices train_indices(n - k);
  s32 train_cnt = 0;
  for (s32 j = 0; j < n; j++) {
    if (std::find(test_indices.begin(), test_indices.end(), j) ==
        test_indices.end()) {
      train_indices[train_cnt] = j;
      train_cnt++;
    }
  }
  return train_indices;
}

/*
 * Each flavor of cross validation can be described by a set of
 * FoldIndices, which store which indices should be used for the
 * test cases.  This function takes a map from FoldName to
 * FoldIndices and a dataset and creates the resulting folds.
 */
template <typename FeatureType>
static inline std::vector<RegressionFold<FeatureType>>
folds_from_fold_indexer(const RegressionDataset<FeatureType> &dataset,
                        const FoldIndexer &groups) {
  // For a dataset with n features, we'll have n folds.
  const s32 n = static_cast<s32>(dataset.features.size());
  std::vector<RegressionFold<FeatureType>> folds;
  // For each fold, partition into train and test sets.
  for (const auto &pair : groups) {
    // These get exposed inside the returned RegressionFold and because
    // we'd like to prevent modification of the output from this function
    // from changing the input FoldIndexer we perform a copy here.
    const FoldName group_name(pair.first);
    const FoldIndices test_indices(pair.second);
    const auto train_indices = get_train_indices(test_indices, n);

    std::vector<FeatureType> train_features =
        subset(train_indices, dataset.features);
    TargetDistribution train_targets = subset(train_indices, dataset.targets);

    std::vector<FeatureType> test_features =
        subset(test_indices, dataset.features);
    TargetDistribution test_targets = subset(test_indices, dataset.targets);

    assert(train_features.size() == train_targets.size());
    assert(test_features.size() == test_targets.size());
    assert(test_targets.size() + train_targets.size() == n);

    const RegressionDataset<FeatureType> train_split(train_features,
                                                     train_targets);
    const RegressionDataset<FeatureType> test_split(test_features,
                                                    test_targets);
    folds.push_back(RegressionFold<FeatureType>(train_split, test_split,
                                                group_name, test_indices));
  }
  return folds;
}

template <typename FeatureType>
static inline FoldIndexer
leave_one_out_indexer(const RegressionDataset<FeatureType> &dataset) {
  FoldIndexer groups;
  for (s32 i = 0; i < static_cast<s32>(dataset.features.size()); i++) {
    FoldName group_name = std::to_string(i);
    groups[group_name] = {i};
  }
  return groups;
}

/*
 * Splits a dataset into cross validation folds where each fold contains all but
 * one predictor/target pair.
 */
template <typename FeatureType>
static inline FoldIndexer leave_one_group_out_indexer(
    const RegressionDataset<FeatureType> &dataset,
    const std::function<FoldName(const FeatureType &)> &get_group_name) {
  FoldIndexer groups;
  for (s32 i = 0; i < static_cast<s32>(dataset.features.size()); i++) {
    const std::string k =
        get_group_name(dataset.features[static_cast<std::size_t>(i)]);
    // Get the existing indices if we've already encountered this group_name
    // otherwise initialize a new one.
    FoldIndices indices;
    if (groups.find(k) == groups.end()) {
      indices = FoldIndices();
    } else {
      indices = groups[k];
    }
    // Add the current index.
    indices.push_back(i);
    groups[k] = indices;
  }
  return groups;
}

/*
 * Generates cross validation folds which represent leave one out
 * cross validation.
 */
template <typename FeatureType>
static inline std::vector<RegressionFold<FeatureType>>
leave_one_out(const RegressionDataset<FeatureType> &dataset) {
  return folds_from_fold_indexer<FeatureType>(
      dataset, leave_one_out_indexer<FeatureType>(dataset));
}

/*
 * Uses a `get_group_name` function to bucket each FeatureType into
 * a group, then holds out one group at a time.
 */
template <typename FeatureType>
static inline std::vector<RegressionFold<FeatureType>> leave_one_group_out(
    const RegressionDataset<FeatureType> &dataset,
    const std::function<FoldName(const FeatureType &)> &get_group_name) {
  const FoldIndexer indexer =
      leave_one_group_out_indexer<FeatureType>(dataset, get_group_name);
  return folds_from_fold_indexer<FeatureType>(dataset, indexer);
}

/*
 * Computes a PredictDistribution for each fold in set of cross validation
 * folds.  The resulting vector of PredictDistributions can then be used
 * for things like computing an EvaluationMetric for each fold, or assembling
 * all the predictions into a single cross validated PredictionDistribution.
 */
template <typename FeatureType>
static inline std::vector<PredictDistribution> cross_validated_predictions(
    const std::vector<RegressionFold<FeatureType>> &folds,
    RegressionModel<FeatureType> *model) {
  // Iteratively make predictions and assemble the output vector
  std::vector<PredictDistribution> predictions;
  for (std::size_t i = 0; i < folds.size(); i++) {
    predictions.push_back(model->fit_and_predict(
        folds[i].train_dataset.features, folds[i].train_dataset.targets,
        folds[i].test_dataset.features));
  }
  return predictions;
}

/*
 * Iterates over previously computed predictions for each fold and
 * returns a vector of scores for each fold.
 */
template <class FeatureType>
static inline Eigen::VectorXd
compute_scores(const std::vector<RegressionFold<FeatureType>> &folds,
               const EvaluationMetric &metric,
               const std::vector<PredictDistribution> &predictions) {
  // Create a vector of metrics, one for each fold.
  Eigen::VectorXd metrics(static_cast<s32>(folds.size()));
  // Loop over each fold, making predictions then evaluating them
  // to create the final output.
  for (std::size_t i = 0; i < folds.size(); i++) {
    metrics[static_cast<s32>(i)] =
        metric(predictions[i], folds[i].test_dataset.targets);
  }
  return metrics;
}

/*
 * Iterates over each fold in a cross validation set and fits/predicts and
 * scores the fold, returning a vector of scores for each fold.
 */
template <class FeatureType>
static inline Eigen::VectorXd
cross_validated_scores(const EvaluationMetric &metric,
                       const std::vector<RegressionFold<FeatureType>> &folds,
                       RegressionModel<FeatureType> *model) {
  // Create a vector of predictions.
  std::vector<PredictDistribution> predictions =
      cross_validated_predictions<FeatureType>(folds, model);
  return compute_scores(folds, metric, predictions);
}

/*
 * Returns a single cross validated prediction distribution
 * for some cross validation folds, taking into account the
 * fact that each fold may contain reordered data.
 *
 * Note that the prediction covariance is not propagated
 * which is a result of having made predictions one fold at
 * a time, so the full dense prediction covariance is
 * unknown.
 */
template <typename FeatureType>
static inline PredictDistribution
cross_validated_predict(const std::vector<RegressionFold<FeatureType>> &folds,
                        RegressionModel<FeatureType> *model) {
  // Get the cross validated predictions, note however that
  // depending on the type of folds, these predictions may
  // be shuffled.
  const std::vector<PredictDistribution> predictions =
      cross_validated_predictions<FeatureType>(folds, model);
  // Create a new prediction mean that will eventually contain
  // the ordered concatenation of each fold's predictions.
  s32 n = 0;
  for (const auto &pred : predictions) {
    n += static_cast<s32>(pred.mean.size());
  }
  Eigen::VectorXd mean(n);
  // Put all the predicted means back in order.
  for (s32 j = 0; j < static_cast<s32>(predictions.size()); j++) {
    const auto pred = predictions[j];
    const auto fold = folds[j];
    for (s32 i = 0; i < static_cast<s32>(pred.mean.size()); i++) {
      mean[static_cast<s32>(fold.test_indices[static_cast<std::size_t>(i)])] =
          pred.mean[i];
    }
  }
  return PredictDistribution(mean);
}

} // namespace albatross

#endif
