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

#ifndef ALBATROSS_CROSSVALIDATION_H
#define ALBATROSS_CROSSVALIDATION_H

#include "core/indexing.h"
#include "core/model.h"
#include <functional>
#include <map>
#include <memory>

namespace albatross {

/*
 * An evaluation metric is a function that takes a prediction distribution and
 * corresponding targets and returns a single real value that summarizes
 * the quality of the prediction.
 */
template <typename PredictType>
using EvaluationMetric = std::function<double(
    const PredictType &prediction, const MarginalDistribution &targets)>;

/*
 * Iterates over previously computed predictions for each fold and
 * returns a vector of scores for each fold.
 */
template <typename FeatureType, typename PredictType = JointDistribution>
static inline Eigen::VectorXd
compute_scores(const EvaluationMetric<PredictType> &metric,
               const std::vector<RegressionFold<FeatureType>> &folds,
               const std::vector<PredictType> &predictions) {
  // Create a vector of metrics, one for each fold.
  Eigen::VectorXd metrics(static_cast<Eigen::Index>(folds.size()));
  // Loop over each fold, making predictions then evaluating them
  // to create the final output.
  for (Eigen::Index i = 0; i < metrics.size(); i++) {
    metrics[i] = metric(predictions[i], folds[i].test_dataset.targets);
  }
  return metrics;
}

template <typename FeatureType, typename CovarianceType>
static inline Eigen::VectorXd
compute_scores(const EvaluationMetric<Eigen::VectorXd> &metric,
               const std::vector<RegressionFold<FeatureType>> &folds,
               const std::vector<Distribution<CovarianceType>> &predictions) {
  std::vector<Eigen::VectorXd> converted;
  for (const auto &pred : predictions) {
    converted.push_back(pred.mean);
  }
  return compute_scores(metric, folds, converted);
}

/*
 * Iterates over each fold in a cross validation set and fits/predicts and
 * scores the fold, returning a vector of scores for each fold.
 */
template <typename FeatureType, typename PredictType = JointDistribution>
static inline Eigen::VectorXd
cross_validated_scores(const EvaluationMetric<PredictType> &metric,
                       const std::vector<RegressionFold<FeatureType>> &folds,
                       RegressionModel<FeatureType> *model) {
  // Create a vector of predictions.
  std::vector<PredictType> predictions =
      model->template cross_validated_predictions<PredictType>(folds);
  return compute_scores<FeatureType, PredictType>(metric, folds, predictions);
}

/*
 * Iterates over each fold in a cross validation set and fits/predicts and
 * scores the fold, returning a vector of scores for each fold.
 */
template <typename FeatureType, typename PredictType = JointDistribution>
static inline Eigen::VectorXd
cross_validated_scores(const EvaluationMetric<PredictType> &metric,
                       const RegressionDataset<FeatureType> &dataset,
                       const FoldIndexer &fold_indexer,
                       RegressionModel<FeatureType> *model) {
  // Create a vector of predictions.
  std::vector<PredictType> predictions =
      model->template cross_validated_predictions<PredictType>(dataset,
                                                               fold_indexer);
  const auto folds = folds_from_fold_indexer(dataset, fold_indexer);
  return compute_scores<FeatureType, PredictType>(metric, folds, predictions);
}

/*
 * Returns a single cross validated prediction distribution
 * for some cross validation folds, taking into account the
 * fact that each fold may contain reordered data.
 */
template <typename FeatureType>
static inline MarginalDistribution concatenate_fold_predictions(
    const std::vector<RegressionFold<FeatureType>> &folds,
    const std::vector<MarginalDistribution> &predictions) {
  // Create a new prediction mean that will eventually contain
  // the ordered concatenation of each fold's predictions.
  Eigen::Index n = 0;
  for (const auto &pred : predictions) {
    n += static_cast<decltype(n)>(pred.size());
  }

  Eigen::VectorXd mean(n);
  Eigen::VectorXd diagonal(n);

  // Put all the predicted means back in order.
  for (std::size_t j = 0; j < predictions.size(); j++) {
    const auto pred = predictions[j];
    const auto fold = folds[j];
    assert(pred.mean.size() == fold.test_dataset.features.size());
    for (Eigen::Index i = 0; i < pred.mean.size(); i++) {
      // The test indices map each element in the current fold back
      // to the original order of the parent dataset.
      auto test_ind = static_cast<Eigen::Index>(fold.test_indices[i]);
      assert(test_ind < n);
      mean[test_ind] = pred.mean[i];
      diagonal[test_ind] = pred.get_diagonal(i);
    }
  }
  return MarginalDistribution(mean, diagonal.asDiagonal());
}

template <typename FeatureType>
static inline MarginalDistribution
cross_validated_predict(const std::vector<RegressionFold<FeatureType>> &folds,
                        RegressionModel<FeatureType> *model) {
  // Get the cross validated predictions, note however that
  // depending on the type of folds, these predictions may
  // be shuffled.
  const std::vector<MarginalDistribution> predictions =
      model->template cross_validated_predictions<MarginalDistribution>(folds);
  return concatenate_fold_predictions(folds, predictions);
}

} // namespace albatross

#endif
