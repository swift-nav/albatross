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
using EvaluationMetric = std::function<double(
    const JointDistribution &prediction, const MarginalDistribution &targets)>;

/*
 * Computes a JointDistribution for each fold in set of cross validation
 * folds.  The resulting vector of JointDistributions can then be used
 * for things like computing an EvaluationMetric for each fold, or assembling
 * all the predictions into a single cross validated PredictionDistribution.
 */
template <typename FeatureType>
static inline std::vector<JointDistribution> cross_validated_predictions(
    const std::vector<RegressionFold<FeatureType>> &folds,
    RegressionModel<FeatureType> *model) {
  // Iteratively make predictions and assemble the output vector
  std::vector<JointDistribution> predictions;
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
compute_scores(const EvaluationMetric &metric,
               const std::vector<RegressionFold<FeatureType>> &folds,
               const std::vector<JointDistribution> &predictions) {
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
  std::vector<JointDistribution> predictions =
      cross_validated_predictions<FeatureType>(folds, model);
  return compute_scores(metric, folds, predictions);
}

/*
 * Returns a single cross validated prediction distribution
 * for some cross validation folds, taking into account the
 * fact that each fold may contain reordered data.
 *
 * Note that the prediction covariance is not returned
 * which is a result of having made predictions one fold at
 * a time, so the full dense prediction covariance is
 * unknown.
 */
template <typename FeatureType>
static inline JointDistribution
cross_validated_predict(const std::vector<RegressionFold<FeatureType>> &folds,
                        RegressionModel<FeatureType> *model) {
  // Get the cross validated predictions, note however that
  // depending on the type of folds, these predictions may
  // be shuffled.
  const std::vector<JointDistribution> predictions =
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
  return JointDistribution(mean);
}

} // namespace albatross

#endif
