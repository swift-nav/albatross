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

#ifndef ALBATROSS_CROSS_VALIDATION_UTILS_H
#define ALBATROSS_CROSS_VALIDATION_UTILS_H

namespace albatross {

template <typename ModelType, typename FeatureType>
inline auto
get_predictions(const ModelType &model,
                const std::vector<RegressionFold<FeatureType>> &folds) {

  using FitType = typename fit_type<ModelType, FeatureType>::type;
  std::map<std::string, Prediction<ModelType, FeatureType, FitType>>
      predictions;
  for (const auto &fold : folds) {
    predictions.emplace(
        fold.name,
        model.fit(fold.train_dataset).predict(fold.test_dataset.features));
  }

  return predictions;
}

template <typename PredictType, typename Prediction>
inline auto get_predict_types(
    const std::map<std::string, Prediction> &prediction_classes,
    PredictTypeIdentity<PredictType> = PredictTypeIdentity<PredictType>()) {
  std::map<std::string, PredictType> predictions;
  for (const auto &pred : prediction_classes) {
    predictions.emplace(pred.first, pred.second.template get<PredictType>());
  }
  return predictions;
}

template <typename PredictionType>
inline std::map<std::string, Eigen::VectorXd>
get_means(const std::map<std::string, PredictionType> &predictions) {
  return get_predict_types<Eigen::VectorXd>(predictions);
}

template <typename PredictionType>
inline std::map<std::string, MarginalDistribution>
get_marginals(const std::map<std::string, PredictionType> &predictions) {
  return get_predict_types<MarginalDistribution>(predictions);
}

template <typename PredictionType>
inline std::map<std::string, JointDistribution>
get_joints(const std::map<std::string, PredictionType> &predictions) {
  return get_predict_types<JointDistribution>(predictions);
}

inline Eigen::VectorXd concatenate_mean_predictions(
    const FoldIndexer &indexer,
    const std::map<std::string, Eigen::VectorXd> &means) {
  assert(indexer.size() == means.size());

  Eigen::Index n =
      static_cast<Eigen::Index>(dataset_size_from_indexer(indexer));
  Eigen::VectorXd pred(n);
  Eigen::Index number_filled = 0;
  // Put all the predicted means back in order.
  for (const auto &pair : indexer) {
    assert(means.at(pair.first).size() ==
           static_cast<Eigen::Index>(pair.second.size()));
    set_subset(means.at(pair.first), pair.second, &pred);
    number_filled += static_cast<Eigen::Index>(pair.second.size());
  }
  assert(number_filled == n);
  return pred;
}

inline MarginalDistribution concatenate_marginal_predictions(
    const FoldIndexer &indexer,
    const std::map<std::string, MarginalDistribution> &marginals) {
  assert(indexer.size() == marginals.size());

  Eigen::Index n =
      static_cast<Eigen::Index>(dataset_size_from_indexer(indexer));
  Eigen::VectorXd mean(n);
  Eigen::VectorXd variance(n);
  Eigen::Index number_filled = 0;
  // Put all the predicted means back in order.
  for (const auto &pair : indexer) {
    assert(marginals.at(pair.first).size() == pair.second.size());
    set_subset(marginals.at(pair.first).mean, pair.second, &mean);
    set_subset(marginals.at(pair.first).covariance.diagonal(), pair.second,
               &variance);
    number_filled += static_cast<Eigen::Index>(pair.second.size());
  }
  assert(number_filled == n);
  return MarginalDistribution(mean, variance.asDiagonal());
}

template <typename EvaluationMetricType, typename FeatureType,
          typename PredictionType>
Eigen::VectorXd cross_validated_scores(
    const EvaluationMetricType &metric,
    const std::vector<RegressionFold<FeatureType>> &folds,
    const std::map<std::string, PredictionType> &predictions) {
  assert(folds.size() == predictions.size());
  Eigen::Index n = static_cast<Eigen::Index>(predictions.size());
  Eigen::VectorXd output(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    assert(static_cast<std::size_t>(folds[i].test_dataset.size()) ==
           static_cast<std::size_t>(predictions.at(folds[i].name).size()));
    output[i] =
        metric(predictions.at(folds[i].name), folds[i].test_dataset.targets);
  }
  return output;
}

template <typename FeatureType, typename CovarianceType>
static inline Eigen::VectorXd cross_validated_scores(
    const EvaluationMetric<Eigen::VectorXd> &metric,
    const std::vector<RegressionFold<FeatureType>> &folds,
    const std::map<std::string, Distribution<CovarianceType>> &predictions) {
  std::map<std::string, Eigen::VectorXd> converted;
  for (const auto &pred : predictions) {
    converted[pred.first] = pred.second.mean;
  }
  return cross_validated_scores(metric, folds, converted);
}

} // namespace albatross

#endif
