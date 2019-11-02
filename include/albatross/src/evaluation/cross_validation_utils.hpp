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

template <typename ModelType, typename FeatureType, typename GroupKey>
inline auto
get_predictions(const ModelType &model,
                const RegressionFolds<GroupKey, FeatureType> &folds) {

  const auto predict_group = [&model](const auto &fold) {
    return model.fit(fold.train_dataset).predict(fold.train_dataset.features);
  };

  return folds.apply(predict_group);
}

template <typename PredictType, typename GroupKey, typename Prediction,
          template <typename...> class Map>
inline auto get_predict_types(
    const Map<GroupKey, Prediction> &predictions,
    PredictTypeIdentity<PredictType> = PredictTypeIdentity<PredictType>()) {

  const auto get_predict_type = [](const auto &prediction) {
    return prediction.template get<PredictType>();
  };

  return Grouped<GroupKey, Prediction>(predictions).apply(get_predict_type);
}

template <typename GroupKey, typename PredictionType>
inline auto get_means(const Grouped<GroupKey, PredictionType> &predictions) {
  return get_predict_types<Eigen::VectorXd>(predictions);
}

template <typename GroupKey, typename PredictionType>
inline auto
get_marginals(const Grouped<GroupKey, PredictionType> &predictions) {
  return get_predict_types<MarginalDistribution>(predictions);
}

template <typename GroupKey, typename PredictionType>
inline auto get_joints(const Grouped<GroupKey, PredictionType> &predictions) {
  return get_predict_types<JointDistribution>(predictions);
}

template <typename GroupKey>
inline Eigen::VectorXd
concatenate_mean_predictions(const GroupIndexer<GroupKey> &indexer,
                             const Grouped<GroupKey, Eigen::VectorXd> &means) {
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

template <typename CovarianceType, typename GroupKey,
          template <typename...> class PredictionContainer>
inline MarginalDistribution concatenate_marginal_predictions(
    const GroupIndexer<GroupKey> &indexer,
    const PredictionContainer<GroupKey, Distribution<CovarianceType>> &preds) {
  assert(indexer.size() == preds.size());

  Eigen::Index n =
      static_cast<Eigen::Index>(dataset_size_from_indexer(indexer));
  Eigen::VectorXd mean(n);
  Eigen::VectorXd variance(n);
  Eigen::Index number_filled = 0;
  // Put all the predicted means back in order.
  for (const auto &pair : indexer) {
    assert(preds.at(pair.first).size() == pair.second.size());
    set_subset(preds.at(pair.first).mean, pair.second, &mean);
    set_subset(preds.at(pair.first).covariance.diagonal(), pair.second,
               &variance);
    number_filled += static_cast<Eigen::Index>(pair.second.size());
  }
  assert(number_filled == n);
  return MarginalDistribution(mean, variance.asDiagonal());
}

template <typename PredictionMetricType, typename GroupKey,
          typename FeatureType, typename PredictionType,
          template <typename...> class PredictionContainer>
Eigen::VectorXd cross_validated_scores(
    const PredictionMetricType &metric,
    const RegressionFolds<GroupKey, FeatureType> &folds,
    const PredictionContainer<GroupKey, PredictionType> &predictions) {

  const auto score_one_group = [&](const GroupKey &key,
                                   const RegressionFold<FeatureType> &fold) {
    assert(static_cast<std::size_t>(fold.test_dataset.size()) ==
           static_cast<std::size_t>(predictions.at(key).size()));
    return metric(predictions.at(key), fold.test_dataset.targets);
  };

  return combine(folds.apply(score_one_group));
}

template <typename FeatureType, typename CovarianceType, typename GroupKey>
static inline Eigen::VectorXd cross_validated_scores(
    const PredictionMetric<Eigen::VectorXd> &metric,
    const RegressionFolds<GroupKey, FeatureType> &folds,
    const Grouped<GroupKey, Distribution<CovarianceType>> &predictions) {

  const auto get_mean = [](const auto &pred) { return pred.mean; };

  return cross_validated_scores(metric, folds, predictions.apply(get_mean));
}

} // namespace albatross

#endif
