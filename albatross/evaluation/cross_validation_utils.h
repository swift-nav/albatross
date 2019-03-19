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
  std::vector<Prediction<ModelType, FeatureType, FitType>> predictions;
  for (const auto &fold : folds) {
    predictions.emplace_back(
        model.fit(fold.train_dataset).predict(fold.test_dataset.features));
  }

  return predictions;
}

template <typename PredictType, typename Prediction>
inline auto get_predict_types(
    const std::vector<Prediction> &prediction_classes,
    PredictTypeIdentity<PredictType> = PredictTypeIdentity<PredictType>()) {
  std::vector<PredictType> predictions;
  for (const auto &pred : prediction_classes) {
    predictions.emplace_back(pred.template get<PredictType>());
  }
  return predictions;
}

template <typename PredictionType>
inline std::vector<Eigen::VectorXd>
get_means(const std::vector<PredictionType> &predictions) {
  return get_predict_types<Eigen::VectorXd>(predictions);
}

template <typename PredictionType>
inline std::vector<MarginalDistribution>
get_marginals(const std::vector<PredictionType> &predictions) {
  return get_predict_types<MarginalDistribution>(predictions);
}

template <typename PredictionType>
inline std::vector<JointDistribution>
get_joints(const std::vector<PredictionType> &predictions) {
  return get_predict_types<JointDistribution>(predictions);
}

template <typename FeatureType>
inline Eigen::VectorXd concatenate_mean_predictions(
    const std::vector<RegressionFold<FeatureType>> &folds,
    const std::vector<Eigen::VectorXd> &means) {
  assert(folds.size() == means.size());

  Eigen::Index n = static_cast<Eigen::Index>(dataset_size_from_folds(folds));
  Eigen::VectorXd pred(n);
  Eigen::Index number_filled = 0;
  // Put all the predicted means back in order.
  for (std::size_t i = 0; i < folds.size(); ++i) {
    Eigen::Index fold_size =
        static_cast<Eigen::Index>(folds[i].test_dataset.size());
    assert(means[i].size() == fold_size);
    set_subset(means[i], folds[i].test_indices, &pred);
    number_filled += static_cast<Eigen::Index>(folds[i].test_indices.size());
  }
  assert(number_filled == n);
  return pred;
}

template <typename FeatureType, typename PredType>
inline Eigen::VectorXd concatenate_mean_predictions(
    const std::vector<RegressionFold<FeatureType>> &folds,
    const std::vector<PredType> &predictions) {
  return concatenate_mean_predictions(folds, get_means(predictions));
}

template <typename FeatureType>
inline MarginalDistribution concatenate_marginal_predictions(
    const std::vector<RegressionFold<FeatureType>> &folds,
    const std::vector<MarginalDistribution> &marginals) {
  assert(folds.size() == marginals.size());

  Eigen::Index n = static_cast<Eigen::Index>(dataset_size_from_folds(folds));
  assert(folds.size() == marginals.size());
  Eigen::VectorXd mean(n);
  Eigen::VectorXd variance(n);
  Eigen::Index number_filled = 0;
  // Put all the predicted means back in order.
  for (std::size_t i = 0; i < folds.size(); ++i) {
    assert(marginals[i].size() == folds[i].test_dataset.size());
    set_subset(marginals[i].mean, folds[i].test_indices, &mean);
    set_subset(marginals[i].covariance.diagonal(), folds[i].test_indices,
               &variance);
    number_filled += static_cast<Eigen::Index>(folds[i].test_indices.size());
  }
  assert(number_filled == n);
  return MarginalDistribution(mean, variance.asDiagonal());
}

template <typename FeatureType, typename PredType>
inline MarginalDistribution concatenate_marginal_predictions(
    const std::vector<RegressionFold<FeatureType>> &folds,
    const std::vector<PredType> &predictions) {
  return concatenate_marginal_predictions(folds, get_marginals(predictions));
}

template <typename EvaluationMetricType, typename FeatureType,
          typename PredictionType>
Eigen::VectorXd
cross_validated_scores(const EvaluationMetricType &metric,
                       const std::vector<RegressionFold<FeatureType>> &folds,
                       const std::vector<PredictionType> &predictions) {
  assert(folds.size() == predictions.size());
  Eigen::Index n = static_cast<Eigen::Index>(predictions.size());
  Eigen::VectorXd output(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    auto fold_size = static_cast<std::size_t>(folds[i].test_dataset.size());
    auto pred_size = static_cast<std::size_t>(predictions[i].size());
    assert(fold_size == pred_size);
    output[i] = metric(predictions[i], folds[i].test_dataset.targets);
  }
  return output;
}

} // namespace albatross

#endif
