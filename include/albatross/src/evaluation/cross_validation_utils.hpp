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
  ALBATROSS_ASSERT(indexer.size() == means.size());

  Eigen::Index n = cast::to_index(dataset_size_from_indexer(indexer));
  Eigen::VectorXd pred(n);
  Eigen::Index number_filled = 0;
  // Put all the predicted means back in order.
  for (const auto &pair : indexer) {
    ALBATROSS_ASSERT(means.at(pair.first).size() ==
                     cast::to_index(pair.second.size()));
    set_subset(means.at(pair.first), pair.second, &pred);
    number_filled += cast::to_index(pair.second.size());
  }
  ALBATROSS_ASSERT(number_filled == n);
  return pred;
}

template <typename DistributionType, typename GroupKey,
          template <typename...> class PredictionContainer>
inline MarginalDistribution concatenate_marginal_predictions(
    const GroupIndexer<GroupKey> &indexer,
    const PredictionContainer<GroupKey, DistributionType> &preds) {
  ALBATROSS_ASSERT(indexer.size() == preds.size());

  Eigen::Index n = cast::to_index(dataset_size_from_indexer(indexer));
  Eigen::VectorXd mean(n);
  Eigen::VectorXd variance(n);
  Eigen::Index number_filled = 0;
  // Put all the predicted means back in order.
  for (const auto &pair : indexer) {
    ALBATROSS_ASSERT(preds.at(pair.first).size() == pair.second.size());
    set_subset(preds.at(pair.first).mean, pair.second, &mean);
    set_subset(preds.at(pair.first).covariance.diagonal(), pair.second,
               &variance);
    number_filled += cast::to_index(pair.second.size());
  }
  ALBATROSS_ASSERT(number_filled == n);
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
    ALBATROSS_ASSERT(static_cast<size_t>(fold.test_dataset.size()) ==
                     static_cast<size_t>(predictions.at(key).size()));
    return metric(predictions.at(key), fold.test_dataset.targets);
  };

  return combine(folds.apply(score_one_group));
}

template <typename FeatureType, typename DistributionType, typename GroupKey,
          std::enable_if_t<is_distribution<DistributionType>::value, int> = 0>
static inline Eigen::VectorXd
cross_validated_scores(const PredictionMetric<Eigen::VectorXd> &metric,
                       const RegressionFolds<GroupKey, FeatureType> &folds,
                       const Grouped<GroupKey, DistributionType> &predictions) {

  const auto get_mean = [](const auto &pred) { return pred.mean; };

  return cross_validated_scores(metric, folds, predictions.apply(get_mean));
}

inline Eigen::VectorXd leave_one_out_conditional_variance(
    const Eigen::SerializableLDLT &covariance_ldlt) {
  // The leave one out variance will be the inverse of the diagonal of the
  // inverse of covariance (that's a mouthful!) For details see Equation 5.12 of
  // Gaussian Processes for Machine Learning
  return covariance_ldlt.inverse_diagonal().array().inverse();
}

inline Eigen::VectorXd
leave_one_out_conditional_variance(const Eigen::MatrixXd &covariance) {
  return leave_one_out_conditional_variance(
      Eigen::SerializableLDLT(covariance));
}

inline MarginalDistribution
leave_one_out_conditional(const JointDistribution &prior,
                          const MarginalDistribution &truth) {
  // Computes the conditional distribution of each variable conditional on
  // all others.
  //
  // For details see Equation 5.12 of Gaussian Processes for Machine Learning

  Eigen::MatrixXd covariance = prior.covariance;
  covariance += truth.covariance;
  Eigen::SerializableLDLT ldlt(covariance.ldlt());
  const Eigen::VectorXd loo_variance = leave_one_out_conditional_variance(ldlt);
  const Eigen::VectorXd deviation = truth.mean - prior.mean;
  Eigen::VectorXd loo_mean = ldlt.solve(deviation);
  loo_mean.array() *= loo_variance.array();
  loo_mean = truth.mean - loo_mean;
  return MarginalDistribution(loo_mean, loo_variance);
}

namespace details {

// The following methods implement leave one group out cross validation
// for more details see:
//
// https://swiftnav-albatross.readthedocs.io/en/latest/gp-details.html

inline Eigen::VectorXd
held_out_prediction(const Eigen::MatrixXd &inverse_block,
                    const Eigen::VectorXd &y, const Eigen::VectorXd &v,
                    PredictTypeIdentity<Eigen::VectorXd>) {
  return y - inverse_block.ldlt().solve(v);
}

inline MarginalDistribution
held_out_prediction(const Eigen::MatrixXd &inverse_block,
                    const Eigen::VectorXd &y, const Eigen::VectorXd &v,
                    PredictTypeIdentity<MarginalDistribution>) {
  const auto A_ldlt = Eigen::SerializableLDLT(inverse_block);
  const Eigen::VectorXd mean = y - A_ldlt.solve(v);
  return MarginalDistribution(mean, A_ldlt.inverse_diagonal());
}

inline JointDistribution
held_out_prediction(const Eigen::MatrixXd &inverse_block,
                    const Eigen::VectorXd &y, const Eigen::VectorXd &v,
                    PredictTypeIdentity<JointDistribution>) {
  const auto A_inv = inverse_block.inverse();
  const Eigen::VectorXd mean = y - A_inv * v;
  return JointDistribution(mean, A_inv);
}

template <typename GroupKey, typename PredictType>
inline std::map<GroupKey, PredictType>
held_out_predictions(const Eigen::SerializableLDLT &covariance,
                     const Eigen::VectorXd &target_mean,
                     const Eigen::VectorXd &information,
                     const GroupIndexer<GroupKey> &group_indexer,
                     PredictTypeIdentity<PredictType> predict_type) {

  const std::vector<GroupIndices> indices = map_values(group_indexer);
  const std::vector<GroupKey> group_keys = map_keys(group_indexer);
  const auto inverse_blocks = covariance.inverse_blocks(indices);

  std::map<GroupKey, PredictType> output;
  for (std::size_t i = 0; i < inverse_blocks.size(); i++) {
    const Eigen::VectorXd yi = subset(target_mean, indices[i]);
    const Eigen::VectorXd vi = subset(information, indices[i]);
    output[group_keys[i]] =
        held_out_prediction(inverse_blocks[i], yi, vi, predict_type);
  }
  return output;
}

template <typename GroupKey, typename PredictType>
inline std::map<GroupKey, PredictType>
leave_one_group_out_conditional(const JointDistribution &prior,
                                const MarginalDistribution &truth,
                                const GroupIndexer<GroupKey> &group_indexer,
                                PredictTypeIdentity<PredictType> predict_type) {
  Eigen::MatrixXd covariance = prior.covariance;
  covariance += truth.covariance;
  Eigen::SerializableLDLT ldlt(covariance);
  const Eigen::VectorXd deviation = truth.mean - prior.mean;
  const Eigen::VectorXd information = ldlt.solve(deviation);
  return held_out_predictions(covariance, truth.mean, information,
                              group_indexer, predict_type);
}

} // namespace details

template <typename GroupKey>
inline std::map<GroupKey, Eigen::VectorXd>
leave_one_group_out_conditional_means(
    const JointDistribution &prior, const MarginalDistribution &truth,
    const GroupIndexer<GroupKey> &group_indexer) {
  return details::leave_one_group_out_conditional(
      prior, truth, group_indexer, PredictTypeIdentity<Eigen::VectorXd>());
}

template <typename GroupKey>
inline std::map<GroupKey, MarginalDistribution>
leave_one_group_out_conditional_marginals(
    const JointDistribution &prior, const MarginalDistribution &truth,
    const GroupIndexer<GroupKey> &group_indexer) {
  return details::leave_one_group_out_conditional(
      prior, truth, group_indexer, PredictTypeIdentity<MarginalDistribution>());
}

template <typename GroupKey>
inline std::map<GroupKey, JointDistribution>
leave_one_group_out_conditional_joints(
    const JointDistribution &prior, const MarginalDistribution &truth,
    const GroupIndexer<GroupKey> &group_indexer) {
  return details::leave_one_group_out_conditional(
      prior, truth, group_indexer, PredictTypeIdentity<JointDistribution>());
}

} // namespace albatross

#endif
