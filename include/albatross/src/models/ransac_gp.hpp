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

#ifndef ALBATROSS_MODELS_RANSAC_GP_H_
#define ALBATROSS_MODELS_RANSAC_GP_H_

namespace albatross {

template <typename ModelType, typename FeatureType> struct FitAndIndices {
  using FitType = typename fit_type<ModelType, FeatureType>::type;

  FitType fit;
  GroupIndices indices;
};

template <typename ModelType, typename FeatureType, typename GroupKey>
inline typename RansacFunctions<FitAndIndices<ModelType, FeatureType>,
                                GroupKey>::FitterFunc
get_gp_ransac_fitter(const RegressionDataset<FeatureType> &dataset,
                     const GroupIndexer<GroupKey> &indexer,
                     const Eigen::MatrixXd &cov) {
  return [&, indexer, cov, dataset](const std::vector<GroupKey> &groups) {
    auto inds = indices_from_groups(indexer, groups);
    const auto train_dataset = subset(dataset, inds);
    const auto train_cov = symmetric_subset(cov, inds);

    using GPFitType = typename FitAndIndices<ModelType, FeatureType>::FitType;
    const GPFitType fit(train_dataset.features, train_cov,
                        train_dataset.targets);
    const FitAndIndices<ModelType, FeatureType> fit_and_indices = {fit, inds};
    return fit_and_indices;
  };
}

template <typename ModelType, typename FeatureType,
          typename IsValidCandidateMetric, typename GroupKey>
inline typename RansacFunctions<FitAndIndices<ModelType, FeatureType>,
                                GroupKey>::IsValidCandidate
get_gp_ransac_is_valid_candidate(const RegressionDataset<FeatureType> &dataset,
                                 const GroupIndexer<GroupKey> &indexer,
                                 const Eigen::MatrixXd &cov,
                                 const IsValidCandidateMetric &metric) {

  return [&, indexer, cov, dataset](const std::vector<GroupKey> &groups) {
    const auto inds = indices_from_groups(indexer, groups);
    return metric(inds, dataset, cov);
  };
}

template <typename ModelType, typename FeatureType, typename InlierMetricType,
          typename GroupKey>
inline typename RansacFunctions<FitAndIndices<ModelType, FeatureType>,
                                GroupKey>::InlierMetric
get_gp_ransac_inlier_metric(const RegressionDataset<FeatureType> &dataset,
                            const GroupIndexer<GroupKey> &indexer,
                            const Eigen::MatrixXd &cov, const ModelType &model,
                            const InlierMetricType &metric) {

  return [&, indexer, cov, model, dataset](
             const GroupKey &group,
             const FitAndIndices<ModelType, FeatureType> &fit_and_indices) {
    const auto inds = indexer.at(group);
    const auto test_dataset = subset(dataset, inds);

    const auto pred =
        get_prediction(model, fit_and_indices.fit, test_dataset.features);
    return metric(pred, test_dataset.targets);
  };
}

template <typename ModelType, typename FeatureType, typename ConsensusMetric,
          typename GroupKey>
inline typename RansacFunctions<FitAndIndices<ModelType, FeatureType>,
                                GroupKey>::ConsensusMetric
get_gp_ransac_consensus_metric(const GroupIndexer<GroupKey> &indexer,
                               const RegressionDataset<FeatureType> &dataset,
                               const Eigen::MatrixXd &cov,
                               const ConsensusMetric &metric) {

  return [&, indexer, cov, dataset](const std::vector<GroupKey> &groups) {
    const auto inds = indices_from_groups(indexer, groups);
    ;
    return metric(inds, dataset.targets, cov);
  };
}

struct DifferentialEntropyConsensusMetric {
  double operator()(const GroupIndices &indices, const MarginalDistribution &,
                    const Eigen::MatrixXd &cov) const {
    const auto consensus_cov = symmetric_subset(cov, indices);
    return differential_entropy(consensus_cov);
  }
};

struct FeatureCountConsensusMetric {

  double operator()(const GroupIndices &indices, const MarginalDistribution &,
                    const Eigen::MatrixXd &) const {
    // Negative because a lower metric is better.
    return (-1.0 * static_cast<double>(indices.size()));
  }
};

struct ChiSquaredConsensusMetric {
  double operator()(const GroupIndices &indices,
                    const MarginalDistribution &targets,
                    const Eigen::MatrixXd &cov) const {
    const auto consensus_prior = symmetric_subset(cov, indices);
    const auto consensus_targets = subset(targets, indices);
    return chi_squared_cdf(consensus_targets.mean, consensus_prior);
  }
};

struct ChiSquaredIsValidCandidateMetric {

  template <typename FeatureType>
  bool operator()(const GroupIndices &inds,
                  const RegressionDataset<FeatureType> &dataset,
                  const Eigen::MatrixXd &cov) const {
    const auto train_dataset = subset(dataset, inds);
    const auto train_cov = symmetric_subset(cov, inds);

    const JointDistribution prior(Eigen::VectorXd::Zero(train_cov.rows()),
                                  train_cov);
    // These thresholds are under the assumption of a perfectly
    // representative prior.
    const double probability_prior_exceeded =
        chi_squared_cdf(prior, train_dataset.targets);
    const double skip_every_1000th_candidate = 0.999;
    return (probability_prior_exceeded < skip_every_1000th_candidate);
  };
};

struct AlwaysAcceptCandidateMetric {
  template <typename FeatureType>
  bool operator()(const GroupIndices &inds,
                  const RegressionDataset<FeatureType> &dataset,
                  const Eigen::MatrixXd &cov) const {
    return true;
  }
};

template <typename ModelType, typename FeatureType, typename InlierMetric,
          typename ConsensusMetric, typename IsValidCandidateMetric,
          typename GroupKey>
inline RansacFunctions<FitAndIndices<ModelType, FeatureType>, GroupKey>
get_gp_ransac_functions(
    const ModelType &model,
    const RegressionDataset<FeatureType> &dataset_with_mean,
    const GroupIndexer<GroupKey> &indexer, const InlierMetric &inlier_metric,
    const ConsensusMetric &consensus_metric,
    const IsValidCandidateMetric &is_valid_candidate_metric) {

  static_assert(is_prediction_metric<InlierMetric>::value,
                "InlierMetric must be an PredictionMetric.");

  RegressionDataset<FeatureType> dataset(dataset_with_mean);
  model.get_mean().remove_from(dataset.features, &dataset.targets.mean);
  const auto full_cov = model.compute_covariance(dataset.features);

  const auto fitter = get_gp_ransac_fitter<ModelType, FeatureType, GroupKey>(
      dataset, indexer, full_cov);

  const auto inlier_metric_from_group =
      get_gp_ransac_inlier_metric<ModelType, FeatureType, InlierMetric,
                                  GroupKey>(dataset, indexer, full_cov, model,
                                            inlier_metric);

  const auto consensus_metric_from_groups =
      get_gp_ransac_consensus_metric<ModelType, FeatureType, ConsensusMetric,
                                     GroupKey>(indexer, dataset, full_cov,
                                               consensus_metric);

  const auto is_valid_candidate =
      get_gp_ransac_is_valid_candidate<ModelType, FeatureType>(
          dataset, indexer, full_cov, is_valid_candidate_metric);

  return RansacFunctions<FitAndIndices<ModelType, FeatureType>, GroupKey>(
      fitter, inlier_metric_from_group, consensus_metric_from_groups,
      is_valid_candidate);
};

template <typename InlierMetric, typename ConsensusMetric,
          typename IsValidCandidateMetric, typename GrouperFunction>
struct GaussianProcessRansacStrategy {

  GaussianProcessRansacStrategy() = default;

  GaussianProcessRansacStrategy(
      const InlierMetric &inlier_metric,
      const ConsensusMetric &consensus_metric,
      const IsValidCandidateMetric &is_valid_candidate,
      GrouperFunction grouper_function)
      : inlier_metric_(inlier_metric), consensus_metric_(consensus_metric),
        is_valid_candidate_(is_valid_candidate),
        grouper_function_(grouper_function){};

  template <typename ModelType, typename FeatureType>
  auto operator()(const ModelType &model,
                  const RegressionDataset<FeatureType> &dataset) const {
    const auto indexer = get_indexer(dataset);
    return get_gp_ransac_functions(model, dataset, indexer, inlier_metric_,
                                   consensus_metric_, is_valid_candidate_);
  }

  template <typename FeatureType>
  auto get_indexer(const RegressionDataset<FeatureType> &dataset) const {
    return dataset.group_by(grouper_function_).indexers();
  }

protected:
  InlierMetric inlier_metric_;
  ConsensusMetric consensus_metric_;
  IsValidCandidateMetric is_valid_candidate_;
  GrouperFunction grouper_function_;
};

using DefaultGPRansacStrategy = GaussianProcessRansacStrategy<
    NegativeLogLikelihood<JointDistribution>, FeatureCountConsensusMetric,
    AlwaysAcceptCandidateMetric, LeaveOneOutGrouper>;

template <typename InlierMetric, typename ConsensusMetric,
          typename GrouperFunction>
auto gp_ransac_strategy(const InlierMetric &inlier_metric,
                        const ConsensusMetric &consensus_metric,
                        GrouperFunction &grouper_function) {
  AlwaysAcceptCandidateMetric always_accept;
  return GaussianProcessRansacStrategy<InlierMetric, ConsensusMetric,
                                       AlwaysAcceptCandidateMetric,
                                       GrouperFunction>(
      inlier_metric, consensus_metric, always_accept, grouper_function);
}

template <typename InlierMetric, typename ConsensusMetric,
          typename IsValidCandidateMetric, typename GrouperFunction>
auto gp_ransac_strategy(const InlierMetric &inlier_metric,
                        const ConsensusMetric &consensus_metric,
                        const IsValidCandidateMetric &is_valid_candidate,
                        GrouperFunction grouper_function) {
  return GaussianProcessRansacStrategy<InlierMetric, ConsensusMetric,
                                       IsValidCandidateMetric, GrouperFunction>(
      inlier_metric, consensus_metric, is_valid_candidate, grouper_function);
}

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_MODELS_RANSAC_GP_H_ */
