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

constexpr double DEFAULT_CHI_SQUARED_THRESHOLD = 0.999;

template <typename GroupKey>
inline typename RansacFunctions<ConditionalFit, GroupKey>::FitterFunc
get_gp_ransac_fitter(const ConditionalGaussian &model,
                     const GroupIndexer<GroupKey> &indexer) {

  return [&, model, indexer](const std::vector<GroupKey> &groups) {
    auto indices = indices_from_groups(indexer, groups);
    return model.fit_from_indices(indices);
  };
}

template <typename IsValidCandidateMetric, typename GroupKey>
inline typename RansacFunctions<ConditionalFit, GroupKey>::IsValidCandidate
get_gp_ransac_is_valid_candidate(const ConditionalGaussian &model,
                                 const GroupIndexer<GroupKey> &indexer,
                                 const IsValidCandidateMetric &metric) {

  return [&, model, indexer](const std::vector<GroupKey> &groups) {
    const auto indices = indices_from_groups(indexer, groups);
    const auto prior = model.get_prior(indices);
    const auto truth = model.get_truth(indices);
    return metric(prior, truth);
  };
}

template <typename InlierMetricType, typename GroupKey>
inline typename RansacFunctions<ConditionalFit, GroupKey>::InlierMetric
get_gp_ransac_inlier_metric(const ConditionalGaussian &model,
                            const GroupIndexer<GroupKey> &indexer,
                            const InlierMetricType &metric) {

  return [&, indexer, model](const GroupKey &group, const ConditionalFit &fit) {
    const auto indices = indexer.at(group);
    const auto pred = get_prediction_reference(model, fit, indices);
    const auto truth = model.get_truth(indices);
    return metric(pred, truth);
  };
}

template <typename ConsensusMetric, typename GroupKey>
inline typename RansacFunctions<ConditionalFit, GroupKey>::ConsensusMetric
get_gp_ransac_consensus_metric(const ConditionalGaussian &model,
                               const GroupIndexer<GroupKey> &indexer,
                               const ConsensusMetric &metric) {

  return [&, model, indexer](const std::vector<GroupKey> &groups) {
    const auto indices = indices_from_groups(indexer, groups);
    const auto prior = model.get_prior(indices);
    const auto truth = model.get_truth(indices);
    return metric(prior, truth);
  };
}

struct DifferentialEntropyConsensusMetric {
  double operator()(const JointDistribution &pred,
                    const MarginalDistribution &truth) const {
    return differential_entropy(pred.covariance);
  }
};

struct FeatureCountConsensusMetric {
  double operator()(const JointDistribution &prior,
                    const MarginalDistribution &truth) const {
    // Negative because a lower metric is better.
    return (-1.0 * static_cast<double>(truth.size()));
  }
};

struct ChiSquaredConsensusMetric {
  double operator()(const JointDistribution &prior,
                    const MarginalDistribution &truth) const {
    return chi_squared_cdf(prior, truth);
  }
};

struct ChiSquaredIsValidCandidateMetric {

  ChiSquaredIsValidCandidateMetric(
      double chi_squared_treshold = DEFAULT_CHI_SQUARED_THRESHOLD)
      : chi_squared_threshold_(chi_squared_treshold) {}

  bool operator()(const JointDistribution &pred,
                  const MarginalDistribution &truth) const {
    // These thresholds are under the assumption of a perfectly
    // representative prior.
    const double probability_prior_exceeded = chi_squared_cdf(pred, truth);
    return (probability_prior_exceeded <= chi_squared_threshold_);
  };

private:
  double chi_squared_threshold_;
};

struct AlwaysAcceptCandidateMetric {
  bool operator()(const JointDistribution &pred,
                  const MarginalDistribution &truth) const {
    return true;
  }
};

template <typename InlierMetric, typename ConsensusMetric,
          typename IsValidCandidateMetric, typename GroupKey>
inline RansacFunctions<ConditionalFit, GroupKey> get_gp_ransac_functions(
    const JointDistribution &prior, const MarginalDistribution &truth,
    const GroupIndexer<GroupKey> &indexer, const InlierMetric &inlier_metric,
    const ConsensusMetric &consensus_metric,
    const IsValidCandidateMetric &is_valid_candidate_metric) {

  static_assert(is_prediction_metric<InlierMetric>::value,
                "InlierMetric must be an PredictionMetric.");

  assert(prior.size() == truth.size());

  const ConditionalGaussian model(prior, truth);

  const auto fitter = get_gp_ransac_fitter<GroupKey>(model, indexer);

  const auto inlier_metric_from_group =
      get_gp_ransac_inlier_metric<InlierMetric, GroupKey>(model, indexer,
                                                          inlier_metric);

  const auto consensus_metric_from_groups =
      get_gp_ransac_consensus_metric<ConsensusMetric, GroupKey>(
          model, indexer, consensus_metric);

  const auto is_valid_candidate = get_gp_ransac_is_valid_candidate(
      model, indexer, is_valid_candidate_metric);

  return RansacFunctions<ConditionalFit, GroupKey>(
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
    return get_gp_ransac_functions(model.prior(dataset.features),
                                   dataset.targets, indexer, inlier_metric_,
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
