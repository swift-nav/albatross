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
  FoldIndices indices;
};

template <typename ModelType, typename FeatureType>
inline
    typename RansacFunctions<FitAndIndices<ModelType, FeatureType>>::FitterFunc
    get_gp_ransac_fitter(const RegressionDataset<FeatureType> &dataset,
                         const FoldIndexer &indexer,
                         const Eigen::MatrixXd &cov) {
  return [&, indexer, cov, dataset](const std::vector<FoldName> &groups) {
    auto inds = indices_from_names(indexer, groups);
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
          typename IsValidCandidateMetric>
inline typename RansacFunctions<
    FitAndIndices<ModelType, FeatureType>>::IsValidCandidate
get_gp_ransac_is_valid_candidate(const RegressionDataset<FeatureType> &dataset,
                                 const FoldIndexer &indexer,
                                 const Eigen::MatrixXd &cov,
                                 const IsValidCandidateMetric &metric) {

  return [&, indexer, cov, dataset](const std::vector<FoldName> &groups) {
    const auto inds = indices_from_names(indexer, groups);
    return metric(inds, dataset, cov);
  };
}

template <typename ModelType, typename FeatureType, typename InlierMetricType>
inline typename RansacFunctions<
    FitAndIndices<ModelType, FeatureType>>::InlierMetric
get_gp_ransac_inlier_metric(const RegressionDataset<FeatureType> &dataset,
                            const FoldIndexer &indexer,
                            const Eigen::MatrixXd &cov, const ModelType &model,
                            const InlierMetricType &metric) {

  return [&, indexer, cov, model, dataset](
             const FoldName &group,
             const FitAndIndices<ModelType, FeatureType> &fit_and_indices) {
    const auto inds = indexer.at(group);
    const auto test_dataset = subset(dataset, inds);

    const auto pred =
        get_prediction(model, fit_and_indices.fit, test_dataset.features);
    return metric(pred, test_dataset.targets);
  };
}

class DifferentialEntropyConsensusMetric {
public:
  DifferentialEntropyConsensusMetric() : indexer_(), cov_(){};

  DifferentialEntropyConsensusMetric(const MarginalDistribution &,
                                     const FoldIndexer &indexer,
                                     const Eigen::MatrixXd &cov)
      : indexer_(indexer), cov_(cov){};

  double operator()(const std::vector<FoldName> &groups) {
    const auto inds = indices_from_names(indexer_, groups);
    const auto consensus_cov = symmetric_subset(cov_, inds);
    return differential_entropy(consensus_cov);
  }

private:
  FoldIndexer indexer_;
  Eigen::MatrixXd cov_;
};

class FeatureCountConsensusMetric {

public:
  FeatureCountConsensusMetric() : indexer_(){};

  FeatureCountConsensusMetric(const MarginalDistribution &,
                              const FoldIndexer &indexer,
                              const Eigen::MatrixXd &)
      : indexer_(indexer){};

  double operator()(const std::vector<FoldName> &groups) const {
    const auto inds = indices_from_names(indexer_, groups);
    // Negative because a lower metric is better.
    return (-1.0 * static_cast<double>(inds.size()));
  }

private:
  FoldIndexer indexer_;
};

class ChiSquaredConsensusMetric {
public:
  ChiSquaredConsensusMetric() : indexer_(), cov_(){};

  ChiSquaredConsensusMetric(const MarginalDistribution &targets,
                            const FoldIndexer &indexer,
                            const Eigen::MatrixXd &cov)
      : targets_(targets), indexer_(indexer), cov_(cov){};

  double operator()(const std::vector<FoldName> &groups) {
    const auto inds = indices_from_names(indexer_, groups);
    const auto consensus_prior = symmetric_subset(cov_, inds);
    const auto consensus_targets = subset(targets_, inds);
    return chi_squared_cdf(consensus_targets.mean, consensus_prior);
  }

private:
  MarginalDistribution targets_;
  FoldIndexer indexer_;
  Eigen::MatrixXd cov_;
};

struct ChiSquaredIsValidCandidateMetric {

  template <typename FeatureType>
  bool operator()(const FoldIndices &inds,
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
  bool operator()(const FoldIndices &inds,
                  const RegressionDataset<FeatureType> &dataset,
                  const Eigen::MatrixXd &cov) const {
    return true;
  }
};

template <typename ModelType, typename FeatureType, typename InlierMetric,
          typename ConsensusMetric, typename IsValidCandidateMetric>
inline RansacFunctions<FitAndIndices<ModelType, FeatureType>>
get_gp_ransac_functions(
    const ModelType &model, const RegressionDataset<FeatureType> &dataset,
    const FoldIndexer &indexer, const InlierMetric &inlier_metric,
    const ConsensusMetric &consensus_metric,
    const IsValidCandidateMetric &is_valid_candidate_metric) {

  static_assert(is_prediction_metric<InlierMetric>::value,
                "InlierMetric must be an PredictionMetric.");

  const auto full_cov = model.compute_covariance(dataset.features);

  const auto fitter =
      get_gp_ransac_fitter<ModelType, FeatureType>(dataset, indexer, full_cov);

  const auto inlier_metric_from_group =
      get_gp_ransac_inlier_metric<ModelType, FeatureType, InlierMetric>(
          dataset, indexer, full_cov, model, inlier_metric);

  const ConsensusMetric consensus_metric_from_group(dataset.targets, indexer,
                                                    full_cov);

  const auto is_valid_candidate =
      get_gp_ransac_is_valid_candidate<ModelType, FeatureType>(
          dataset, indexer, full_cov, is_valid_candidate_metric);

  return RansacFunctions<FitAndIndices<ModelType, FeatureType>>(
      fitter, inlier_metric_from_group, consensus_metric_from_group,
      is_valid_candidate);
};

template <typename InlierMetric, typename ConsensusMetric,
          typename IndexingFunction, typename IsValidCandidateMetric>
struct GaussianProcessRansacStrategy {

  GaussianProcessRansacStrategy() = default;

  GaussianProcessRansacStrategy(const InlierMetric &inlier_metric,
                                const ConsensusMetric &consensus_metric,
                                const IndexingFunction &indexing_function)
      : inlier_metric_(inlier_metric), consensus_metric_(consensus_metric),
        indexing_function_(indexing_function), is_valid_candidate_(){};

  template <typename ModelType, typename FeatureType>
  RansacFunctions<FitAndIndices<ModelType, FeatureType>>
  operator()(const ModelType &model,
             const RegressionDataset<FeatureType> &dataset) const {
    const auto indexer = get_indexer(dataset);
    return get_gp_ransac_functions(model, dataset, indexer, inlier_metric_,
                                   consensus_metric_, is_valid_candidate_);
  }

  template <typename FeatureType>
  FoldIndexer get_indexer(const RegressionDataset<FeatureType> &dataset) const {
    return indexing_function_(dataset);
  }

protected:
  InlierMetric inlier_metric_;
  ConsensusMetric consensus_metric_;
  IndexingFunction indexing_function_;
  IsValidCandidateMetric is_valid_candidate_;
};

using DefaultGPRansacStrategy =
    GaussianProcessRansacStrategy<NegativeLogLikelihood<JointDistribution>,
                                  FeatureCountConsensusMetric, LeaveOneOut>;

template <typename InlierMetric, typename ConsensusMetric,
          typename IndexingFunction>
auto gp_ransac_strategy(const InlierMetric &inlier_metric,
                        const IndexingFunction &indexing_function,
                        const ConsensusMetric &consensus_metric) {
  return GaussianProcessRansacStrategy<InlierMetric, ConsensusMetric,
                                       IndexingFunction>(
      inlier_metric, consensus_metric, indexing_function);
}

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_MODELS_RANSAC_GP_H_ */
