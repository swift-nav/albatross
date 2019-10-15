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

#ifndef ALBATROSS_MODELS_RANSAC_H
#define ALBATROSS_MODELS_RANSAC_H

namespace albatross {

inline bool accept_all_candidates(const std::vector<FoldName> &) {
  return true;
}

template <typename FitType> struct RansacFunctions {
  // A function which takes a bunch of keys and fits a model
  // to the corresponding subset of data.
  using FitterFunc = std::function<FitType(const std::vector<FoldName> &)>;

  // A function which takes a fit and a set of indices
  // and returns a metric which represents how well the model
  // predicted the subset corresponding to the indices.
  using InlierMetric = std::function<double(const FoldName &, const FitType &)>;

  // A function which returns an error metric value which indicates
  // how good a set of inliers is, lower is better.
  using ConsensusMetric = std::function<double(const std::vector<FoldName> &)>;

  using IsValidCandidate = std::function<bool(const std::vector<FoldName> &)>;

  RansacFunctions(FitterFunc fitter_, InlierMetric inlier_metric_,
                  ConsensusMetric consensus_metric_,
                  IsValidCandidate is_valid_candidate_ = accept_all_candidates)
      : fitter(fitter_), inlier_metric(inlier_metric_),
        consensus_metric(consensus_metric_),
        is_valid_candidate(is_valid_candidate_){};

  FitterFunc fitter;
  InlierMetric inlier_metric;
  ConsensusMetric consensus_metric;
  IsValidCandidate is_valid_candidate;
};

inline bool contains_group(const std::vector<FoldName> &vect,
                           const FoldName &group) {
  return std::find(vect.begin(), vect.end(), group) != vect.end();
}

typedef enum ransac_return_code_e {
  RANSAC_RETURN_CODE_INVALID = -1,
  RANSAC_RETURN_CODE_SUCCESS,
  RANSAC_RETURN_CODE_NO_CONSENSUS,
  RANSAC_RETURN_CODE_INVALID_ARGUMENTS,
  RANSAC_RETURN_CODE_EXCEEDED_MAX_FAILED_CANDIDATES,
  RANSAC_RETURN_CODE_FAILURE
} ransac_return_code_t;

inline bool ransac_success(const ransac_return_code_t &rc) {
  return rc == RANSAC_RETURN_CODE_SUCCESS;
}

struct RansacOutput {

  RansacOutput()
      : return_code(RANSAC_RETURN_CODE_INVALID), inliers(), outliers(),
        consensus_metric(HUGE_VAL){};

  ransac_return_code_t return_code;
  std::vector<FoldName> inliers;
  std::vector<FoldName> outliers;
  double consensus_metric;
};

struct RansacConfig {

  RansacConfig(){};

  RansacConfig(double inlier_threshold_, std::size_t random_sample_size_,
               std::size_t min_consensus_size_, std::size_t max_iterations_,
               std::size_t max_failed_candidates_)
      : inlier_threshold(inlier_threshold_),
        random_sample_size(random_sample_size_),
        min_consensus_size(min_consensus_size_),
        max_iterations(max_iterations_),
        max_failed_candidates(max_failed_candidates_){};

  double inlier_threshold;
  std::size_t random_sample_size;
  std::size_t min_consensus_size;
  std::size_t max_iterations;
  std::size_t max_failed_candidates;
};

/*
 * This RANdom SAmple Consensus (RANSAC) algorithm works as follows.
 *
 *   1) Randomly sample a small number of data points and fit a
 *      reference model to that data.
 *   2) Assemble all the data points that agree with the
 *      reference model into a set of inliers (a consensus).
 *   3) Evaluate the quality of the consensus.
 *   4) Repeat N times keeping track of the best consensus.
 *
 * One of the large drawbacks of this approach is the computational
 * load since it requires fitting and predicting repeatedly.
 * The goal of this implementation is to provide a way
 * for the user to optionally perform a lot of computation upfront,
 * then use call backs which take indices as input to selectively
 * update/downdate the model to produce the fits and evaluation
 * metrics.
 */
template <typename FitType>
RansacOutput ransac(const RansacFunctions<FitType> &ransac_functions,
                    const std::vector<FoldName> &groups,
                    double inlier_threshold, std::size_t random_sample_size,
                    std::size_t min_consensus_size, std::size_t max_iterations,
                    std::size_t max_failed_candidates) {

  RansacOutput output;
  output.return_code = RANSAC_RETURN_CODE_FAILURE;

  if (min_consensus_size >= groups.size() ||
      random_sample_size >= groups.size() || max_iterations <= 0) {
    output.return_code = RANSAC_RETURN_CODE_INVALID_ARGUMENTS;
    return output;
  }

  std::default_random_engine gen;

  std::size_t i = 0;
  std::size_t failed_candidates = 0;
  while (i < max_iterations) {
    // Sample a random subset of the data and fit a model.
    auto candidate_groups =
        random_without_replacement(groups, random_sample_size, gen);

    // Sometimes it's hard to design an inlier metric which is
    // reliable if the candidate groups are tainted with outliers.
    // Consider a situation where there are multiple correlated
    // outliers and one of those ends up in the candidate set, it's
    // possible the model will then reasonably predict inliers
    // AND outliers resulting in a large consensus set.  This
    // is_valid_candidate step allows you to filter those cases out.
    if (!ransac_functions.is_valid_candidate(candidate_groups)) {
      ++failed_candidates;
      if (failed_candidates >= max_failed_candidates) {
        output.return_code = RANSAC_RETURN_CODE_EXCEEDED_MAX_FAILED_CANDIDATES;
        return output;
      }
      continue;
    }

    const auto fit = ransac_functions.fitter(candidate_groups);

    // Any group that's part of the candidate set is automatically an inlier.
    std::vector<FoldName> candidate_consensus = candidate_groups;
    std::vector<FoldName> outliers;

    // Find which of the other groups agree with the reference model
    // which gives us a consensus (set of inliers).
    for (const auto &possible_inlier : groups) {
      if (!contains_group(candidate_groups, possible_inlier)) {
        double metric_value =
            ransac_functions.inlier_metric(possible_inlier, fit);
        if (metric_value < inlier_threshold) {
          candidate_consensus.emplace_back(possible_inlier);
        } else {
          outliers.emplace_back(possible_inlier);
        }
      }
    }

    // If there is enough agreement, consider this random set of inliers
    // as a candidate model.
    if (candidate_consensus.size() >= min_consensus_size) {
      double consensus_metric_value =
          ransac_functions.consensus_metric(candidate_consensus);
      if (consensus_metric_value < output.consensus_metric) {
        output.inliers = candidate_consensus;
        output.consensus_metric = consensus_metric_value;
        output.outliers = outliers;
      }
    }

    ++i;
  }

  if (output.inliers.size()) {
    output.return_code = RANSAC_RETURN_CODE_SUCCESS;
  } else {
    output.return_code = RANSAC_RETURN_CODE_NO_CONSENSUS;
  }

  return output;
}

template <typename FitType>
auto ransac(const RansacFunctions<FitType> &ransac_functions,
            const FoldIndexer &indexer, double inlier_threshold,
            std::size_t random_sample_size, std::size_t min_consensus_size,
            std::size_t max_iterations) {
  return ransac(ransac_functions, map_keys(indexer), inlier_threshold,
                random_sample_size, min_consensus_size, max_iterations,
                max_iterations);
}

template <typename FitType>
auto ransac(const RansacFunctions<FitType> &ransac_functions,
            const FoldIndexer &indexer, const RansacConfig &config) {
  return ransac(ransac_functions, map_keys(indexer), config.inlier_threshold,
                config.random_sample_size, config.min_consensus_size,
                config.max_iterations, config.max_failed_candidates);
}

template <typename ModelType, typename FeatureType, typename InlierMetric,
          typename ConsensusMetric,
          typename FitModelType =
              typename fit_model_type<ModelType, FeatureType>::type>
inline RansacFunctions<FitModelType> get_generic_ransac_functions(
    const ModelType &model, const RegressionDataset<FeatureType> &dataset,
    const FoldIndexer &indexer, const InlierMetric &inlier_metric,
    const ConsensusMetric &consensus_metric) {

  static_assert(is_prediction_metric<InlierMetric>::value,
                "InlierMetric must be an PredictionMetric.");

  static_assert(is_model_metric<ConsensusMetric, FeatureType, ModelType>::value,
                "ConsensusMetric must be a ModelMetric valid for the feature "
                "and model provided");

  std::function<FitModelType(const std::vector<FoldName> &)> fitter =
      [&, indexer](const std::vector<FoldName> &groups) {
        auto inds = indices_from_names(indexer, groups);
        return model.fit(subset(dataset, inds));
      };

  auto inlier_metric_from_group = [&, indexer](const FoldName &group,
                                               const FitModelType &fit) {
    FoldIndices inds = indexer.at(group);
    const auto pred = fit.predict(subset(dataset.features, inds));
    const auto target = subset(dataset.targets, inds);
    return inlier_metric(pred, target);
  };

  auto consensus_metric_from_group =
      [&, indexer](const std::vector<FoldName> &groups) {
        auto inds = indices_from_names(indexer, groups);
        RegressionDataset<FeatureType> inlier_dataset = subset(dataset, inds);
        return consensus_metric(inlier_dataset, model);
      };

  return RansacFunctions<FitModelType>(fitter, inlier_metric_from_group,
                                       consensus_metric_from_group);
};

/*
 * Ransac Strategy implementation
 *
 * A RansacStrategy defines how Ransac should be performed
 * before seeing the actual model and dataset.  It is
 * responsible for defining how the dataset should be split
 * into groups and for producing the functions which produce
 * a fit/inlier_metric/consensus_metric when given group names.
 */
template <typename InlierMetric, typename ConsensusMetric,
          typename IndexingFunction>
struct GenericRansacStrategy {

  static_assert(is_prediction_metric<InlierMetric>::value,
                "InlierMetric is not a valid prediction metric");

  GenericRansacStrategy() = default;

  GenericRansacStrategy(const InlierMetric &inlier_metric,
                        const ConsensusMetric &consensus_metric,
                        const IndexingFunction &indexing_function)
      : inlier_metric_(inlier_metric), consensus_metric_(consensus_metric),
        indexing_function_(indexing_function){};

  template <typename ModelType, typename FeatureType,
            typename FitModelType =
                typename fit_model_type<ModelType, FeatureType>::type>
  RansacFunctions<FitModelType>
  operator()(const ModelType &model,
             const RegressionDataset<FeatureType> &dataset) const {
    const auto indexer = get_indexer(dataset);
    return get_generic_ransac_functions(model, dataset, indexer, inlier_metric_,
                                        consensus_metric_);
  }

  template <typename FeatureType>
  FoldIndexer get_indexer(const RegressionDataset<FeatureType> &dataset) const {
    return indexing_function_(dataset);
  }

protected:
  InlierMetric inlier_metric_;
  ConsensusMetric consensus_metric_;
  IndexingFunction indexing_function_;
};

using DefaultRansacStrategy =
    GenericRansacStrategy<NegativeLogLikelihood<JointDistribution>,
                          LeaveOneOutLikelihood<JointDistribution>,
                          LeaveOneOut>;

template <typename InlierMetric, typename ConsensusMetric,
          typename IndexingFunction>
auto get_generic_ransac_strategy(const InlierMetric &inlier_metric,
                                 const ConsensusMetric &consensus_metric,
                                 const IndexingFunction &indexing_function) {
  return GenericRansacStrategy<InlierMetric, ConsensusMetric, IndexingFunction>(
      inlier_metric, consensus_metric, indexing_function);
}

template <typename ModelType, typename StrategyType, typename FeatureType>
struct RansacFit {};

/*
 * Ransac Model Implementation.
 */
template <typename ModelType, typename StrategyType, typename FeatureType>
struct Fit<RansacFit<ModelType, StrategyType, FeatureType>> {

  using FitModelType = typename fit_model_type<ModelType, FeatureType>::type;

  struct EmptyFit {
    bool operator==(const EmptyFit &other) const { return true; };

    template <typename Archive>
    void serialize(Archive &archive, const std::uint32_t){};
  };

  Fit() : maybe_empty_fit_model(EmptyFit()){};

  Fit(const FitModelType &fit_model_, const RansacOutput &ransac_output_)
      : maybe_empty_fit_model(fit_model_), ransac_output(ransac_output_){};

  Fit(const RansacOutput &ransac_output_)
      : maybe_empty_fit_model(EmptyFit()), ransac_output(ransac_output_){};

  bool operator==(const Fit &other) const {
    return maybe_empty_fit_model == other.maybe_empty_fit_model;
  }

  bool has_fit_model() const {
    return maybe_empty_fit_model.template is<FitModelType>();
  }

  const FitModelType &get_fit_model_or_assert() const {
    assert(maybe_empty_fit_model.template is<FitModelType>());
    return maybe_empty_fit_model.template get<FitModelType>();
  }

  variant<EmptyFit, FitModelType> maybe_empty_fit_model;
  RansacOutput ransac_output;
};

/*
 * This wraps any other model and performs ransac each time fit is called.
 */
template <typename ModelType, typename StrategyType>
class Ransac : public ModelBase<Ransac<ModelType, StrategyType>> {
public:
  Ransac(){};

  Ransac(const ModelType &sub_model, const StrategyType &strategy,
         const RansacConfig &config)
      : sub_model_(sub_model), strategy_(strategy), config_(config){};

  Ransac(const ModelType &sub_model, const StrategyType &strategy,
         double inlier_threshold, std::size_t random_sample_size,
         std::size_t min_consensus_size, std::size_t max_iterations)
      : Ransac(sub_model, strategy,
               RansacConfig(inlier_threshold, random_sample_size,
                            min_consensus_size, max_iterations, 0)){};

  Ransac(const ModelType &sub_model, const StrategyType &strategy,
         double inlier_threshold, std::size_t random_sample_size,
         std::size_t min_consensus_size, std::size_t max_iterations,
         std::size_t max_failed_candidates)
      : Ransac(sub_model, strategy,
               RansacConfig(inlier_threshold, random_sample_size,
                            min_consensus_size, max_iterations,
                            max_failed_candidates)){};

  std::string get_name() const {
    return "ransac[" + sub_model_.get_name() + "]";
    ;
  };

  ParameterStore get_params() const override { return sub_model_.get_params(); }

  void unchecked_set_param(const std::string &name,
                           const Parameter &param) override {
    sub_model_.set_param(name, param);
  }

  template <typename FeatureType>
  Fit<RansacFit<ModelType, StrategyType, FeatureType>>
  _fit_impl(const std::vector<FeatureType> &features,
            const MarginalDistribution &targets) const {

    static_assert(has_call_operator<StrategyType, ModelType,
                                    RegressionDataset<FeatureType>>::value,
                  "Invalid Ransac Strategy");

    RegressionDataset<FeatureType> dataset(features, targets);

    auto indexer = strategy_.get_indexer(dataset);
    auto ransac_functions = strategy_(sub_model_, dataset);

    const auto ransac_output = ransac(ransac_functions, indexer, config_);

    if (ransac_success(ransac_output.return_code)) {
      const auto good_inds = indices_from_names(indexer, ransac_output.inliers);
      const auto consensus_set = subset(dataset, good_inds);
      return Fit<RansacFit<ModelType, StrategyType, FeatureType>>(
          sub_model_.fit(consensus_set), ransac_output);
    } else {
      std::cout << ransac_output.return_code << std::endl;
      return Fit<RansacFit<ModelType, StrategyType, FeatureType>>(
          ransac_output);
    }
  }

  template <typename PredictFeatureType, typename FitType, typename PredictType>
  PredictType _predict_impl(const std::vector<PredictFeatureType> &features,
                            const FitType &ransac_fit_,
                            PredictTypeIdentity<PredictType> &&) const {
    // If RANSAC failed it's up to the user to determine that from the output of
    // fit and deal with it appropriately.
    assert(ransac_fit_.has_fit_model());
    return ransac_fit_.get_fit_model_or_assert()
        .predict(features)
        .template get<PredictType>();
  }

  ModelType sub_model_;
  StrategyType strategy_;
  RansacConfig config_;
};

template <typename ModelType>
template <typename StrategyType>
Ransac<ModelType, StrategyType> ModelBase<ModelType>::ransac(
    const StrategyType &strategy, double inlier_threshold,
    std::size_t random_sample_size, std::size_t min_consensus_size,
    std::size_t max_iterations) const {
  return Ransac<ModelType, StrategyType>(derived(), strategy, inlier_threshold,
                                         random_sample_size, min_consensus_size,
                                         max_iterations);
}

template <typename ModelType>
template <typename StrategyType>
Ransac<ModelType, StrategyType>
ModelBase<ModelType>::ransac(const StrategyType &strategy,
                             const RansacConfig &config) const {
  return Ransac<ModelType, StrategyType>(derived(), strategy, config);
}

} // namespace albatross

#endif
