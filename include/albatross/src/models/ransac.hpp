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

template <typename GroupKey>
inline bool accept_all_candidates(const std::vector<GroupKey> &) {
  return true;
}

template <typename FitType, typename GroupKey> struct RansacFunctions {
  // A function which takes a bunch of keys and fits a model
  // to the corresponding subset of data.
  using FitterFunc = std::function<FitType(const std::vector<GroupKey> &)>;

  // A function which takes a fit and a set of indices
  // and returns a metric which represents how well the model
  // predicted the subset corresponding to the indices.
  using InlierMetric = std::function<double(const GroupKey &, const FitType &)>;

  // A function which returns an error metric value which indicates
  // how good a set of inliers is, lower is better.
  using ConsensusMetric = std::function<double(const std::vector<GroupKey> &)>;

  using IsValidCandidate = std::function<bool(const std::vector<GroupKey> &)>;

  RansacFunctions(
      FitterFunc fitter_, InlierMetric inlier_metric_,
      ConsensusMetric consensus_metric_,
      IsValidCandidate is_valid_candidate_ = accept_all_candidates<GroupKey>)
      : fitter(fitter_), inlier_metric(inlier_metric_),
        consensus_metric(consensus_metric_),
        is_valid_candidate(is_valid_candidate_){};

  FitterFunc fitter;
  InlierMetric inlier_metric;
  ConsensusMetric consensus_metric;
  IsValidCandidate is_valid_candidate;
};

template <typename GroupKey>
inline bool contains_group(const std::vector<GroupKey> &vect,
                           const GroupKey &group) {
  return vector_contains(vect, group);
}

typedef enum ransac_return_code_e {
  RANSAC_RETURN_CODE_INVALID = -1,
  RANSAC_RETURN_CODE_SUCCESS,
  RANSAC_RETURN_CODE_NO_CONSENSUS,
  RANSAC_RETURN_CODE_INVALID_ARGUMENTS,
  RANSAC_RETURN_CODE_EXCEEDED_MAX_FAILED_CANDIDATES,
  RANSAC_RETURN_CODE_FAILURE
} ransac_return_code_t;

inline std::string to_string(const ransac_return_code_t &return_code) {
  switch (return_code) {
  case RANSAC_RETURN_CODE_INVALID:
    return "invalid";
  case RANSAC_RETURN_CODE_SUCCESS:
    return "success";
  case RANSAC_RETURN_CODE_NO_CONSENSUS:
    return "no_consensus";
  case RANSAC_RETURN_CODE_INVALID_ARGUMENTS:
    return "invalid_arguments";
  case RANSAC_RETURN_CODE_EXCEEDED_MAX_FAILED_CANDIDATES:
    return "exceeded_max_failed_candidates";
  case RANSAC_RETURN_CODE_FAILURE:
    return "failure";
  default:
    ALBATROSS_ASSERT(false);
    return "unknown return code";
  }
}

inline bool ransac_success(const ransac_return_code_t &rc) {
  return rc == RANSAC_RETURN_CODE_SUCCESS;
}

template <typename GroupKey> struct RansacIteration {
  std::vector<GroupKey> candidates;
  std::map<GroupKey, double> inliers;
  std::map<GroupKey, double> outliers;
  double consensus_metric_value;

  RansacIteration() : consensus_metric_value(NAN){};

  std::vector<GroupKey> consensus() const {
    std::vector<GroupKey> output(candidates);
    for (const auto &pair : inliers) {
      output.emplace_back(pair.first);
    }
    return output;
  }

  bool operator==(const RansacIteration &other) const {
    const bool is_same_consensus_metric =
        ((std::isnan(consensus_metric_value) &&
          std::isnan(other.consensus_metric_value)) ||
         consensus_metric_value == other.consensus_metric_value);
    return (candidates == other.candidates && inliers == other.inliers &&
            outliers == other.outliers && is_same_consensus_metric);
  }
};

template <typename GroupKey> struct RansacOutput {

  RansacOutput() : return_code(RANSAC_RETURN_CODE_INVALID){};

  using key_type = GroupKey;

  ransac_return_code_t return_code;
  RansacIteration<GroupKey> best;
  std::vector<RansacIteration<GroupKey>> iterations;

  bool operator==(const RansacOutput &other) const {
    return (return_code == other.return_code && best == best &&
            iterations == other.iterations);
  }
};

struct RansacConfig {

  RansacConfig()
      : inlier_threshold(NAN), random_sample_size(0), min_consensus_size(0),
        max_iterations(0), max_failed_candidates(0){};

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
template <typename FitType, typename GroupKey>
RansacOutput<GroupKey>
ransac(const RansacFunctions<FitType, GroupKey> &ransac_functions,
       const std::vector<GroupKey> &groups, double inlier_threshold,
       std::size_t random_sample_size, std::size_t min_consensus_size,
       std::size_t max_iterations, std::size_t max_failed_candidates) {

  RansacOutput<GroupKey> output;
  output.return_code = RANSAC_RETURN_CODE_FAILURE;

  if (min_consensus_size >= groups.size() ||
      min_consensus_size < random_sample_size ||
      random_sample_size >= groups.size() || random_sample_size <= 0 ||
      max_iterations <= 0) {
    output.return_code = RANSAC_RETURN_CODE_INVALID_ARGUMENTS;
    return output;
  }

  std::default_random_engine gen;

  std::size_t i = 0;
  std::size_t failed_candidates = 0;
  while (i < max_iterations) {

    output.iterations.emplace_back(RansacIteration<GroupKey>());
    RansacIteration<GroupKey> &iteration = output.iterations.back();

    // Sample a random subset of the data and fit a model.
    iteration.candidates =
        random_without_replacement(groups, random_sample_size, gen);

    // Sometimes it's hard to design an inlier metric which is
    // reliable if the candidate groups are tainted with outliers.
    // Consider a situation where there are multiple correlated
    // outliers and one of those ends up in the candidate set, it's
    // possible the model will then reasonably predict inliers
    // AND outliers resulting in a large consensus set.  This
    // is_valid_candidate step allows you to filter those cases out.
    if (!ransac_functions.is_valid_candidate(iteration.candidates)) {
      ++failed_candidates;
      if (failed_candidates >= max_failed_candidates) {
        output.return_code = RANSAC_RETURN_CODE_EXCEEDED_MAX_FAILED_CANDIDATES;
        return output;
      }
      continue;
    }

    const auto fit = ransac_functions.fitter(iteration.candidates);

    // Find which of the other groups agree with the reference model
    // which gives us a consensus (set of inliers).
    for (const auto &possible_inlier : groups) {
      if (!contains_group(iteration.candidates, possible_inlier)) {
        double metric_value =
            ransac_functions.inlier_metric(possible_inlier, fit);
        const bool accepted = metric_value <= inlier_threshold;
        if (accepted) {
          iteration.inliers.insert({possible_inlier, metric_value});
        } else {
          iteration.outliers.insert({possible_inlier, metric_value});
        }
      }
    }

    // If there is enough agreement, consider this random set of inliers
    // as a candidate model.
    const auto consensus = iteration.consensus();
    if (consensus.size() >= min_consensus_size) {
      iteration.consensus_metric_value =
          ransac_functions.consensus_metric(consensus);
      const bool best_so_far =
          std::isnan(output.best.consensus_metric_value) ||
          iteration.consensus_metric_value < output.best.consensus_metric_value;
      if (best_so_far) {
        output.best = RansacIteration<GroupKey>(iteration);
      }
    }

    ++i;
  }

  if (output.best.consensus().size()) {
    output.return_code = RANSAC_RETURN_CODE_SUCCESS;
  } else {
    output.return_code = RANSAC_RETURN_CODE_NO_CONSENSUS;
  }

  return output;
}

template <typename FitType, typename GroupKey>
auto ransac(const RansacFunctions<FitType, GroupKey> &ransac_functions,
            const GroupIndexer<GroupKey> &indexer, double inlier_threshold,
            std::size_t random_sample_size, std::size_t min_consensus_size,
            std::size_t max_iterations) {
  return ransac(ransac_functions, map_keys(indexer), inlier_threshold,
                random_sample_size, min_consensus_size, max_iterations,
                max_iterations);
}

template <typename FitType, typename GroupKey>
auto ransac(const RansacFunctions<FitType, GroupKey> &ransac_functions,
            const GroupIndexer<GroupKey> &indexer, const RansacConfig &config) {
  return ransac(ransac_functions, map_keys(indexer), config.inlier_threshold,
                config.random_sample_size, config.min_consensus_size,
                config.max_iterations, config.max_failed_candidates);
}

template <typename ModelType, typename FeatureType, typename InlierMetric,
          typename ConsensusMetric, typename GroupKey,
          typename FitModelType =
              typename fit_model_type<ModelType, FeatureType>::type>
inline RansacFunctions<FitModelType, GroupKey> get_generic_ransac_functions(
    const ModelType &model, const RegressionDataset<FeatureType> &dataset,
    const GroupIndexer<GroupKey> &indexer, const InlierMetric &inlier_metric,
    const ConsensusMetric &consensus_metric) {

  static_assert(is_prediction_metric<InlierMetric>::value,
                "InlierMetric must be an PredictionMetric.");

  static_assert(is_model_metric<ConsensusMetric, FeatureType, ModelType>::value,
                "ConsensusMetric must be a ModelMetric valid for the feature "
                "and model provided");

  std::function<FitModelType(const std::vector<GroupKey> &)> fitter =
      [&, indexer](const std::vector<GroupKey> &groups) {
        auto inds = indices_from_groups(indexer, groups);
        return model.fit(subset(dataset, inds));
      };

  auto inlier_metric_from_group = [&, indexer](const GroupKey &group,
                                               const FitModelType &fit) {
    GroupIndices inds = indexer.at(group);
    const auto pred = fit.predict(subset(dataset.features, inds));
    const auto target = subset(dataset.targets, inds);
    return inlier_metric(pred, target);
  };

  auto consensus_metric_from_group =
      [&, indexer](const std::vector<GroupKey> &groups) {
        auto inds = indices_from_groups(indexer, groups);
        RegressionDataset<FeatureType> inlier_dataset = subset(dataset, inds);
        return consensus_metric(inlier_dataset, model);
      };

  return RansacFunctions<FitModelType, GroupKey>(
      fitter, inlier_metric_from_group, consensus_metric_from_group);
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
          typename GrouperFunction>
struct GenericRansacStrategy {

  static_assert(is_prediction_metric<InlierMetric>::value,
                "InlierMetric is not a valid prediction metric");

  GenericRansacStrategy() = default;

  GenericRansacStrategy(const InlierMetric &inlier_metric,
                        const ConsensusMetric &consensus_metric,
                        GrouperFunction grouper_function)
      : inlier_metric_(inlier_metric), consensus_metric_(consensus_metric),
        grouper_function_(grouper_function){};

  template <typename ModelType, typename FeatureType,
            typename FitModelType =
                typename fit_model_type<ModelType, FeatureType>::type>
  auto operator()(const ModelType &model,
                  const RegressionDataset<FeatureType> &dataset) const {
    const auto indexer = get_indexer(dataset);
    return get_generic_ransac_functions(model, dataset, indexer, inlier_metric_,
                                        consensus_metric_);
  }

  template <typename FeatureType>
  auto get_indexer(const RegressionDataset<FeatureType> &dataset) const {
    return dataset.group_by(grouper_function_).indexers();
  }

protected:
  InlierMetric inlier_metric_;
  ConsensusMetric consensus_metric_;
  GrouperFunction grouper_function_;
};

using DefaultRansacStrategy =
    GenericRansacStrategy<NegativeLogLikelihood<JointDistribution>,
                          LeaveOneOutLikelihood<JointDistribution>,
                          LeaveOneOutGrouper>;

template <typename InlierMetric, typename ConsensusMetric,
          typename GrouperFunction>
auto get_generic_ransac_strategy(const InlierMetric &inlier_metric,
                                 const ConsensusMetric &consensus_metric,
                                 GrouperFunction grouper_function) {
  return GenericRansacStrategy<InlierMetric, ConsensusMetric, GrouperFunction>(
      inlier_metric, consensus_metric, grouper_function);
}

template <typename ModelType, typename StrategyType, typename FeatureType,
          typename GroupKey>
struct RansacFit {};

/*
 * Ransac Model Implementation.
 */
template <typename ModelType, typename StrategyType, typename FeatureType,
          typename GroupKey>
struct Fit<RansacFit<ModelType, StrategyType, FeatureType, GroupKey>> {

  using FitModelType = typename fit_model_type<ModelType, FeatureType>::type;

  struct EmptyFit {
    bool operator==(const EmptyFit &other ALBATROSS_UNUSED) const {
      return true;
    };

    template <typename Archive>
    void serialize(Archive &archive ALBATROSS_UNUSED, const std::uint32_t){};
  };

  Fit() : maybe_empty_fit_model(EmptyFit()){};

  Fit(const FitModelType &fit_model_,
      const RansacOutput<GroupKey> &ransac_output_)
      : maybe_empty_fit_model(fit_model_), ransac_output(ransac_output_){};

  Fit(const RansacOutput<GroupKey> &ransac_output_)
      : maybe_empty_fit_model(EmptyFit()), ransac_output(ransac_output_){};

  bool operator==(const Fit &other) const {
    return (maybe_empty_fit_model == other.maybe_empty_fit_model &&
            ransac_output == other.ransac_output);
  }

  bool has_fit_model() const {
    return maybe_empty_fit_model.template is<FitModelType>();
  }

  const FitModelType &get_fit_model_or_assert() const {
    ALBATROSS_ASSERT(maybe_empty_fit_model.template is<FitModelType>());
    return maybe_empty_fit_model.template get<FitModelType>();
  }

  variant<EmptyFit, FitModelType> maybe_empty_fit_model;
  RansacOutput<GroupKey> ransac_output;
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

  void set_param(const std::string &name, const Parameter &param) override {
    sub_model_.set_param(name, param);
  }

  template <typename FeatureType>
  auto _fit_impl(const std::vector<FeatureType> &features,
                 const MarginalDistribution &targets) const {

    static_assert(has_call_operator<StrategyType, ModelType,
                                    RegressionDataset<FeatureType>>::value,
                  "Invalid Ransac Strategy");

    RegressionDataset<FeatureType> dataset(features, targets);

    auto indexer = strategy_.get_indexer(dataset);
    auto ransac_functions = strategy_(sub_model_, dataset);

    const auto ransac_output = ransac(ransac_functions, indexer, config_);
    using GroupKey = typename decltype(ransac_output)::key_type;

    if (ransac_success(ransac_output.return_code)) {
      const auto good_inds =
          indices_from_groups(indexer, ransac_output.best.consensus());
      const auto consensus_set = subset(dataset, good_inds);
      return Fit<RansacFit<ModelType, StrategyType, FeatureType, GroupKey>>(
          sub_model_.fit(consensus_set), ransac_output);
    } else {
      return Fit<RansacFit<ModelType, StrategyType, FeatureType, GroupKey>>(
          ransac_output);
    }
  }

  template <typename PredictFeatureType, typename FitType, typename PredictType>
  PredictType _predict_impl(const std::vector<PredictFeatureType> &features,
                            const FitType &ransac_fit_,
                            PredictTypeIdentity<PredictType> &&) const {
    // If RANSAC failed it's up to the user to determine that from the output of
    // fit and deal with it appropriately.
    ALBATROSS_ASSERT(ransac_fit_.has_fit_model());
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

template <typename X,
          typename std::enable_if_t<is_streamable<X>::value, int> = 0>
inline std::ostream &operator<<(std::ostream &os,
                                const RansacIteration<X> &iteration) {
  os << "CANDIDATES: ";
  for (const auto &k : iteration.candidates) {
    os << k << ", ";
  }
  os << std::endl;

  os << "INLIERS: " << std::endl;
  for (const auto &pair : iteration.inliers) {
    os << "    " << pair.first << " : " << pair.second << std::endl;
  }

  os << "OUTLIERS: " << std::endl;
  for (const auto &pair : iteration.outliers) {
    os << "    " << pair.first << " : " << pair.second << std::endl;
  }

  os << "CONSENSUS_METRIC: " << iteration.consensus_metric_value;
  return os;
}

template <typename X,
          typename std::enable_if_t<is_streamable<X>::value, int> = 0>
inline std::ostream &operator<<(std::ostream &os,
                                const RansacOutput<X> &output) {
  for (std::size_t i = 0; i < output.iterations.size(); ++i) {
    os << "==== Iteration " << i << "====" << std::endl;
    os << output.iterations[i] << std::endl;
  }
  os << "==== Best ====" << std::endl;
  os << output.best << std::endl;
  os << "RETURN CODE: " << to_string(output.return_code) << std::endl;

  return os;
}

} // namespace albatross
#endif
