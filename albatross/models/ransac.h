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

using Indexer = std::vector<std::size_t>;
using GroupIndexer = std::vector<std::vector<std::size_t>>;

// This struct is just a type helper to make it obvious that
// the `FitType` used in the Fitter needs to be the same as
// the one used in `InlierMetric`
template <typename FitType> struct RansacFunctions {
  // A function which takes a bunch of indices and fits a model
  // to the corresponding subset of data.
  using Fitter = std::function<FitType(const Indexer &)>;
  // A function which takes a fit and a set of indices
  // and returns a metric which represents how well the model
  // predicted the subset corresponding to the indices.
  using InlierMetric = std::function<double(const Indexer &, const FitType &)>;
  // A function which returns a metric indicating how good a
  // model is when fit to a set of inliers (given by Indexer)
  using ModelMetric = std::function<double(const Indexer &)>;
};

inline Indexer concatenate_subset_of_groups(const Indexer &subset_indices,
                                            const GroupIndexer &indexer) {

  Indexer output;
  for (const auto &i : subset_indices) {
    assert(i < static_cast<std::size_t>(indexer.size()));
    output.insert(output.end(), indexer[i].begin(), indexer[i].end());
  }
  return output;
}

/*
 * This RANdom SAmple Consensus (RANSAC) algorithm works as follows.
 *
 *   1) Randomly sample a small number of data points and fit a
 *      reference model to that data.
 *   2) Assemble all the data points that agree with the
 *      reference model into a set of inliers.
 *   3) Evaluate the quality of the inliers.
 *   4) Repeat N times keeping track of the best set of inliers.
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
Indexer
ransac(const typename RansacFunctions<FitType>::Fitter &fitter,
       const typename RansacFunctions<FitType>::InlierMetric &inlier_metric,
       const typename RansacFunctions<FitType>::ModelMetric &model_metric,
       const GroupIndexer &indexer, double inlier_threshold,
       std::size_t random_sample_size, std::size_t min_inliers,
       std::size_t max_iterations) {

  std::default_random_engine gen;

  Indexer reference;
  double best_metric = HUGE_VAL;
  Indexer best_inds;

  for (std::size_t i = 0; i < max_iterations; i++) {
    // Sample a random subset of the data and fit a model.
    reference = randint_without_replacement(random_sample_size, 0,
                                            indexer.size() - 1, gen);
    auto ref_inds = concatenate_subset_of_groups(reference, indexer);
    const auto fit = fitter(ref_inds);

    // Find which of the other groups agree with the reference model
    // which gives us a set of inliers.
    auto test_groups = indices_complement(reference, indexer.size());
    Indexer inliers;
    for (const auto &test_ind : test_groups) {
      double metric_value = inlier_metric(indexer[test_ind], fit);
      if (metric_value < inlier_threshold) {
        inliers.push_back(test_ind);
      }
    }
    // If there is enough agreement, consider this random set of inliers
    // as a candidate model.
    if (inliers.size() + random_sample_size >= min_inliers) {
      const auto inlier_inds = concatenate_subset_of_groups(inliers, indexer);
      ref_inds.insert(ref_inds.end(), inlier_inds.begin(), inlier_inds.end());
      std::sort(ref_inds.begin(), ref_inds.end());
      double model_metric_value = model_metric(ref_inds);
      if (model_metric_value < best_metric) {
        best_inds = ref_inds;
        best_metric = model_metric_value;
      }
    }
  }
  return best_inds;
}

/*
 * Creates the lambda functions required to run ransac on a
 * generic RegressionModel.
 *
 * Note: This will iteratively call fit/predict for the same features which may
 * end up being prohibitively computationally expensive.  See the ransac
 * Gaussian
 * process implementation for an example of ways to speed things up for specific
 * models.
 */
template <typename ModelType, typename FeatureType, typename MetricPredictType>
RegressionDataset<FeatureType>
ransac(const RegressionDataset<FeatureType> &dataset,
       const FoldIndexer &fold_indexer, const ModelBase<ModelType> &model,
       const EvaluationMetric<MetricPredictType> &metric,
       double inlier_threshold, std::size_t random_sample_size,
       std::size_t min_inliers, int max_iterations) {

  using FitType = typename fit_model_type<ModelType, FeatureType>::type;

  typename RansacFunctions<FitType>::Fitter fitter =
      [&](const std::vector<std::size_t> &inds) {
        return model.fit(subset(dataset, inds));
      };

  typename RansacFunctions<FitType>::InlierMetric inlier_metric = [&](
      const std::vector<std::size_t> &inds, const FitType &fit) {
    const auto pred = fit.predict(subset(dataset.features, inds));
    const auto target = subset(dataset.targets, inds);
    const MetricPredictType prediction = pred.template get<MetricPredictType>();
    return metric(prediction, target);
  };

  typename RansacFunctions<FitType>::ModelMetric model_metric =
      [&](const std::vector<std::size_t> &inds) {
        RegressionDataset<FeatureType> inlier_dataset = subset(dataset, inds);
        const auto inlier_loo = leave_one_out_indexer(inlier_dataset.features);
        double mean_score = model.cross_validate()
                                .scores(metric, inlier_dataset, inlier_loo)
                                .mean();
        return mean_score;
      };

  const auto best_inds = ransac<FitType>(
      fitter, inlier_metric, model_metric, map_values(fold_indexer),
      inlier_threshold, random_sample_size, min_inliers, max_iterations);
  return subset(dataset, best_inds);
}

template <typename ModelType, typename MetricType, typename FeatureType>
struct Fit<Ransac<ModelType, MetricType>, FeatureType> {

  using FitModelType = typename fit_model_type<ModelType, FeatureType>::type;

  Fit(){};

  Fit(const FitModelType &fit_model_) : fit_model(fit_model_){};

  template <typename Archive> void serialize(Archive &archive) {
    archive(cereal::make_nvp("fit_model", fit_model));
  }

  FitModelType fit_model;
};

/*
 * This wraps any other model and performs ransac each time fit is called.
 */
template <typename ModelType, typename MetricType>
class Ransac : public ModelBase<Ransac<ModelType, MetricType>> {
public:
  Ransac(){};

  Ransac(const ModelType &sub_model, const MetricType &metric,
         double inlier_threshold, std::size_t min_inliers,
         std::size_t random_sample_size, std::size_t max_iterations)
      : sub_model_(sub_model), metric_(metric),
        inlier_threshold_(inlier_threshold), min_inliers_(min_inliers),
        random_sample_size_(random_sample_size),
        max_iterations_(max_iterations){};

  static_assert(
      std::is_base_of<EvaluationMetric<Eigen::VectorXd>, MetricType>::value ||
          std::is_base_of<EvaluationMetric<MarginalDistribution>,
                          MetricType>::value ||
          std::is_base_of<EvaluationMetric<JointDistribution>,
                          MetricType>::value,
      "MetricType must be an EvaluationMetric.");

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
  Fit<Ransac<ModelType, MetricType>, FeatureType>
  _fit_impl(const std::vector<FeatureType> &features,
            const MarginalDistribution &targets) const {
    // Remove outliers
    RegressionDataset<FeatureType> dataset(features, targets);
    const auto fold_indexer = leave_one_out_indexer(dataset.features);
    RegressionDataset<FeatureType> inliers =
        ransac(dataset, fold_indexer, sub_model_, metric_, inlier_threshold_,
               random_sample_size_, min_inliers_, max_iterations_);
    // Then generate a fit.
    return Fit<Ransac<ModelType, MetricType>, FeatureType>(
        sub_model_.fit(inliers));
  }

  template <typename PredictFeatureType, typename FitType, typename PredictType>
  PredictType _predict_impl(const std::vector<PredictFeatureType> &features,
                            const FitType &ransac_fit_,
                            PredictTypeIdentity<PredictType> &&) const {
    return ransac_fit_.fit_model.predict(features).template get<PredictType>();
  }

  void save() const;
  void load();

  template <typename Archive> void serialize(Archive &archive) {
    archive(cereal::make_nvp("sub_model", sub_model_));
    archive(cereal::make_nvp("metric", metric_));
    archive(cereal::make_nvp("inlier_threshold", inlier_threshold_));
    archive(cereal::make_nvp("min_inliers", min_inliers_));
    archive(cereal::make_nvp("random_sample_size", random_sample_size_));
    archive(cereal::make_nvp("max_iterations", max_iterations_));
  }

  ModelType sub_model_;
  MetricType metric_;
  double inlier_threshold_;
  std::size_t min_inliers_;
  std::size_t random_sample_size_;
  std::size_t max_iterations_;
};

template <typename ModelType>
template <typename MetricType>
Ransac<ModelType, MetricType> ModelBase<ModelType>::ransac(
    const MetricType &metric, double inlier_threshold, std::size_t min_inliers,
    std::size_t random_sample_size, std::size_t max_iterations) const {
  return Ransac<ModelType, MetricType>(derived(), metric, inlier_threshold,
                                       min_inliers, random_sample_size,
                                       max_iterations);
}

} // namespace albatross

#endif
