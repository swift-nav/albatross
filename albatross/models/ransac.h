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

#include "core/traits.h"
#include "crossvalidation.h"
#include "evaluate.h"
#include "random_utils.h"
#include <random>

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
template <typename FeatureType, typename PredictType>
RegressionDataset<FeatureType>
ransac(const RegressionDataset<FeatureType> &dataset,
       const FoldIndexer &fold_indexer, RegressionModel<FeatureType> *model,
       EvaluationMetric<PredictType> &metric, double inlier_threshold,
       std::size_t random_sample_size, std::size_t min_inliers,
       int max_iterations) {

  using FitType = RegressionModel<FeatureType> *;

  typename RansacFunctions<FitType>::Fitter fitter =
      [&](const std::vector<std::size_t> &inds) {
        RegressionDataset<FeatureType> dataset_subset(
            subset(inds, dataset.features), subset(inds, dataset.targets));
        model->fit(dataset_subset);
        return model;
      };

  typename RansacFunctions<FitType>::InlierMetric inlier_metric =
      [&](const std::vector<std::size_t> &inds, const FitType &fit) {
        const auto pred = fit->predict(subset(inds, dataset.features));
        const auto target = subset(inds, dataset.targets);
        double metric_value = metric(pred, target);
        return metric_value;
      };

  typename RansacFunctions<FitType>::ModelMetric model_metric =
      [&](const std::vector<std::size_t> &inds) {
        RegressionDataset<FeatureType> inlier_dataset(
            subset(inds, dataset.features), subset(inds, dataset.targets));
        const auto inlier_loo = leave_one_out_indexer(inlier_dataset);
        return cross_validated_scores(metric, inlier_dataset, inlier_loo, model)
            .mean();

      };

  const auto best_inds = ransac<FitType>(
      fitter, inlier_metric, model_metric, map_values(fold_indexer),
      inlier_threshold, random_sample_size, min_inliers, max_iterations);
  RegressionDataset<FeatureType> best_dataset(
      subset(best_inds, dataset.features), subset(best_inds, dataset.targets));
  return best_dataset;
}

/*
 * This wraps any other RegressionModel and performs ransac each time fit is
 * called.
 *
 * Note that the model pointer passed into the constructor is NOT const and will
 * be updated any time `fit` is called on this class.
 */
template <typename ModelType, typename FeatureType>
class GenericRansac : public RegressionModel<FeatureType> {
public:
  GenericRansac(ModelType *sub_model, double inlier_threshold,
                std::size_t min_inliers, std::size_t random_sample_size,
                std::size_t max_iterations,
                const IndexerFunction<FeatureType> &indexer_function =
                    leave_one_out_indexer<FeatureType>)
      : sub_model_(sub_model), inlier_threshold_(inlier_threshold),
        min_inliers_(min_inliers), random_sample_size_(random_sample_size),
        max_iterations_(max_iterations), indexer_function_(indexer_function),
        metric_(evaluation_metrics::negative_log_likelihood){};

  std::string get_name() const override {
    std::ostringstream oss;
    oss << "ransac[" << sub_model_->get_name() << "]";
    return oss.str();
  };

  bool has_been_fit() const override { return sub_model_->has_been_fit(); }

  ParameterStore get_params() const override {
    return sub_model_->get_params();
  }

  void unchecked_set_param(const std::string &name,
                           const Parameter &param) override {
    sub_model_->set_param(name, param);
  }

  virtual std::unique_ptr<RegressionModel<FeatureType>>
  ransac_model(double inlier_threshold, std::size_t min_inliers,
               std::size_t random_sample_size,
               std::size_t max_iterations) override {
    assert(false); // "cant ransac a ransac model!"
    return nullptr;
  }

protected:
  void fit_(const std::vector<FeatureType> &features,
            const MarginalDistribution &targets) override {
    RegressionDataset<FeatureType> dataset(features, targets);
    const auto fold_indexer = indexer_function_(dataset);
    RegressionDataset<FeatureType> inliers =
        ransac(dataset, fold_indexer, sub_model_, metric_, inlier_threshold_,
               random_sample_size_, min_inliers_, max_iterations_);
    this->insights_["post_ransac_feature_count"] =
        std::to_string(inliers.features.size());
    this->sub_model_->add_insights(this->insights_);

    if (inliers.features.size() > 0) {
      this->sub_model_->fit(inliers);
    }
  }

  JointDistribution
  predict_(const std::vector<FeatureType> &features) const override {
    return sub_model_->template predict<JointDistribution>(features);
  }

  MarginalDistribution
  predict_marginal_(const std::vector<FeatureType> &features) const override {
    return sub_model_->template predict<MarginalDistribution>(features);
  }

  Eigen::VectorXd
  predict_mean_(const std::vector<FeatureType> &features) const override {
    return sub_model_->template predict<Eigen::VectorXd>(features);
  }

  ModelType *sub_model_;
  double inlier_threshold_;
  std::size_t min_inliers_;
  std::size_t random_sample_size_;
  std::size_t max_iterations_;
  IndexerFunction<FeatureType> indexer_function_;
  EvaluationMetric<JointDistribution> metric_;
};

template <typename FeatureType, typename ModelType>
inline std::unique_ptr<GenericRansac<ModelType, FeatureType>>
make_generic_ransac_model(
    ModelType *model, double inlier_threshold, std::size_t min_inliers,
    std::size_t random_sample_size, std::size_t max_iterations,
    const IndexerFunction<FeatureType> &indexer_function) {
  return std::make_unique<GenericRansac<ModelType, FeatureType>>(
      model, inlier_threshold, min_inliers, random_sample_size, max_iterations,
      indexer_function);
}

} // namespace albatross

#endif
