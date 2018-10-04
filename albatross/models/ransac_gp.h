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

#ifndef ALBATROSS_MODELS_RANSAC_GP_H
#define ALBATROSS_MODELS_RANSAC_GP_H

#include "crossvalidation.h"
#include "gp.h"
#include "outlier.h"
#include <random>

namespace albatross {

template <typename FeatureType, typename CovarianceFunction>
class RansacGaussianProcessRegression
    : public GaussianProcessRegression<FeatureType, CovarianceFunction> {
public:
  using BaseModel = GaussianProcessRegression<FeatureType, CovarianceFunction>;
  using FitType = GaussianProcessFit<FeatureType>;

  RansacGaussianProcessRegression(CovarianceFunction &covariance_function,
                                  double inlier_threshold_,
                                  std::size_t min_inliers_,
                                  std::size_t min_features_,
                                  std::size_t max_iterations_)
      : BaseModel(covariance_function), inlier_threshold(inlier_threshold_),
        min_inliers(min_inliers_), min_features(min_features_),
        max_iterations(max_iterations_){};

  template <typename Archive> void save(Archive &archive) const {
    archive(cereal::base_class<BaseModel>(this),
            cereal::make_nvp("inlier_threshold", inlier_threshold),
            cereal::make_nvp("min_inliers", min_inliers),
            cereal::make_nvp("min_features", min_features),
            cereal::make_nvp("max_iterations", max_iterations));
  }

  template <typename Archive> void load(Archive &archive) {
    archive(cereal::base_class<BaseModel>(this),
            cereal::make_nvp("inlier_threshold", inlier_threshold),
            cereal::make_nvp("min_inliers", min_inliers),
            cereal::make_nvp("min_features", min_features),
            cereal::make_nvp("max_iterations", max_iterations));
  }

protected:
  virtual std::vector<JointDistribution> cross_validated_predictions_(
      const RegressionDataset<FeatureType> &dataset,
      const FoldIndexer &fold_indexer,
      const detail::PredictTypeIdentity<JointDistribution> &identity) override {
    const auto folds = folds_from_fold_indexer(dataset, fold_indexer);
    return this->template cross_validated_predictions<JointDistribution>(folds);
  }

  virtual std::vector<MarginalDistribution> cross_validated_predictions_(
      const RegressionDataset<FeatureType> &dataset,
      const FoldIndexer &fold_indexer,
      const detail::PredictTypeIdentity<MarginalDistribution> &identity)
      override {
    const auto folds = folds_from_fold_indexer(dataset, fold_indexer);
    return this->template cross_validated_predictions<MarginalDistribution>(
        folds);
  }

  virtual std::vector<Eigen::VectorXd> cross_validated_predictions_(
      const RegressionDataset<FeatureType> &dataset,
      const FoldIndexer &fold_indexer,
      const detail::PredictTypeIdentity<PredictMeanOnly> &identity) override {
    const auto folds = folds_from_fold_indexer(dataset, fold_indexer);
    return this->template cross_validated_predictions<PredictMeanOnly>(folds);
  }

  FitType
  serializable_fit_(const std::vector<FeatureType> &features,
                    const MarginalDistribution &targets) const override {
    /*
     * This function is basically just setting up the call to `ransac` by
     * defining `fitter`, `outlier_metric` and `model_metric` functions
     * that can perform each operation efficiently for Gaussian processes.
     */

    Eigen::MatrixXd cov =
        symmetric_covariance(this->covariance_function_, features);

    if (max_iterations < 1) {
      return GaussianProcessFit<FeatureType>(features, cov, targets);
    }

    struct FitAndIndices {
      FitType model_fit;
      Indexer fit_indices;
    };

    std::function<FitAndIndices(const Indexer &)> fitter =
        [&](const Indexer &indexer) {
          const auto train_features = subset(indexer, features);
          const auto train_targets = subset(indexer, targets);
          auto train_cov = symmetric_subset(indexer, cov);
          const FitAndIndices fit_and_indices = {
              GaussianProcessFit<FeatureType>(train_features, train_cov,
                                              train_targets),
              indexer};
          return fit_and_indices;
        };

    std::function<double(const Indexer &, const FitAndIndices &)>
        outlier_metric = [&](const Indexer &test_indices,
                             const FitAndIndices &fit_and_indices) {
          const auto cross_cov =
              subset(fit_and_indices.fit_indices, test_indices, cov);
          const auto pred_cov = symmetric_subset(test_indices, cov);
          const auto pred = predict_from_covariance_and_fit(
              cross_cov, pred_cov, fit_and_indices.model_fit);
          const auto target = subset(test_indices, targets);
          double metric_value = metric(pred, target);
          return metric_value;
        };

    std::function<double(const Indexer &)> model_metric = [&](
        const Indexer &inliers) {
      RegressionDataset<FeatureType> inlier_dataset(subset(inliers, features),
                                                    subset(inliers, targets));
      const FoldIndexer inlier_loo = leave_one_out_indexer(inlier_dataset);

      BaseModel model;
      model.set_params(this->get_params());
      double mean_metric = cross_validated_scores<double>(
                               metric, inlier_dataset, inlier_loo, &model)
                               .mean();
      return mean_metric;
    };

    // Now that the ransac functions are defined we can actually perform
    // RANSAC, then return the subsequent outlier free fit.
    const RegressionDataset<FeatureType> dataset(features, targets);
    const auto loo_indexer = leave_one_out_indexer(dataset);

    auto inliers = ransac<FitAndIndices>(
        fitter, outlier_metric, model_metric, map_values(loo_indexer),
        inlier_threshold, min_features, min_inliers, max_iterations);

    const auto inlier_features = subset(inliers, features);
    const auto inlier_targets = subset(inliers, targets);
    const auto inlier_cov = symmetric_subset(inliers, cov);

    return GaussianProcessFit<FeatureType>(inlier_features, inlier_cov,
                                           inlier_targets);
  }

  double inlier_threshold;
  std::size_t min_inliers;
  std::size_t min_features;
  std::size_t max_iterations;
  EvaluationMetric<JointDistribution> metric =
      albatross::evaluation_metrics::negative_log_likelihood;
};

template <typename FeatureType, typename CovFunc>
std::unique_ptr<RansacGaussianProcessRegression<FeatureType, CovFunc>>
ransac_gp_pointer_from_covariance(CovFunc covariance_function,
                                  double inlier_threshold = 1.,
                                  std::size_t min_inliers = 3,
                                  std::size_t min_features = 3,
                                  std::size_t max_iterations = 20) {
  return std::make_unique<
      RansacGaussianProcessRegression<FeatureType, CovFunc>>(
      covariance_function, inlier_threshold, min_inliers, min_features,
      max_iterations);
};

} // namespace albatross

#endif
