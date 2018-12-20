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
#include "ransac.h"
#include <random>

namespace albatross {

template <typename FeatureType> struct FitAndIndices {
  GaussianProcessFit<FeatureType> model_fit;
  Indexer fit_indices;
};

// Calculate a numerically stable log determinant of a symmetric matrix using
// Cholesky
inline double log_determinant_of_symmetric(const Eigen::MatrixXd &M) {
  double log_determinant = 0;
  const Eigen::LDLT<Eigen::MatrixXd> ldlt(M);
  const auto diagonal = ldlt.vectorD();
  for (Eigen::Index i = 0; i < M.rows(); ++i) {
    log_determinant += log(diagonal(i));
  }
  return log_determinant;
}

/*
 * Loosely describes the entropy of a model given the
 * covariance matrix.  This can be thought of as describing
 * the dispersion of the data.
 *
 * https://en.wikipedia.org/wiki/Differential_entropy
 */
inline double differential_entropy(const Eigen::MatrixXd &cov) {
  double k = static_cast<double>(cov.rows());
  double ld = log_determinant_of_symmetric(cov);
  return 0.5 * (k * (1 + log(2 * M_PI) + ld));
}

template <typename FeatureType>
inline typename RansacFunctions<FitAndIndices<FeatureType>>::Fitter
get_gp_ransac_fitter(const std::vector<FeatureType> &features,
                     const MarginalDistribution &targets,
                     const Eigen::MatrixXd &cov) {
  return [&](const Indexer &indexer) {
    const auto train_features = subset(indexer, features);
    const auto train_targets = subset(indexer, targets);
    auto train_cov = symmetric_subset(indexer, cov);
    const FitAndIndices<FeatureType> fit_and_indices = {
        GaussianProcessFit<FeatureType>(train_features, train_cov,
                                        train_targets),
        indexer};
    return fit_and_indices;
  };
}

template <typename FeatureType>
inline typename RansacFunctions<FitAndIndices<FeatureType>>::InlierMetric
get_gp_ransac_inlier_metric(const std::vector<FeatureType> &features,
                            const MarginalDistribution &targets,
                            const Eigen::MatrixXd &cov,
                            const EvaluationMetric<JointDistribution> &metric) {
  return [&](const Indexer &test_indices,
             const FitAndIndices<FeatureType> &fit_and_indices) {
    const auto cross_cov =
        subset(fit_and_indices.fit_indices, test_indices, cov);
    const auto pred_cov = symmetric_subset(test_indices, cov);
    const auto pred = predict_from_covariance_and_fit(
        cross_cov, pred_cov, fit_and_indices.model_fit);
    const auto target = subset(test_indices, targets);
    double metric_value = metric(pred, target);
    return metric_value;
  };
}

template <typename FeatureType>
inline typename RansacFunctions<FitAndIndices<FeatureType>>::ModelMetric
get_gp_ransac_model_entropy_metric(const std::vector<FeatureType> &features,
                                   const MarginalDistribution &targets,
                                   const Eigen::MatrixXd &cov) {
  return [&](const Indexer &inliers) {
    // Here the metric for two models of the same dimensions will
    // result in preferring one which has more dispersed data.
    auto inlier_cov = symmetric_subset(inliers, cov);
    double metric_value = differential_entropy(inlier_cov);
    return metric_value;
  };
}

/*
 * This wraps any other RegressionModel and performs ransac each time fit is
 * called.
 *
 * Note that the model pointer passed into the constructor is NOT const and will
 * be updated any time `fit` is called on this class.
 */
template <typename FeatureType, typename CovarianceType>
class GaussianProcessRansac : public RegressionModel<FeatureType> {
public:
  using ModelType = GaussianProcessRegression<FeatureType, CovarianceType>;

  GaussianProcessRansac(ModelType *sub_model, double inlier_threshold,
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
    oss << "gp_ransac[" << sub_model_->get_name() << "]";
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

    Eigen::MatrixXd cov = sub_model_->compute_covariance(features);

    RegressionDataset<FeatureType> dataset(features, targets);
    const auto fold_indexer = indexer_function_(dataset);

    const auto fitter = get_gp_ransac_fitter(features, targets, cov);

    const auto inlier_metric =
        get_gp_ransac_inlier_metric(features, targets, cov, metric_);

    const auto model_metric =
        get_gp_ransac_model_entropy_metric(features, targets, cov);

    const auto inliers = ransac<FitAndIndices<FeatureType>>(
        fitter, inlier_metric, model_metric, map_values(fold_indexer),
        inlier_threshold_, random_sample_size_, min_inliers_, max_iterations_);
    const auto inlier_features = subset(inliers, features);
    const auto inlier_targets = subset(inliers, targets);
    const auto inlier_cov = symmetric_subset(inliers, cov);

    const GaussianProcessFit<FeatureType> fit(inlier_features, inlier_cov,
                                              inlier_targets);
    this->insights_["post_ransac_feature_count"] =
        std::to_string(inliers.size());
    this->sub_model_->set_fit(fit);
    this->sub_model_->add_insights(this->insights_);
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

template <typename FeatureType, typename CovarianceType>
inline std::unique_ptr<GaussianProcessRansac<FeatureType, CovarianceType>>
make_gp_ransac_model(
    GaussianProcessRegression<FeatureType, CovarianceType> *model,
    double inlier_threshold, std::size_t min_inliers,
    std::size_t random_sample_size, std::size_t max_iterations,
    const IndexerFunction<FeatureType> &indexer_function) {
  return std::make_unique<GaussianProcessRansac<FeatureType, CovarianceType>>(
      model, inlier_threshold, min_inliers, random_sample_size, max_iterations,
      indexer_function);
}

} // namespace albatross

#endif
