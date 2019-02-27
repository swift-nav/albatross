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

#ifndef ALBATROSS_TUNING_METRICS_H
#define ALBATROSS_TUNING_METRICS_H

namespace albatross {

template <typename FeatureType>
using TuningMetric = std::function<double(
    const RegressionDataset<FeatureType> &, RegressionModel<FeatureType> *)>;

template <typename FeatureType, typename SubFeatureType = FeatureType>
inline double gp_nll(const RegressionDataset<FeatureType> &dataset,
                     RegressionModel<FeatureType> *model) {
  using SerializableGP =
      SerializableRegressionModel<FeatureType,
                                  GaussianProcessFit<SubFeatureType>>;
  SerializableGP *gp_model = static_cast<SerializableGP *>(model);
  gp_model->fit(dataset);
  GaussianProcessFit<SubFeatureType> model_fit = gp_model->get_fit();

  double nll =
      negative_log_likelihood(dataset.targets.mean, model_fit.train_ldlt);
  nll -= model->prior_log_likelihood();
  return nll;
}

inline double loo_nll(const RegressionDataset<double> &dataset,
                      RegressionModel<double> *model) {
  auto loo_indexer = leave_one_out_indexer(dataset);
  EvaluationMetric<JointDistribution> nll =
      evaluation_metrics::negative_log_likelihood;
  double prior_nll = model->prior_log_likelihood();
  return cross_validated_scores(nll, dataset, loo_indexer, model).sum() -
         prior_nll;
}

inline double loo_rmse(const RegressionDataset<double> &dataset,
                       RegressionModel<double> *model) {
  auto loo_indexer = leave_one_out_indexer(dataset);
  EvaluationMetric<Eigen::VectorXd> rmse =
      evaluation_metrics::root_mean_square_error;
  return cross_validated_scores(rmse, dataset, loo_indexer, model).mean();
}

using TuningMetricAggregator =
    std::function<double(const std::vector<double> &metrics)>;

/*
 * Returns the mean of metrics computed across multiple datasets.
 */
inline double mean_aggregator(const std::vector<double> &metrics) {
  double mean = 0.;
  for (const auto &metric : metrics) {
    mean += metric;
  }
  mean /= static_cast<double>(metrics.size());
  return mean;
}

} // namespace albatross
#endif
