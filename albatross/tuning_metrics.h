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

struct GaussianProcessLikelihoodTuningMetric {

  template <typename FeatureType, typename CovFunc, typename GPImplType>
  double
  operator()(const RegressionDataset<FeatureType> &dataset,
             const GaussianProcessBase<CovFunc, GPImplType> &model) const {
    const auto gp_fit = model.fit(dataset).get_fit();
    double nll =
        negative_log_likelihood(dataset.targets.mean, gp_fit.train_ldlt);
    nll -= model.prior_log_likelihood();
    return nll;
  }
};

template <typename PredictType = JointDistribution>
struct LeaveOneOutLikelihood {

  template <typename FeatureType, typename ModelType>
  double operator()(const RegressionDataset<FeatureType> &dataset,
                    const ModelBase<ModelType> &model) const {
    NegativeLogLikelihood<PredictType> nll;
    LeaveOneOut loo;
    const auto scores = model.cross_validate().scores(nll, dataset, loo);
    double data_nll = scores.sum();
    double prior_nll = model.prior_log_likelihood();
    return data_nll - prior_nll;
  }
};

struct LeaveOneOutRMSE {
  template <typename FeatureType, typename ModelType>
  double operator()(const RegressionDataset<FeatureType> &dataset,
                    const ModelBase<ModelType> &model) const {
    RootMeanSquareError rmse;
    LeaveOneOut loo;
    return model.cross_validate().scores(rmse, dataset, loo).mean();
  }
};

} // namespace albatross
#endif
