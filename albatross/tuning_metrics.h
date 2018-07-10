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

#include "core/model.h"
#include "core/serialize.h"
#include "evaluate.h"
#include "models/gp.h"
#include <map>
#include <vector>

namespace albatross {

template <typename FeatureType>
using TuningMetric = std::function<double(
    const RegressionDataset<FeatureType> &, RegressionModel<FeatureType> *)>;

/*
 * Use caution with this metric!  You'll have to manually ensure that
 * the template parameters are correct for the specific model you wish
 * to tune as internally it assumes that the RegressionModel<T>
 * is actually a SerializableRegressionModel<X, GaussianProcessFit<Y>>
 * and if that is not the case at run time a segfault will surely follow.
 *
 * BUT if you do in fact know your regression model is a GP this will
 * result in n^3 versus n^4 complexity.
 */
template <typename FeatureType, typename SubFeatureType = FeatureType>
inline double gp_fast_loo_nll(const RegressionDataset<FeatureType> &dataset,
                              RegressionModel<FeatureType> *model) {
  using SerializableGP =
      SerializableRegressionModel<FeatureType,
                                  GaussianProcessFit<SubFeatureType>>;
  SerializableGP *gp_model = static_cast<SerializableGP *>(model);
  const auto predictions =
      fast_gp_loo_cross_validated_predict(dataset, gp_model);
  const auto deviations = dataset.targets.mean - predictions.mean;
  return negative_log_likelihood(deviations, predictions.covariance) -
         model->prior_log_likelihood();
}

inline double loo_nll(const RegressionDataset<double> &dataset,
                      RegressionModel<double> *model) {
  auto loo_folds = leave_one_out(dataset);
  return cross_validated_scores(evaluation_metrics::negative_log_likelihood,
                                loo_folds, model)
             .sum() -
         model->prior_log_likelihood();
}

inline double loo_rmse(const RegressionDataset<double> &dataset,
                       RegressionModel<double> *model) {
  auto loo_folds = leave_one_out(dataset);
  return cross_validated_scores(evaluation_metrics::root_mean_square_error,
                                loo_folds, model)
      .mean();
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
