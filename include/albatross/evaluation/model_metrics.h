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

#ifndef ALBATROSS_EVALUATION_MODEL_METRICS_H_
#define ALBATROSS_EVALUATION_MODEL_METRICS_H_

namespace albatross {

template <typename MetricType> class ModelMetric {
public:
  template <typename FeatureType, typename ModelType,
            typename std::enable_if<
                has_valid_call_impl<MetricType, RegressionDataset<FeatureType>,
                                    ModelType>::value,
                int>::type = 0>
  double operator()(const RegressionDataset<FeatureType> &dataset,
                    const ModelBase<ModelType> &model) const {
    return derived()._call_impl(dataset, model);
  }

  template <typename FeatureType, typename ModelType,
            typename std::enable_if<
                !has_valid_call_impl<MetricType, RegressionDataset<FeatureType>,
                                     ModelType>::value,
                int>::type = 0>
  double operator()(const RegressionDataset<FeatureType> &dataset,
                    const ModelBase<ModelType> &model) const =
      delete; // Metric Not Valid for these types.

  template <class Archive> void serialize(Archive &){};

protected:
  /*
   * CRTP Helpers
   */
  MetricType &derived() { return *static_cast<MetricType *>(this); }
  const MetricType &derived() const {
    return *static_cast<const MetricType *>(this);
  }
};

template <typename PredictType = JointDistribution>
struct LeaveOneOutLikelihood
    : public ModelMetric<LeaveOneOutLikelihood<PredictType>> {

  template <typename FeatureType, typename ModelType>
  double _call_impl(const RegressionDataset<FeatureType> &dataset,
                    const ModelBase<ModelType> &model) const {
    NegativeLogLikelihood<PredictType> nll;
    LeaveOneOut loo;
    const auto scores = model.cross_validate().scores(nll, dataset, loo);
    double data_nll = scores.sum();
    double prior_nll = model.prior_log_likelihood();
    return data_nll - prior_nll;
  }
};

struct LeaveOneOutRMSE : public ModelMetric<LeaveOneOutRMSE> {
  template <typename FeatureType, typename ModelType>
  double _call_impl(const RegressionDataset<FeatureType> &dataset,
                    const ModelBase<ModelType> &model) const {
    RootMeanSquareError rmse;
    LeaveOneOut loo;

    double rmse_score =
        model.cross_validate().scores(rmse, dataset, loo).mean();
    return rmse_score;
  }
};
}

#endif /* ALBATROSS_EVALUATION_MODEL_METRICS_H_ */
