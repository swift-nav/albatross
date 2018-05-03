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

#ifndef ALBATROSS_TUNE_H
#define ALBATROSS_TUNE_H

#include "core/model.h"
#include "nlopt.hpp"
#include <map>
#include <vector>

namespace albatross {

inline std::vector<ParameterValue>
transform_parameters(const std::vector<ParameterValue> &x) {
  std::vector<ParameterValue> transformed(x.size());
  std::transform(x.begin(), x.end(), transformed.begin(), log);
  return transformed;
}

inline std::vector<ParameterValue>
inverse_parameters(const std::vector<ParameterValue> &x) {
  std::vector<ParameterValue> inverted(x.size());
  std::transform(x.begin(), x.end(), inverted.begin(), exp);
  return inverted;
}

template <class FeatureType>
using TuningMetric = std::function<double(
    const RegressionDataset<FeatureType> &, RegressionModel<FeatureType> *)>;

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

template <class FeatureType> struct TuneModelConfg {
  RegressionModelCreator<FeatureType> model_creator;
  std::vector<RegressionDataset<FeatureType>> datasets;
  TuningMetric<FeatureType> metric;
  TuningMetricAggregator aggregator;
  std::ostream &output_stream;

  TuneModelConfg(const RegressionModelCreator<FeatureType> &model_creator_,
                 const RegressionDataset<FeatureType> &dataset_,
                 const TuningMetric<FeatureType> &metric_,
                 const TuningMetricAggregator &aggregator_ = mean_aggregator,
                 std::ostream &output_stream_ = std::cout)
      : model_creator(model_creator_), datasets({dataset_}), metric(metric_),
        aggregator(aggregator_), output_stream(output_stream_){};

  TuneModelConfg(const RegressionModelCreator<FeatureType> &model_creator_,
                 const std::vector<RegressionDataset<FeatureType>> &datasets_,
                 const TuningMetric<FeatureType> &metric_,
                 const TuningMetricAggregator &aggregator_ = mean_aggregator,
                 std::ostream &output_stream_ = std::cout)
      : model_creator(model_creator_), datasets(datasets_), metric(metric_),
        aggregator(aggregator_), output_stream(output_stream_){};
};

/*
 * This function API is defined by nlopt, when an optimization algorithm
 * requires the gradient nlopt expects that the grad argument gets
 * modified inside this function.  The TuneModelConfig which holds
 * any information about which functions to call etc, needs to be passed
 * in through a void pointer.
 */
template <class FeatureType>
double objective_function(const std::vector<double> &x,
                          std::vector<double> &grad, void *void_tune_config) {
  if (!grad.empty()) {
    throw std::invalid_argument("The algorithm being used by nlopt requires"
                                "a gradient but one isn't available.");
  }

  const TuneModelConfg<FeatureType> config =
      *static_cast<TuneModelConfg<FeatureType> *>(void_tune_config);

  const auto model = config.model_creator();

  model->set_params_from_vector(inverse_parameters(x));

  std::vector<double> metrics;
  for (std::size_t i = 0; i < config.datasets.size(); i++) {
    metrics.push_back(config.metric(config.datasets[i], model.get()));
  }
  double metric = config.aggregator(metrics);

  config.output_stream << "-------------------" << std::endl;
  config.output_stream << model->pretty_string() << std::endl;
  config.output_stream << "objective: " << metric << std::endl;
  config.output_stream << "-------------------" << std::endl;
  return metric;
}

template <class FeatureType>
ParameterStore
tune_regression_model(const TuneModelConfg<FeatureType> &config) {

  const auto example_model = config.model_creator();
  auto x = transform_parameters(example_model->get_params_as_vector());

  assert(x.size());

  // The various algorithms in nlopt are coded by the first two characters.
  // In this case LN stands for local, gradient free.
  nlopt::opt opt(nlopt::LN_PRAXIS, (unsigned)x.size());
  opt.set_min_objective(objective_function<FeatureType>, (void *)&config);
  opt.set_ftol_abs(1e-8);
  opt.set_ftol_rel(1e-6);
  // the sensitivity to parameters varies greatly between parameters so
  // terminating based on change in x isn't a great criteria, we only
  // terminate based on xtol if the change is super small.
  opt.set_xtol_rel(1e-8);
  double minf;
  opt.optimize(x, minf);

  // Tell the user what the final parameters were.
  example_model->set_params_from_vector(inverse_parameters(x));
  config.output_stream << "==================" << std::endl;
  config.output_stream << "TUNED MODEL PARAMS" << std::endl;
  config.output_stream << "==================" << std::endl;
  config.output_stream << example_model->pretty_string() << std::endl;

  return example_model->get_params();
}
}
#endif
