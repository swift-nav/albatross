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
#include "evaluate.h"
#include "nlopt.hpp"
#include "tuning_metrics.h"
#include <map>
#include <vector>

namespace albatross {

inline std::string to_string(const nlopt::result result) {
  std::map<nlopt::result, std::string> result_strings = {
      {nlopt::result::FAILURE, "generic failure"},
      {nlopt::result::INVALID_ARGS, "invalid_arguments"},
      {nlopt::result::OUT_OF_MEMORY, "out of memory"},
      {nlopt::result::ROUNDOFF_LIMITED, "roundoff limited"},
      {nlopt::result::FORCED_STOP, "forced stop"},
      {nlopt::result::SUCCESS, "generic success"},
      {nlopt::result::STOPVAL_REACHED, "stop value reached"},
      {nlopt::result::FTOL_REACHED, "ftol reached"},
      {nlopt::result::XTOL_REACHED, "xtol reached"},
      {nlopt::result::MAXEVAL_REACHED, "maxeval reached"},
      {nlopt::result::MAXTIME_REACHED, "maxtime reached"}};
  return result_strings[result];
}

template <class FeatureType> struct TuneModelConfig {
  RegressionModelCreator<FeatureType> model_creator;
  std::vector<RegressionDataset<FeatureType>> datasets;
  TuningMetric<FeatureType> metric;
  TuningMetricAggregator aggregator;
  std::ostream &output_stream;
  nlopt::opt optimizer;

  TuneModelConfig(const RegressionModelCreator<FeatureType> &model_creator_,
                  const RegressionDataset<FeatureType> &dataset_,
                  const TuningMetric<FeatureType> &metric_,
                  const TuningMetricAggregator &aggregator_ = mean_aggregator,
                  std::ostream &output_stream_ = std::cout)
      : model_creator(model_creator_), datasets({dataset_}), metric(metric_),
        aggregator(aggregator_), output_stream(output_stream_), optimizer() {
    set_default_optimizer();
  };

  TuneModelConfig(const RegressionModelCreator<FeatureType> &model_creator_,
                  const std::vector<RegressionDataset<FeatureType>> &datasets_,
                  const TuningMetric<FeatureType> &metric_,
                  const TuningMetricAggregator &aggregator_ = mean_aggregator,
                  std::ostream &output_stream_ = std::cout)
      : model_creator(model_creator_), datasets(datasets_), metric(metric_),
        aggregator(aggregator_), output_stream(output_stream_), optimizer() {
    set_default_optimizer();
  };

  void
  set_default_optimizer(const nlopt::algorithm &algorithm = nlopt::LN_PRAXIS) {
    // The various algorithms in nlopt are coded by the first two characters.
    // In this case LN stands for local, gradient free.
    auto m = model_creator();

    auto tunable_params = m->get_tunable_parameters();
    optimizer = nlopt::opt(algorithm, (unsigned)tunable_params.values.size());
    optimizer.set_ftol_abs(1e-8);
    optimizer.set_ftol_rel(1e-6);
    optimizer.set_lower_bounds(tunable_params.lower_bounds);
    optimizer.set_upper_bounds(tunable_params.upper_bounds);
    // the sensitivity to parameters varies greatly between parameters so
    // terminating based on change in x isn't a great criteria, we only
    // terminate based on xtol if the change is super small.
    optimizer.set_xtol_abs(1e-18);
    optimizer.set_xtol_rel(1e-18);
  }
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

  const TuneModelConfig<FeatureType> config =
      *static_cast<TuneModelConfig<FeatureType> *>(void_tune_config);

  const auto model = config.model_creator();

  model->set_tunable_params_values(x);

  if (!model->params_are_valid()) {
    config.output_stream << "Invalid Parameters:" << std::endl;
    config.output_stream << pretty_param_details(model->get_params())
                         << std::endl;
    assert(false);
  }

  std::vector<double> metrics;
  for (std::size_t i = 0; i < config.datasets.size(); i++) {
    metrics.push_back(config.metric(config.datasets[i], model.get()));
  }
  double metric = config.aggregator(metrics);

  if (std::isnan(metric)) {
    metric = INFINITY;
  }
  config.output_stream << "-------------------" << std::endl;
  config.output_stream << model->pretty_string() << std::endl;
  config.output_stream << "objective: " << metric << std::endl;
  config.output_stream << "-------------------" << std::endl;
  return metric;
}

template <class FeatureType>
ParameterStore
tune_regression_model(const TuneModelConfig<FeatureType> &config) {

  const auto example_model = config.model_creator();
  auto x = example_model->get_tunable_parameters().values;

  assert(x.size());
  nlopt::opt opt = config.optimizer;
  opt.set_min_objective(objective_function<FeatureType>, (void *)&config);
  double minf;
  nlopt::result result = opt.optimize(x, minf);

  // Tell the user what the final parameters were.
  example_model->set_tunable_params_values(x);
  config.output_stream << "==================" << std::endl;
  config.output_stream << "TUNED MODEL PARAMS" << std::endl;
  config.output_stream << "nlopt termination code: " << to_string(result)
                       << std::endl;
  config.output_stream << "==================" << std::endl;
  config.output_stream << example_model->pretty_string() << std::endl;

  return example_model->get_params();
}
} // namespace albatross
#endif
