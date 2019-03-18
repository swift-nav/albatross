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

namespace albatross {

template <typename ModelType, typename MetricType, class FeatureType>
struct ModelTuner;

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

/*
 * This function API is defined by nlopt, when an optimization algorithm
 * requires the gradient nlopt expects that the grad argument gets
 * modified inside this function.  The TuneModelConfig which holds
 * any information about which functions to call etc, needs to be passed
 * in through a void pointer.
 */
template <typename ModelType, typename MetricType, typename FeatureType>
double objective_function(const std::vector<double> &x,
                          std::vector<double> &grad, void *void_tune_config) {
  if (!grad.empty()) {
    throw std::invalid_argument("The algorithm being used by nlopt requires"
                                "a gradient but one isn't available.");
  }

  const ModelTuner<ModelType, MetricType, FeatureType> config =
      *static_cast<ModelTuner<ModelType, MetricType, FeatureType> *>(
          void_tune_config);

  ModelType model(config.model);
  model.set_tunable_params_values(x);

  if (!model.params_are_valid()) {
    config.output_stream << "Invalid Parameters:" << std::endl;
    config.output_stream << pretty_param_details(model.get_params())
                         << std::endl;
    assert(false);
  }

  std::vector<double> metrics;
  for (std::size_t i = 0; i < config.datasets.size(); i++) {
    metrics.push_back(config.metric(config.datasets[i], model));
  }
  double metric = config.aggregator(metrics);

  if (std::isnan(metric)) {
    metric = INFINITY;
  }
  config.output_stream << "-------------------" << std::endl;
  config.output_stream << model.pretty_string() << std::endl;
  config.output_stream << "objective: " << metric << std::endl;
  config.output_stream << "-------------------" << std::endl;
  return metric;
}

template <typename ModelType, typename MetricType, class FeatureType>
struct ModelTuner {
  ModelType model;
  MetricType metric;
  std::vector<RegressionDataset<FeatureType>> datasets;
  TuningMetricAggregator aggregator;
  std::ostream &output_stream;
  nlopt::opt optimizer;

  ModelTuner(const ModelType &model_, const MetricType &metric_,
             const std::vector<RegressionDataset<FeatureType>> &datasets_,
             const TuningMetricAggregator &aggregator_,
             std::ostream &output_stream_)
      : model(model_), metric(metric_), datasets(datasets_),
        aggregator(aggregator_), output_stream(output_stream_), optimizer() {
    initialize_optimizer();
  };

  ParameterStore tune() {
    auto x = model.get_tunable_parameters().values;

    assert(x.size());
    optimizer.set_min_objective(
        objective_function<ModelType, MetricType, FeatureType>, (void *)this);
    double minf;
    nlopt::result result = optimizer.optimize(x, minf);

    // Tell the user what the final parameters were.
    model.set_tunable_params_values(x);
    output_stream << "==================" << std::endl;
    output_stream << "TUNED MODEL PARAMS" << std::endl;
    output_stream << "nlopt termination code: " << to_string(result)
                  << std::endl;
    output_stream << "==================" << std::endl;
    output_stream << model.pretty_string() << std::endl;

    return model.get_params();
  }

  void
  initialize_optimizer(const nlopt::algorithm &algorithm = nlopt::LN_PRAXIS) {
    // The various algorithms in nlopt are coded by the first two characters.
    // In this case LN stands for local, gradient free.
    auto tunable_params = model.get_tunable_parameters();

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

template <typename ModelType, typename MetricType, typename FeatureType>
auto get_tuner(const ModelType &model, const MetricType &metric,
               const std::vector<RegressionDataset<FeatureType>> &datasets,
               const TuningMetricAggregator &aggregator = mean_aggregator,
               std::ostream &output_stream = std::cout) {
  return ModelTuner<ModelType, MetricType, FeatureType>(
      model, metric, datasets, aggregator, output_stream);
}

template <typename ModelType, typename MetricType, typename FeatureType>
auto get_tuner(const ModelType &model, const MetricType &metric,
               const RegressionDataset<FeatureType> &dataset,
               const TuningMetricAggregator &aggregator = mean_aggregator,
               std::ostream &output_stream = std::cout) {
  std::vector<RegressionDataset<FeatureType>> datasets;
  datasets.emplace_back(dataset);
  return get_tuner(model, metric, datasets, aggregator, output_stream);
}

} // namespace albatross
#endif
