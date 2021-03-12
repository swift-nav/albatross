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

template <typename ObjectiveFunction>
inline double objective_function_wrapper(const std::vector<double> &x,
                                         std::vector<double> &grad,
                                         void *objective_func) {
  return (*static_cast<ObjectiveFunction *>(objective_func))(x, grad);
}

template <typename ObjectiveFunction>
inline void set_objective_function(nlopt::opt &optimizer,
                                   ObjectiveFunction &objective) {
  optimizer.set_min_objective(objective_function_wrapper<ObjectiveFunction>,
                              (void *)&objective);
}

inline nlopt::opt
default_optimizer(const ParameterStore &params,
                  const nlopt::algorithm &algorithm = nlopt::LN_SBPLX) {
  // The various algorithms in nlopt are coded by the first two characters.
  // In this case LN stands for local, gradient free.
  const auto tunable_params = get_tunable_parameters(params);

  nlopt::opt optimizer(algorithm, (unsigned)tunable_params.values.size());
  optimizer.set_ftol_abs(1e-8);
  optimizer.set_ftol_rel(1e-6);
  optimizer.set_lower_bounds(tunable_params.lower_bounds);
  optimizer.set_upper_bounds(tunable_params.upper_bounds);
  // the sensitivity to parameters varies greatly between parameters so
  // terminating based on change in x isn't a great criteria, we only
  // terminate based on xtol if the change is super small.
  optimizer.set_xtol_abs(1e-18);
  optimizer.set_xtol_rel(1e-18);
  return optimizer;
}

inline ParameterStore uninformative_params(const std::vector<double> &values) {
  ParameterStore params;
  for (std::size_t i = 0; i < values.size(); ++i) {
    params[std::to_string(i)] = {values[i], UninformativePrior()};
  }
  return params;
}

inline ParameterStore run_optimizer(const ParameterStore &params,
                                    nlopt::opt &optimizer,
                                    std::ostream &output_stream) {

  auto x = get_tunable_parameters(params).values;

  assert(static_cast<std::size_t>(optimizer.get_dimension()) == x.size());

  double minf;
  nlopt::result result = optimizer.optimize(x, minf);

  const auto output = set_tunable_params_values(params, x);
  output_stream << "==================" << std::endl;
  output_stream << "TUNED PARAMS" << std::endl;
  output_stream << "minimum: " << minf << std::endl;
  output_stream << "nlopt termination code: " << to_string(result) << std::endl;
  output_stream << "==================" << std::endl;
  output_stream << pretty_params(output) << std::endl;

  return output;
}

struct GenericTuner {
  ParameterStore initial_params;
  nlopt::opt optimizer;
  std::ostream &output_stream;
  bool use_async;

  GenericTuner(const ParameterStore &initial_params_,
               std::ostream &output_stream_ = std::cout)
      : initial_params(initial_params_), optimizer(),
        output_stream(output_stream_), use_async(false) {
    optimizer = default_optimizer(initial_params);
  };

  GenericTuner(const std::vector<double> &initial_params,
               std::ostream &output_stream_ = std::cout)
      : GenericTuner(uninformative_params(initial_params), output_stream_){};

  template <
      typename ObjectiveFunction,
      std::enable_if_t<is_invocable<ObjectiveFunction, ParameterStore>::value,
                       int> = 0>
  ParameterStore tune(ObjectiveFunction &objective) {

    static_assert(is_invocable_with_result<ObjectiveFunction, double,
                                           ParameterStore>::value,
                  "ObjectiveFunction was expected to take the form `double "
                  "f(const ParameterStore &x)`");

    auto param_wrapped_objective = [&](const std::vector<double> &x,
                                       std::vector<double> &grad) {
      const ParameterStore params =
          set_tunable_params_values(initial_params, x);

      if (!params_are_valid(params)) {
        this->output_stream << "Invalid Parameters:" << std::endl;
        this->output_stream << pretty_param_details(params) << std::endl;
        assert(false);
      }

      double metric = objective(params);

      if (grad.size() > 0) {

        const auto tunable = get_tunable_parameters(initial_params);

        const auto grad_eval =
            compute_gradient(objective, params, metric, use_async);
        this->output_stream << "gradient" << std::endl;
        for (std::size_t i = 0; i < grad_eval.size(); ++i) {
          this->output_stream << "  " << tunable.names[i] << " : "
                              << grad_eval[i] << std::endl;
          grad[i] = grad_eval[i];
        }
      }

      if (std::isnan(metric)) {
        metric = INFINITY;
      }
      this->output_stream << "-------------------" << std::endl;
      this->output_stream << pretty_params(params) << std::endl;
      this->output_stream << "objective: " << metric << std::endl;
      this->output_stream << "-------------------" << std::endl;
      return metric;
    };

    set_objective_function(optimizer, param_wrapped_objective);

    return run_optimizer(initial_params, optimizer, output_stream);
  }

  template <
      typename ObjectiveFunction,
      std::enable_if_t<
          is_invocable<ObjectiveFunction, std::vector<double>>::value, int> = 0>
  std::vector<double> tune(ObjectiveFunction &objective) {

    static_assert(is_invocable_with_result<ObjectiveFunction, double,
                                           std::vector<double>>::value,
                  "ObjectiveFunction was expected to take the form `double "
                  "f(const std::vector<double> &x)`");

    auto grad_free_objective = [&](const std::vector<double> &x,
                                   std::vector<double> &grad) {
      double metric = objective(x);

      if (grad.size() > 0) {
        const auto grad_eval =
            compute_gradient(objective, x, metric, use_async);
        this->output_stream << "gradient" << std::endl;
        for (std::size_t i = 0; i < grad_eval.size(); ++i) {
          this->output_stream << "  " << i << " : " << grad_eval[i]
                              << std::endl;
          grad[i] = grad_eval[i];
        }
      }

      if (std::isnan(metric)) {
        metric = INFINITY;
      }
      this->output_stream << "-------------------" << std::endl;
      for (std::size_t i = 0; i < x.size(); ++i) {
        this->output_stream << "  " << i << " : " << x[i] << std::endl;
      }
      this->output_stream << "objective: " << metric << std::endl;
      this->output_stream << "-------------------" << std::endl;
      return metric;
    };

    set_objective_function(optimizer, grad_free_objective);
    const auto params = run_optimizer(initial_params, optimizer, output_stream);
    return get_tunable_parameters(params).values;
  }

  template <
      typename ObjectiveFunction,
      std::enable_if_t<is_invocable<ObjectiveFunction, Eigen::VectorXd>::value,
                       int> = 0>
  Eigen::VectorXd tune(ObjectiveFunction &objective) {

    static_assert(is_invocable_with_result<ObjectiveFunction, double,
                                           Eigen::VectorXd>::value,
                  "ObjectiveFunction was expected to take the form `double "
                  "f(const Eigen::VectorXd &x)`");

    auto objective_converted_to_eigen = [&](const std::vector<double> &x) {
      std::vector<double> x_copy(x);
      const Eigen::Map<Eigen::VectorXd> eigen_x(
          &x_copy[0], static_cast<Eigen::Index>(x_copy.size()));

      return objective(eigen_x);
    };

    auto vector_output = tune(objective_converted_to_eigen);
    const Eigen::Map<Eigen::VectorXd> eigen_output(
        &vector_output[0], static_cast<Eigen::Index>(vector_output.size()));
    return eigen_output;
  }

  template <typename ObjectiveFunction,
            std::enable_if_t<
                !is_invocable<ObjectiveFunction, std::vector<double>>::value &&
                    !is_invocable<ObjectiveFunction, ParameterStore>::value &&
                    !is_invocable<ObjectiveFunction, Eigen::VectorXd>::value,
                int> = 0>
  void tune(ObjectiveFunction &objective)
      ALBATROSS_FAIL(ObjectiveFunction,
                     "Unsupported function signature for ObjectiveFunction");
};

template <typename ModelType, typename MetricType, class FeatureType>
struct ModelTuner {
  ModelType model;
  MetricType metric;
  std::vector<RegressionDataset<FeatureType>> datasets;
  TuningMetricAggregator aggregator;
  std::ostream &output_stream;
  nlopt::opt optimizer;

  static_assert(is_model_metric<MetricType, FeatureType, ModelType>::value,
                "metric is not valid for this feature/model pair");

  ModelTuner(const ModelType &model_, const MetricType &metric_,
             const std::vector<RegressionDataset<FeatureType>> &datasets_,
             const TuningMetricAggregator &aggregator_,
             std::ostream &output_stream_)
      : model(model_), metric(metric_), datasets(datasets_),
        aggregator(aggregator_), output_stream(output_stream_), optimizer() {
    optimizer = default_optimizer(model.get_params());
  };

  ParameterStore tune() {

    auto objective = [&](const ParameterStore &params) {
      ModelType m(model);
      m.set_params(params);
      std::vector<double> metrics;
      for (std::size_t i = 0; i < this->datasets.size(); i++) {
        metrics.push_back(this->metric(this->datasets[i], m));
      }
      return this->aggregator(metrics);
    };

    GenericTuner generic_tuner(model.get_params(), output_stream);
    generic_tuner.optimizer = optimizer;
    return generic_tuner.tune(objective);
  }

  void
  initialize_optimizer(const nlopt::algorithm &algorithm = nlopt::LN_SBPLX) {
    optimizer = default_optimizer(model.get_params(), algorithm);
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
