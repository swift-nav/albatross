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

#include <map>
#include <vector>
#include "core/model.h"
#include "nlopt.hpp"

namespace albatross {

std::vector<ParameterValue> transform_parameters(
    const std::vector<ParameterValue>& x) {
  std::vector<ParameterValue> transformed(x.size());
  std::transform(x.begin(), x.end(), transformed.begin(), log);
  return transformed;
}

std::vector<ParameterValue> inverse_parameters(
    const std::vector<ParameterValue>& x) {
  std::vector<ParameterValue> inverted(x.size());
  std::transform(x.begin(), x.end(), inverted.begin(), exp);
  return inverted;
}

template <class Predictor>
using TuningMetric = std::function<double(const RegressionDataset<Predictor> &,
                                          RegressionModel<Predictor> *)>;

template <class Predictor>
struct TuneModelConfg {
  RegressionModelCreator<Predictor> model_creator;
  RegressionDataset<Predictor> dataset;
  TuningMetric<Predictor> metric;

  TuneModelConfg(const RegressionModelCreator<Predictor> &model_creator_,
                 const RegressionDataset<Predictor> &dataset_,
                 const TuningMetric<Predictor> &metric_)
      : model_creator(model_creator_), dataset(dataset_), metric(metric_){};
};

template <class Predictor>
double objective_function(const std::vector<double> &x,
                          std::vector<double> &grad, void *void_tune_config) {
  if (!grad.empty()) {
    throw std::invalid_argument(
        "The algorithm being used by nlopt requires"
        "a gradient but one isn't available.");
  }

  const TuneModelConfg<Predictor> config =
      *static_cast<TuneModelConfg<Predictor> *>(void_tune_config);

  const auto model = config.model_creator();

  model->set_params_from_vector(inverse_parameters(x));

  const double metric = config.metric(config.dataset, model.get());

  std::cout << "-------------------" << std::endl;
  std::cout << model->pretty_string() << std::endl;
  std::cout << "objective: " << metric << std::endl;
  std::cout << "-------------------" << std::endl;
  return metric;
}

template <class Predictor>
ParameterStore tune_regression_model(
    const RegressionModelCreator<Predictor> &model_creator,
    const RegressionDataset<Predictor> &dataset,
    const TuningMetric<Predictor> &metric) {
  const auto example_model = model_creator();

  TuneModelConfg<Predictor> config(model_creator, dataset, metric);

  auto x = transform_parameters(example_model->get_params_as_vector());

  assert(x.size());

  // The various algorithms in nlopt are coded by the first two characters.
  // In this case LN stands for local, gradient free.
  nlopt::opt opt(nlopt::LN_PRAXIS, (unsigned)x.size());
  opt.set_min_objective(objective_function<Predictor>, (void *)&config);
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
  std::cout << "==================" << std::endl;
  std::cout << "TUNED MODEL PARAMS" << std::endl;
  std::cout << "==================" << std::endl;
  std::cout << example_model->pretty_string() << std::endl;

  return example_model->get_params();
}

}
#endif
