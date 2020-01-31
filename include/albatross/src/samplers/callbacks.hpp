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

#ifndef ALBATROSS_SAMPLERS_CALLBACKS_HPP_
#define ALBATROSS_SAMPLERS_CALLBACKS_HPP_

namespace albatross {

struct NullCallback {
  void operator()(std::size_t, const EnsembleSamplerState &){};
};

inline std::vector<std::string>
get_sampler_csv_columns(const ParameterStore &example) {
  std::vector<std::string> columns;
  columns.push_back("iteration");
  columns.push_back("log_probability");
  columns.push_back("ensemble_index");
  const auto param_names = map_keys(example);
  columns.insert(columns.end(), param_names.begin(), param_names.end());
  return columns;
}

inline void write_ensemble_sampler_state(
    std::ostream &stream, const ParameterStore &param_store,
    const EnsembleSamplerState &ensemble, std::size_t iteration,
    const std::vector<std::string> &columns) {

  for (std::size_t i = 0; i < ensemble.size(); ++i) {
    std::map<std::string, std::string> row;

    const auto params =
        set_tunable_params_values(ensemble[i].params, param_store);
    for (const auto &param : params) {
      row[param.first] = std::to_string(param.second.value);
    }
    row["iteration"] = std::to_string(iteration);
    row["log_probability"] = std::to_string(ensemble[i].log_prob);
    row["ensemble_index"] = std::to_string(i);
    write_row(stream, row, columns);
  }
}

struct MaximumLikelihoodTrackingCallback {

  MaximumLikelihoodTrackingCallback(const ParameterStore &param_store_,
                                    std::shared_ptr<std::ostream> &stream_)
      : param_store(param_store_), stream(stream_){};

  MaximumLikelihoodTrackingCallback(const ParameterStore &param_store_,
                                    std::shared_ptr<std::ostream> &&stream_)
      : param_store(param_store_), stream(std::move(stream_)){};

  void operator()(std::size_t iteration,
                  const EnsembleSamplerState &ensembles) {
    for (const auto &state : ensembles) {
      if (state.log_prob > max_ll) {
        max_ll = state.log_prob;
        param_store = set_tunable_params_values(state.params, param_store);
        (*stream) << "===================" << std::endl;
        (*stream) << "Iteration: " << iteration << std::endl;
        (*stream) << "LL: " << max_ll << std::endl;
        (*stream) << pretty_params(param_store) << std::endl;
      }
    }
  }

  double max_ll = -HUGE_VAL;
  ParameterStore param_store;
  std::shared_ptr<std::ostream> stream;
};

struct CsvWritingCallback {

  CsvWritingCallback(const ParameterStore &param_store_,
                     std::shared_ptr<std::ostream> &stream_)
      : param_store(param_store_), stream(stream_){};

  CsvWritingCallback(const ParameterStore &param_store_,
                     std::shared_ptr<std::ostream> &&stream_)
      : param_store(param_store_), stream(std::move(stream_)){};

  void operator()(std::size_t iteration,
                  const EnsembleSamplerState &ensembles) {
    if (iteration == 0) {
      columns = get_sampler_csv_columns(param_store);
      write_header(*stream, columns);
    }
    write_ensemble_sampler_state(*stream, param_store, ensembles, iteration,
                                 columns);
  }

  ParameterStore param_store;
  std::shared_ptr<std::ostream> stream;
  std::vector<std::string> columns;
};

template <typename ModelType>
CsvWritingCallback get_csv_writing_callback(const ModelType &model,
                                            std::string path) {
  return CsvWritingCallback(model.get_params(),
                            std::make_shared<std::ofstream>(path));
}

template <typename ModelType>
CsvWritingCallback
get_csv_writing_callback(const ModelType &model,
                         std::shared_ptr<std::ostream> &stream) {
  return CsvWritingCallback(model.get_params(), stream);
}

} // namespace albatross

#endif /* ALBATROSS_SAMPLERS_CALLBACKS_HPP_ */
