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

inline std::vector<std::string> get_columns(const ParameterStore &example) {
  std::vector<std::string> columns;
  columns.push_back("iteration");
  columns.push_back("log_probability");
  columns.push_back("ensemble_index");
  const auto param_names = map_keys(example);
  columns.insert(columns.end(), param_names.begin(), param_names.end());
  return columns;
}

template <typename ModelType>
inline void
write_ensemble_sampler_state(std::ostream &stream, const ModelType &model,
                             const EnsembleSamplerState &ensemble,
                             std::size_t iteration,
                             const std::vector<std::string> &columns) {

  ModelType m(model);
  for (std::size_t i = 0; i < ensemble.size(); ++i) {
    std::map<std::string, std::string> row;

    m.set_tunable_params_values(ensemble[i].params);
    for (const auto &param : m.get_params()) {
      row[param.first] = std::to_string(param.second.value);
    }
    row["iteration"] = std::to_string(iteration);
    row["log_probability"] = std::to_string(ensemble[i].log_prob);
    row["ensemble_index"] = std::to_string(i);
    write_row(stream, row, columns);
  }
}

template <typename ModelType> struct CsvWritingCallback {

  CsvWritingCallback(const ModelType &model_,
                     std::shared_ptr<std::ostream> &stream_)
      : model(model_), stream(stream_){};

  CsvWritingCallback(const ModelType &model_,
                     std::shared_ptr<std::ostream> &&stream_)
      : model(model_), stream(std::move(stream_)){};

  void operator()(std::size_t iteration,
                  const EnsembleSamplerState &ensembles) {
    if (iteration == 0) {
      columns = get_columns(model.get_params());
      write_header(*stream, columns);
    }
    write_ensemble_sampler_state(*stream, model, ensembles, iteration, columns);
  }

  ModelType model;
  std::shared_ptr<std::ostream> stream;
  std::vector<std::string> columns;
};

template <typename ModelType>
CsvWritingCallback<ModelType> get_csv_writing_callback(const ModelType &model,
                                                       std::string path) {
  return CsvWritingCallback<ModelType>(model,
                                       std::make_shared<std::ofstream>(path));
}

template <typename ModelType>
CsvWritingCallback<ModelType>
get_csv_writing_callback(const ModelType &model,
                         std::shared_ptr<std::ostream> &stream) {
  return CsvWritingCallback<ModelType>(model, stream);
}

} // namespace albatross

#endif /* ALBATROSS_SAMPLERS_CALLBACKS_HPP_ */
