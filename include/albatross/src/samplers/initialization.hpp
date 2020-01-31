/*
 * Copyright (C) 2020 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef INCLUDE_ALBATROSS_SRC_SAMPLERS_INITIALIZATION_HPP_
#define INCLUDE_ALBATROSS_SRC_SAMPLERS_INITIALIZATION_HPP_

namespace albatross {

inline std::vector<std::string> split_string(const std::string &s,
                                             char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream token_stream(s);
  while (std::getline(token_stream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

std::vector<double> parse_line(const std::string &line) {
  std::vector<double> output;
  for (const auto &s : split_string(line, ',')) {
    output.push_back(std::stod(s));
  }
  return output;
}

std::vector<std::map<std::string, double>>
initial_params_from_csv(std::istream &ss) {

  std::vector<std::map<std::string, double>> output;

  std::string line;
  double iteration = 0;

  assert(std::getline(ss, line));
  const std::vector<std::string> columns = split_string(line, ',');

  assert(columns[0] == "iteration");
  assert(columns[1] == "log_probability");
  assert(columns[2] == "ensemble_index");

  while (std::getline(ss, line)) {
    const std::vector<double> values = parse_line(line);
    if (values[0] > iteration) {
      iteration = values[0];
      output.clear();
    }

    std::map<std::string, double> param_values;
    assert(values.size() == columns.size());
    for (std::size_t i = 3; i < columns.size(); ++i) {
      param_values[columns[i]] = values[i];
    }

    output.emplace_back(param_values);
  }

  return output;
}

std::vector<std::vector<double>>
initial_params_from_csv(const ParameterStore &param_store, std::istream &ss) {

  const auto all_params = initial_params_from_csv(ss);

  std::vector<std::vector<double>> output;
  for (const auto &value_map : all_params) {
    assert(value_map.size() == param_store.size());
    ParameterStore params(param_store);
    for (const auto &value_pair : value_map) {
      params[value_pair.first].value = ensure_value_within_bounds(
          params[value_pair.first], value_pair.second);
    }

    output.emplace_back(get_tunable_parameters(params).values);
  }

  return output;
}

template <typename JitterDistribution>
std::vector<std::vector<double>> initial_params_from_jitter(
    const ParameterStore &params, JitterDistribution &jitter_distribution,
    std::default_random_engine &gen, std::size_t n = -1) {

  n = std::max(n, 2 * params.size() + 1);

  std::vector<std::vector<double>> output;
  std::vector<double> double_params = get_tunable_parameters(params).values;
  output.push_back(double_params);
  for (std::size_t i = 0; i < n - 1; ++i) {

    std::vector<double> perturbed(double_params);
    for (auto &d : perturbed) {
      d += jitter_distribution(gen);
    };

    output.push_back(perturbed);
  }
  return output;
}

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_SRC_SAMPLERS_INITIALIZATION_HPP_ */
