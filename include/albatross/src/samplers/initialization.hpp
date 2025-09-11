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
    try {
      output.push_back(std::stod(s));
    } catch (...) {
      output.push_back(NAN);
      std::cout << "BAD VALUE: " << s << std::endl;
    }
  }
  return output;
}

std::vector<std::map<std::string, double>>
initial_params_from_csv(std::istream &ss) {
  std::vector<std::map<std::string, double>> output;

  std::string line;
  double iteration = 0;

  ALBATROSS_ASSERT(std::getline(ss, line));
  const std::vector<std::string> columns = split_string(line, ',');

  ALBATROSS_ASSERT(columns[0] == "iteration");
  ALBATROSS_ASSERT(columns[1] == "log_probability");
  ALBATROSS_ASSERT(columns[2] == "ensemble_index");

  while (std::getline(ss, line)) {
    const std::vector<double> values = parse_line(line);

    // Only store the params from the last iteration in the file.
    if (values[0] > iteration) {
      iteration = values[0];
      output.clear();
    }

    std::map<std::string, double> param_values;
    ALBATROSS_ASSERT(values.size() == columns.size());
    // Skip the first three columns which contain metadata
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
    ALBATROSS_ASSERT(value_map.size() == param_store.size());
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
std::vector<std::vector<double>>
initial_params_from_jitter(const ParameterStore &params,
                           JitterDistribution &jitter_distribution,
                           std::default_random_engine &gen, std::size_t n = 0) {
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

template <typename ComputeLogProb>
inline EnsembleSamplerState
ensure_finite_initial_state(ComputeLogProb &&compute_log_prob,
                            const EnsembleSamplerState &ensembles,
                            std::default_random_engine &gen) {
  auto all_finite = [](const std::vector<double> &xs) {
    for (const auto &x : xs) {
      if (!std::isfinite(x)) {
        return false;
      }
    }
    return true;
  };

  EnsembleSamplerState output;
  for (const auto &state : ensembles) {
    if (std::isfinite(state.log_prob) && all_finite(state.params)) {
      output.push_back(state);
    }
  }
  ALBATROSS_ASSERT(output.size() > 2 &&
                   "Need at least two finite initial states");

  std::uniform_real_distribution<double> uniform_real(0.0, 1.0);

  while (output.size() < ensembles.size()) {
    const auto random_pair = random_without_replacement(output, 2, gen);

    SamplerState attempt(random_pair[0]);
    for (std::size_t i = 0; i < attempt.params.size(); ++i) {
      const auto a = uniform_real(gen);
      attempt.params[i] += a * (random_pair[1].params[i] - attempt.params[i]);
    }

    attempt.log_prob = compute_log_prob(attempt.params);
    if (std::isfinite(attempt.log_prob)) {
      output.push_back(attempt);
    }
  }
  return output;
}

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_SRC_SAMPLERS_INITIALIZATION_HPP_ */
