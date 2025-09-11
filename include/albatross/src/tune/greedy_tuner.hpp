/*
 * Copyright (C) 2021 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef ALBATROSS_SRC_TUNE_GREEDY_TUNER_HPP_
#define ALBATROSS_SRC_TUNE_GREEDY_TUNER_HPP_

namespace albatross {

/*
 * The greedy_tune method is an alternative to the other tuning approaches
 * in albatross.  It was designed for coarse tuning of computationally
 * complex objective functions.  The method uses multi-threading to drive
 * a relatively simple tuning approach which works by:
 *
 *   - Picking one of the parameters.
 *   - Using the range defined by the prior to produce new guesses.
 *   - Evaluating the objective at all the guesses (one guess per thread).
 *   - Picking the best parameter value.
 *   - Repeating for the next parameter.
 *
 * There are almost certainly no global convergence guarantees with
 * this algorithm, but in practice it seems to do a good job of coarsely
 * tuning the objective.  One strategy would be to start with an iteration
 * of this greedy approach and follow up with one of the methods in tune.hpp
 */

namespace details {

inline std::vector<double> query_ratios(std::size_t n) {
  const double log_min = -5.;
  const double log_max = -0.1;
  if (n == 1) {
    return {log_min};
  }

  const double step = (log_max - log_min) / (cast::to_double(n) - 1);
  std::vector<double> output;
  for (std::size_t i = 0; i < n; ++i) {
    output.push_back(std::pow(10, log_min + cast::to_double(i) * step));
  }

  return output;
}

inline std::vector<double> get_queries(double value, double low, double high,
                                       std::size_t n) {
  if (std::isinf(high)) {
    high = 1e8;
  }
  if (std::isinf(low)) {
    low = -1e8;
  }

  const double low_range = value - low;
  const double high_range = high - value;

  std::vector<double> queries;

  if (value == low) {
    // only search higher since we're at the lower bound
    const auto ratios = query_ratios(2 * n);
    for (std::size_t i = 0; i < ratios.size(); ++i) {
      queries.push_back(value + high_range * ratios[i]);
    }
  } else if (value == high) {
    // only search lower since we're at the upper bound
    const auto ratios = query_ratios(2 * n);
    for (std::size_t i = 0; i < ratios.size(); ++i) {
      queries.push_back(value - low_range * ratios[ratios.size() - 1 - i]);
    }
  } else {
    // search in both directions
    const auto ratios = query_ratios(n);
    for (std::size_t i = 0; i < ratios.size(); ++i) {
      queries.push_back(value - low_range * ratios[ratios.size() - 1 - i]);
    }
    for (std::size_t i = 0; i < ratios.size(); ++i) {
      queries.push_back(value + high_range * ratios[i]);
    }
  }

  return queries;
}

inline ParameterStore set_tunable_param(const ParameterStore &params,
                                        std::size_t i, double val) {
  auto perturbed = get_tunable_parameters(params);
  perturbed.values[i] = val;
  return set_tunable_params_values(params, perturbed.values, true);
};
}  // namespace details

template <typename Function>
inline ParameterStore greedy_tune(Function evaluate_function,
                                  const ParameterStore &params,
                                  std::size_t n_queries_each_direction = 4,
                                  std::size_t n_iterations = 10,
                                  ThreadPool *threads = nullptr,
                                  std::ostream *os = &std::cout) {
  static_assert(
      has_call_operator<Function, ParameterStore>::value,
      "evaluate_function must have a single ParameterStore argument.");

  albatross::TunableParameters tunable = get_tunable_parameters(params);

  if (os) {
    (*os) << "Will be tuning the following:" << std::endl;
    for (std::size_t i = 0; i < tunable.names.size(); ++i) {
      (*os) << "    " << tunable.names[i] << "    [ ";
      const auto queries = details::get_queries(
          tunable.values[i], tunable.lower_bounds[i], tunable.upper_bounds[i],
          n_queries_each_direction);
      for (const auto &v : queries) {
        (*os)
            << details::set_tunable_param(params, i, v)[tunable.names[i]].value
            << ", ";
      }
      (*os) << " ]" << std::endl;
    }
    (*os) << "Initial params:" << std::endl;
    (*os) << albatross::pretty_params(params) << std::endl;
  }

  double best_value = HUGE_VAL;
  ParameterStore best_params(params);

  for (std::size_t iter = 0; iter < n_iterations; ++iter) {
    for (std::size_t i = 0; i < tunable.names.size(); ++i) {
      tunable = get_tunable_parameters(best_params);
      auto values = details::get_queries(
          tunable.values[i], tunable.lower_bounds[i], tunable.upper_bounds[i],
          n_queries_each_direction);
      if (iter == 0 && i == 0) {
        // on the very first iteration we need to include the initial params
        // because they may be optimal.  We could do this before starting
        // tuning, but then we'd have to wait a full function evaluation
        // before making any progress.
        values.push_back(tunable.values[i]);
      }

      const auto get_params = [&](double v) {
        return details::set_tunable_param(best_params, i, v);
      };

      const auto proposed_params = apply(values, get_params);

      if (os) {
        (*os) << "NEXT ATTEMPTS: " << tunable.names[i] << " : ";
        for (const auto &p : proposed_params) {
          (*os) << p.at(tunable.names[i]).value << ",";
        }
        (*os) << std::endl;
      }

      const auto evaluations =
          apply(proposed_params, evaluate_function, threads);

      if (os) {
        (*os) << "EVALUATIONS: " << std::endl;
      }
      for (std::size_t j = 0; j < proposed_params.size(); ++j) {
        if (os) {
          (*os) << "    " << tunable.names[i] << " = "
                << proposed_params[j].at(tunable.names[i]).value << "   :   "
                << evaluations[j];
        }
        if (evaluations[j] < best_value) {
          best_params = proposed_params[j];
          best_value = evaluations[j];
          if (os) {
            (*os) << "   BEST SO FAR";
          }
        }
        if (os) {
          (*os) << std::endl;
        }
      }

      if (os) {
        (*os) << "===============" << std::endl;
        (*os) << "BEST_VALUE : " << best_value << std::endl;
        (*os) << albatross::pretty_params(best_params) << std::endl;
        (*os) << "===============" << std::endl;
      }
    }
  }
  return best_params;
}

}  // namespace albatross

#endif /* ALBATROSS_SRC_TUNE_GREEDY_TUNER_HPP_ */
