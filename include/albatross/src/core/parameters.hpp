/*
 * Copyright (C) 2022 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef INCLUDE_ALBATROSS_SRC_CORE_PARAMETERS_HPP_
#define INCLUDE_ALBATROSS_SRC_CORE_PARAMETERS_HPP_

namespace albatross {

constexpr double PARAMETER_EPSILON =
    std::numeric_limits<ParameterValue>::epsilon();
constexpr double PARAMETER_MAX = std::numeric_limits<ParameterValue>::max();

struct TunableParameters {
  std::vector<std::string> names;
  std::vector<double> values;
  std::vector<double> lower_bounds;
  std::vector<double> upper_bounds;
};

struct Parameter {
  ParameterValue value;
  PriorContainer prior;

  Parameter() : value(), prior(){};
  explicit Parameter(ParameterValue value_) : value(value_), prior() {}

  Parameter(ParameterValue value_, const PriorContainer &prior_)
      : value(value_), prior(prior_){};

  template <typename PriorType,
            typename std::enable_if<
                is_in_variant<PriorType, PossiblePriors>::value, int>::type = 0>
  Parameter(ParameterValue value_, const PriorType &prior_)
      : value(value_), prior(prior_){};

  bool operator==(const ParameterValue &other_value) const {
    return (value == other_value);
  }

  bool operator==(const Parameter &other) const {
    return (value == other.value && prior == other.prior);
  }

  bool operator!=(const Parameter &other) const { return !operator==(other); }

  bool within_bounds() const {
    return (value >= prior.lower_bound() && value <= prior.upper_bound());
  }

  bool is_valid() const { return within_bounds(); }

  bool is_fixed() const { return prior.is_fixed(); }

  double prior_log_likelihood() const { return prior.log_pdf(value); }
};

/*
 * Prints out a set of parameters in a way that is both
 * readable and can be easily copy/pasted into code.
 */
inline std::string pretty_params(const ParameterStore &params) {
  std::ostringstream ss;
  ss << std::setprecision(12);
  ss << std::scientific;
  ss << "{" << std::endl;
  for (const auto &pair : params) {
    ss << "    {\"" << pair.first << "\", " << pair.second.value << "},"
       << std::endl;
  }
  ss << "};" << std::endl;
  return ss.str();
}

inline std::string pretty_priors(const ParameterStore &params) {
  std::ostringstream ss;
  ss << "PRIORS:" << std::endl;
  for (const auto &pair : params) {
    std::string prior_name;
    prior_name = pair.second.prior.get_name();
    ss << "    \"" << pair.first << "\": " << prior_name << std::endl;
  }
  return ss.str();
}

inline std::string pretty_param_details(const ParameterStore &params) {
  std::ostringstream ss;

  auto compare_size = [](const auto &x, const auto &y) {
    return x.first.size() < y.first.size();
  };
  auto max_name_length =
      std::max_element(params.begin(), params.end(), compare_size)
          ->first.size();

  for (const auto &pair : params) {
    std::string prior_name;
    prior_name = pair.second.prior.get_name();
    ss << "    " << std::left
       << std::setw(static_cast<int>(max_name_length + 1)) << pair.first
       << " value: " << std::left << std::setw(12) << pair.second.value
       << " valid: " << std::left << std::setw(3) << pair.second.is_valid()
       << " prior: " << std::setw(15) << prior_name << " bounds: ["
       << pair.second.prior.lower_bound() << ", "
       << pair.second.prior.upper_bound() << "]" << std::endl;
  }
  return ss.str();
}

inline TunableParameters get_tunable_parameters(const ParameterStore &params) {
  TunableParameters output;

  for (const auto &pair : params) {
    if (!pair.second.is_fixed()) {
      double v = pair.second.value;
      double lb = pair.second.prior.lower_bound();
      double ub = pair.second.prior.upper_bound();

      // Without these checks nlopt will fail in a much more obscure way.
      if (v < lb) {
        std::cout << "INVALID PARAMETER: " << pair.first
                  << " expected to be greater than " << lb << " but is: " << v
                  << std::endl;
        ALBATROSS_ASSERT(false);
      }
      if (v > ub) {
        std::cout << "INVALID PARAMETER: " << pair.first
                  << " expected to be less than " << ub << " but is: " << v
                  << std::endl;
        ALBATROSS_ASSERT(false);
      }

      bool use_log_scale = pair.second.prior.is_log_scale();
      if (use_log_scale) {
        lb = log(lb);
        ub = log(ub);
        v = log(v);
      }

      output.names.push_back(pair.first);
      output.values.push_back(v);
      output.lower_bounds.push_back(lb);
      output.upper_bounds.push_back(ub);
    }
  }
  return output;
}

inline double ensure_value_within_bounds(const Parameter &param,
                                         const double value) {
  const double lb = param.prior.lower_bound();
  if (value < lb) {
    return lb;
  };

  const double ub = param.prior.upper_bound();
  if (value > ub) {
    return ub;
  };

  return value;
}

inline ParameterStore
set_tunable_params_values(const ParameterStore &params,
                          const std::vector<ParameterValue> &x,
                          const bool force_bounds = true) {
  ParameterStore output(params);
  std::size_t i = 0;
  for (const auto &pair : params) {
    if (!pair.second.is_fixed()) {
      double v = x[i];
      const bool use_log_scale = pair.second.prior.is_log_scale();
      if (use_log_scale) {
        v = exp(v);
      }
      if (force_bounds) {
        v = ensure_value_within_bounds(pair.second, v);
      }

      output[pair.first].value = v;
      i++;
    }
  }
  // Make sure we used all the parameters that were passed in.
  ALBATROSS_ASSERT(x.size() == i);
  return output;
}

inline bool params_are_valid(const ParameterStore &params) {
  for (const auto &pair : params) {
    if (!pair.second.is_valid()) {
      return false;
    }
  }
  return true;
}

inline bool modify_param(const ParameterKey &name,
                         const std::function<void(Parameter *)> &operation,
                         ParameterStore *params, bool assert_exists = true) {
  const auto it = params->find(name);
  if (it != params->end()) {
    operation(&it->second);
    return true;
  }

  if (assert_exists) {
    std::cerr << "Error: Parameter `" << name << "` not found";
    ALBATROSS_ASSERT(false && "Attempt to modify a param which doesn't exist");
  }
  return false;
}

inline void set_param(const ParameterKey &name, const Parameter &param,
                      ParameterStore *params, bool assert_exists = true) {
  modify_param(
      name, [&](Parameter *p) { (*p) = param; }, params, assert_exists);
}

inline void set_param_value(const ParameterKey &name,
                            const ParameterValue &value, ParameterStore *params,
                            bool assert_exists = true) {
  modify_param(
      name, [&](Parameter *p) { p->value = value; }, params, assert_exists);
}

inline void set_param_prior(const ParameterKey &name,
                            const ParameterPrior &prior, ParameterStore *params,
                            bool assert_exists = true) {
  modify_param(
      name, [&](Parameter *p) { p->prior = prior; }, params, assert_exists);
}

inline void set_params(const ParameterStore &input_params,
                       ParameterStore *params) {
  for (const auto &pair : input_params) {
    set_param(pair.first, pair.second, params);
  }
}

inline void
set_param_values(const std::map<ParameterKey, ParameterValue> &param_values,
                 ParameterStore *params) {
  for (const auto &pair : param_values) {
    set_param_value(pair.first, pair.second, params);
  }
}

// These set_*_if_exists methods return a bool indicating whether the
// parameter was set or not.

inline bool set_param_if_exists(const ParameterKey &name,
                                const Parameter &param,
                                ParameterStore *params) {
  return modify_param(
      name, [&](Parameter *p) { (*p) = param; }, params, false);
}

inline bool set_param_value_if_exists(const ParameterKey &name,
                                      const ParameterValue &value,
                                      ParameterStore *params) {
  return modify_param(
      name, [&](Parameter *p) { p->value = value; }, params, false);
}

inline bool set_param_prior_if_exists(const ParameterKey &name,
                                      const ParameterPrior &prior,
                                      ParameterStore *params) {
  return modify_param(
      name, [&](Parameter *p) { p->prior = prior; }, params, false);
}

inline bool set_params_if_exists(const ParameterStore &input_params,
                                 ParameterStore *params) {
  bool all_set = input_params.size() > 0;
  for (const auto &pair : input_params) {
    all_set &= set_param_if_exists(pair.first, pair.second, params);
  }
  return all_set;
}

inline bool set_param_values_if_exists(
    const std::map<ParameterKey, ParameterValue> &param_values,
    ParameterStore *params) {
  bool all_set = param_values.size() > 0;
  for (const auto &pair : param_values) {
    all_set &= set_param_value_if_exists(pair.first, pair.second, params);
  }
  return all_set;
}

inline double parameter_prior_log_likelihood(const ParameterStore &params) {
  double sum = 0.;
  for (const auto &pair : params) {
    sum += pair.second.prior_log_likelihood();
  }
  return sum;
}

} // namespace albatross

#endif /* INCLUDE_ALBATROSS_SRC_CORE_PARAMETERS_HPP_ */
