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

#ifndef ALBATROSS_CORE_PARAMETER_HANDLING_MIXIN_H
#define ALBATROSS_CORE_PARAMETER_HANDLING_MIXIN_H

namespace albatross {

constexpr double PARAMETER_EPSILON =
    std::numeric_limits<ParameterValue>::epsilon();
constexpr double PARAMETER_MAX = std::numeric_limits<ParameterValue>::max();

struct TunableParameters {
  std::vector<double> values;
  std::vector<double> lower_bounds;
  std::vector<double> upper_bounds;
};

struct Parameter {
  ParameterValue value;
  PriorContainer prior;

  Parameter() : value(), prior(){};
  Parameter(ParameterValue value_) : value(value_), prior() {}

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
    ss << "    " << std::left << std::setw(max_name_length + 1) << pair.first
       << " value: " << std::left << std::setw(12) << pair.second.value
       << " valid: " << std::left << std::setw(3) << pair.second.is_valid()
       << " prior: " << std::setw(15) << prior_name << " bounds: ["
       << pair.second.prior.lower_bound() << ", "
       << pair.second.prior.upper_bound() << "]" << std::endl;
  }
  return ss.str();
}

/*
 * This mixin class is intended to be included an any class which
 * depends on some set of parameters which we want to programatically
 * change for things such as optimization routines / serialization.
 */
class ParameterHandlingMixin {
public:
  ParameterHandlingMixin() : params_(){};
  ParameterHandlingMixin(const ParameterStore &params) : params_(params){};

  virtual ~ParameterHandlingMixin(){};

  void check_param_key(const ParameterKey &key) const {
    const ParameterStore current_params = get_params();
    if (!map_contains(current_params, key)) {
      std::cerr << "Error: Key `" << key << "` not found in parameters: "
                << pretty_params(current_params);
      exit(EXIT_FAILURE);
    }
  }

  /*
   * Provides a safe interface to the parameter values
   */
  void set_params(const ParameterStore &params) {
    for (const auto &pair : params) {
      check_param_key(pair.first);
      unchecked_set_param(pair.first, pair.second);
    }
  }

  void set_params_if_exists(const ParameterStore &params) {
    const ParameterStore current_params = get_params();
    for (const auto &pair : params) {
      if (map_contains(current_params, pair.first)) {
        unchecked_set_param(pair.first, pair.second);
      }
    }
  }

  void set_param_values(const std::map<ParameterKey, ParameterValue> &values) {
    for (const auto &pair : values) {
      check_param_key(pair.first);
      unchecked_set_param(pair.first, pair.second);
    }
  }

  void set_param_values_if_exists(const std::map<ParameterKey, ParameterValue> &values) {
    const ParameterStore current_params = get_params();
    for (const auto &pair : values) {
      if (map_contains(current_params, pair.first)) {
        unchecked_set_param(pair.first, pair.second);
      }
    }
  }

  void set_param_value(const ParameterKey &key, const ParameterValue &value) {
    check_param_key(key);
    unchecked_set_param(key, value);
  }

  void set_param(const ParameterKey &key, const Parameter &param) {
    check_param_key(key);
    unchecked_set_param(key, param);
  }

  // This just avoids the situation where a user would call `set_param`
  // with a double, which may then be viewed by the compiler as the
  // initialization argument for a `Parameter` which would then
  // inadvertently overwrite the prior.
  void set_param(const ParameterKey &key, const ParameterValue &value) {
    set_param_value(key, value);
  }

  void set_prior(const ParameterKey &key, const ParameterPrior &prior) {
    check_param_key(key);
    unchecked_set_prior(key, prior);
  }

  bool params_are_valid() const {
    for (const auto &pair : get_params()) {
      if (!pair.second.is_valid()) {
        return false;
      }
    }
    return true;
  }

  double prior_log_likelihood() const {
    double sum = 0.;
    for (const auto &pair : get_params()) {
      sum += pair.second.prior_log_likelihood();
    }
    return sum;
  }

  /*
   * These methods which collapse a set of vectors to a vector, and
   * set them from a vector facilitate things like tuning in which
   * some function doesn't actually care what any of the parameters
   * correspond to.
   */

  TunableParameters get_tunable_parameters() const {
    TunableParameters output;

    const ParameterStore params = get_params();
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
          assert(false);
        }
        if (v > ub) {
          std::cout << "INVALID PARAMETER: " << pair.first
                    << " expected to be less than " << ub << " but is: " << v
                    << std::endl;
          assert(false);
        }

        bool use_log_scale = pair.second.prior.is_log_scale();
        if (use_log_scale) {
          lb = log(lb);
          ub = log(ub);
          v = log(v);
        }

        output.values.push_back(v);
        output.lower_bounds.push_back(lb);
        output.upper_bounds.push_back(ub);
      }
    }
    return output;
  }

  void set_tunable_params_values(const std::vector<ParameterValue> &x) {
    const ParameterStore params = get_params();
    std::size_t i = 0;
    for (const auto &pair : params) {
      if (!pair.second.is_fixed()) {
        double v = x[i];
        const bool use_log_scale = pair.second.prior.is_log_scale();
        if (use_log_scale) {
          v = exp(v);
        }

        const double lb = pair.second.prior.lower_bound();
        const double ub = pair.second.prior.upper_bound();

        // this silly diversion is to make lint think lb and ub get used;
        if (v < lb) {
          assert(false);
        };
        if (v > ub) {
          assert(false);
        };

        unchecked_set_param(pair.first, v);
        i++;
      }
    }
    // Make sure we used all the parameters that were passed in.
    assert(x.size() == i);
  }

  ParameterValue get_param_value(const ParameterKey &name) const {
    return get_params().at(name).value;
  }

  void unchecked_set_param(const ParameterKey &name,
                           const ParameterValue value) {
    Parameter param = {value, get_params()[name].prior};
    unchecked_set_param(name, param);
  }

  void unchecked_set_prior(const ParameterKey &name,
                           const ParameterPrior &prior) {
    Parameter param = {get_params()[name].value, prior};
    unchecked_set_param(name, param);
  }

  /*
   * For debugging.
   */
  std::string pretty_string() const { return pretty_params(get_params()); }

  /*
   * The following methods are ones that may want to be overriden for
   * clasess that contain nested params (for example).
   */

  virtual ParameterStore get_params() const { return params_; }

  virtual void unchecked_set_param(const ParameterKey &name,
                                   const Parameter &param) {
    params_[name] = param;
  }

protected:
  ParameterStore params_;
};

} // namespace albatross

#endif
