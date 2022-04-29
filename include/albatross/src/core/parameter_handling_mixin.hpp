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
      assert(false);
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

  void set_param_values_if_exists(
      const std::map<ParameterKey, ParameterValue> &values) {
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
    return albatross::params_are_valid(this->get_params());
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
    return albatross::get_tunable_parameters(this->get_params());
  }

  void set_tunable_params_values(const std::vector<ParameterValue> &x,
                                 bool force_bounds = true) {

    const auto modified_params =
        albatross::set_tunable_params_values(get_params(), x, force_bounds);

    for (const auto &pair : modified_params) {
      unchecked_set_param(pair.first, pair.second);
    }
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
