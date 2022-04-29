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
    albatross::set_params(params, param_lookup_function());
  }

  void set_params_if_exists(const ParameterStore &params) {
    albatross::set_params_if_exists(params, param_lookup_function());
  }

  void set_param_values(const std::map<ParameterKey, ParameterValue> &values) {
    albatross::set_param_values(values, param_lookup_function());
  }

  void set_param_values_if_exists(
      const std::map<ParameterKey, ParameterValue> &values) {
    albatross::set_param_values_if_exists(values, param_lookup_function());
  }

  void set_param_value(const ParameterKey &key, const ParameterValue &value) {
    albatross::set_param_value(key, value, param_lookup_function());
  }

  void set_param(const ParameterKey &key, const Parameter &param) {
    albatross::set_param(key, param, param_lookup_function());
  }

  // This just avoids the situation where a user would call `set_param`
  // with a double, which may then be viewed by the compiler as the
  // initialization argument for a `Parameter` which would then
  // inadvertently overwrite the prior.
  void set_param(const ParameterKey &key, const ParameterValue &value) {
    albatross::set_param_value(key, value, param_lookup_function());
  }

  void set_prior(const ParameterKey &key, const ParameterPrior &prior) {
    albatross::set_param_prior(key, prior, param_lookup_function());
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
    this->set_params(modified_params);
  }

  ParameterValue get_param_value(const ParameterKey &name) const {
    return get_params().at(name).value;
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

  virtual Parameter *get_param_pointer(const ParameterKey &name) {
    return param_lookup(name, &params_);
  }

  std::function<Parameter *(const ParameterKey &)> param_lookup_function() {
    return [this](const auto &k) { return this->get_param_pointer(k); };
  }

protected:
  ParameterStore params_;
};

} // namespace albatross

#endif
