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

namespace details {

// DEFINE_CLASS_METHOD_TRAITS(get_param_pointer);
DEFINE_CLASS_METHOD_TRAITS(on_parameter_change);

} // namespace details

template <typename ParameterHandler>
inline std::function<Parameter *(const ParameterKey &)>
param_lookup_function(ParameterHandler *parameter_handler) {
  return [parameter_handler](const auto &k) {
    return parameter_handler->get_param_pointer(k);
  };
}

template <
    typename ParameterHandler,
    typename std::enable_if_t<
        details::has_on_parameter_change<ParameterHandler>::value, int> = 0>
inline void call_on_parameter_change(ParameterHandler *param_handler) {
  param_handler->on_parameter_change();
}

template <
    typename ParameterHandler,
    typename std::enable_if_t<
        !details::has_on_parameter_change<ParameterHandler>::value, int> = 0>
inline void call_on_parameter_change(ParameterHandler *param_handler) {}

template <typename ParameterHandler>
inline void set_param(const ParameterKey &name, const Parameter &param,
                      ParameterHandler *param_handler) {
  set_param(name, param, param_lookup_function(param_handler));
  call_on_parameter_change(param_handler);
}

template <typename ParameterHandler>
inline void set_param_value(const ParameterKey &name,
                            const ParameterValue &value,
                            ParameterHandler *param_handler) {
  set_param_value(name, value, param_lookup_function(param_handler));
  call_on_parameter_change(param_handler);
}

template <typename ParameterHandler>
inline void set_param_prior(const ParameterKey &name,
                            const ParameterPrior &prior,
                            ParameterHandler *param_handler) {
  set_param_prior(name, prior, param_lookup_function(param_handler));
  call_on_parameter_change(param_handler);
}

template <typename ParameterHandler>
inline void set_param_value_if_exists(const ParameterKey &name,
                                      const ParameterValue &value,
                                      ParameterHandler *param_handler) {
  set_param_value_if_exists(name, value, param_lookup_function(param_handler));
  call_on_parameter_change(param_handler);
}

template <typename ParameterHandler>
inline void set_param_if_exists(const ParameterKey &name,
                                const ParameterValue &value,
                                ParameterHandler *param_handler) {
  set_param_if_exists(name, value, param_lookup_function(param_handler));
  call_on_parameter_change(param_handler);
}

template <typename ParameterHandler>
inline void set_params(const ParameterStore &params,
                       ParameterHandler *param_handler) {
  set_params(params, param_lookup_function(param_handler));
  call_on_parameter_change(param_handler);
}

template <typename ParameterHandler>
inline void set_params_if_exists(const ParameterStore &params,
                                 ParameterHandler *param_handler) {
  set_params_if_exists(params, param_lookup_function(param_handler));
  call_on_parameter_change(param_handler);
}

template <typename ParameterHandler>
inline void
set_param_values(const std::map<ParameterKey, ParameterValue> &params,
                 ParameterHandler *param_handler) {
  set_param_values(params, param_lookup_function(param_handler));
  call_on_parameter_change(param_handler);
}

template <typename ParameterHandler>
inline void
set_param_values_if_exists(const std::map<ParameterKey, ParameterValue> &params,
                           ParameterHandler *param_handler) {
  set_param_values_if_exists(params, param_lookup_function(param_handler));
  call_on_parameter_change(param_handler);
}

/*
 * This mixin class is intended to be included an any class which
 * depends on some set of parameters which we want to programmatically
 * change for things such as optimization routines / serialization.
 */
class ParameterHandlingMixin {
public:
  ParameterHandlingMixin() : params_(){};
  ParameterHandlingMixin(const ParameterStore &params) : params_(params){};

  virtual ~ParameterHandlingMixin(){};

  /*
   * Provides a safe interface to the parameter values
   */
  void set_params(const ParameterStore &params) {
    return albatross::set_params(params, this);
  }

  void set_params_if_exists(const ParameterStore &params) {
    albatross::set_params_if_exists(params, this);
  }

  void set_param_values(const std::map<ParameterKey, ParameterValue> &values) {
    albatross::set_param_values(values, this);
  }

  void set_param_values_if_exists(
      const std::map<ParameterKey, ParameterValue> &values) {
    albatross::set_param_values_if_exists(values, this);
  }

  void set_param_value(const ParameterKey &key, const ParameterValue &value) {
    albatross::set_param_value(key, value, this);
  }

  void set_param(const ParameterKey &key, const Parameter &param) {
    albatross::set_param(key, param, this);
  }

  // This just avoids the situation where a user would call `set_param`
  // with a double, which may then be viewed by the compiler as the
  // initialization argument for a `Parameter` which would then
  // inadvertently overwrite the prior.
  void set_param(const ParameterKey &key, const ParameterValue &value) {
    albatross::set_param_value(key, value, this);
  }

  void set_prior(const ParameterKey &key, const ParameterPrior &prior) {
    albatross::set_param_prior(key, prior, this);
  }

  bool params_are_valid() const {
    return albatross::params_are_valid(this->get_params());
  }

  double prior_log_likelihood() const {
    return albatross::parameter_prior_log_likelihood(get_params());
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
    on_parameter_change();
  }

  ParameterValue get_param_value(const ParameterKey &name) const {
    return get_params().at(name).value;
  }

  /*
   * For debugging.
   */
  std::string pretty_string() const { return pretty_params(get_params()); }

  /*
   * The following methods are ones that may want to be overridden for
   * classes that contain nested params (for example).
   */

  virtual ParameterStore get_params() const { return params_; }

  virtual Parameter *get_param_pointer(const ParameterKey &name) {
    return param_lookup(name, &params_);
  }

  /*
   * Sometimes the parameters held in a model need to get converted
   * to an internal representation.  For example, if a parameter
   * represented a grid spacing you might want to update a precomputed
   * grid if the grid spacing were to change.  This callback provides
   * a way to make sure that happens when the parameters change.
   */
  virtual void on_parameter_change() {
    albatross::params_are_valid(get_params());
  };

protected:
  ParameterStore params_;
};

} // namespace albatross

#endif
