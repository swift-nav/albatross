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

DEFINE_CLASS_METHOD_TRAITS(get_params);
DEFINE_CLASS_METHOD_TRAITS(set_param);

template <typename T> class is_param_handler {
  template <typename C,
            typename std::enable_if<
                has_get_params<C>::value &&
                    has_set_param<C, ParameterKey, Parameter>::value,
                int>::type = 0>
  static std::true_type test(C *);
  template <typename> static std::false_type test(...);

public:
  static constexpr bool value = decltype(test<T>(0))::value;
};

} // namespace details

// These unchecked setting methods assume the parameter exists, if
// the parameter does not exist we'll be relying on safety checks
// within the ParameterHandler which may or may not exist.  Some
// implementations may fail hard if a paramter doesn't exist, others
// may perform a null operation.  To get more predictable behavior
// stick to the set_param* methods.

template <typename ParameterHandler,
          typename std::enable_if_t<
              details::is_param_handler<ParameterHandler>::value, int> = 0>
inline void unchecked_set_param(const ParameterKey &name,
                                const Parameter &param,
                                ParameterHandler *param_handler) {
  param_handler->set_param(name, param);
}

template <typename ParameterHandler,
          typename std::enable_if_t<
              details::is_param_handler<ParameterHandler>::value, int> = 0>
inline void unchecked_set_params(const ParameterStore &params,
                                 ParameterHandler *param_handler) {
  for (const auto &pair : params) {
    unchecked_set_param(pair.first, pair.second, param_handler);
  }
}

// These set_* methods first get all the available parameters.  This
// makes them relatively inefficient but provides the ability to do
// a level of safety checks to make sure parameters exist and deal
// with them accordingly.

template <typename ParameterHandler,
          typename std::enable_if_t<
              details::is_param_handler<ParameterHandler>::value, int> = 0>
inline void set_param_value(const ParameterKey &name,
                            const ParameterValue &value,
                            ParameterHandler *param_handler) {
  auto params = param_handler->get_params();
  set_param_value(name, value, &params);
  unchecked_set_param(name, params.at(name), param_handler);
}

template <typename ParameterHandler,
          typename std::enable_if_t<
              details::is_param_handler<ParameterHandler>::value, int> = 0>
inline void set_param_prior(const ParameterKey &name,
                            const ParameterPrior &prior,
                            ParameterHandler *param_handler) {
  auto params = param_handler->get_params();
  set_param_prior(name, prior, &params);
  unchecked_set_param(name, params.at(name), param_handler);
}

template <typename ParameterHandler,
          typename std::enable_if_t<
              details::is_param_handler<ParameterHandler>::value, int> = 0>
inline void set_params(const ParameterStore &input_params,
                       ParameterHandler *param_handler) {
  auto params = param_handler->get_params();
  set_params(input_params, &params);
  unchecked_set_params(params, param_handler);
}

template <typename ParameterHandler>
inline void
set_param_values(const std::map<ParameterKey, ParameterValue> &input_params,
                 ParameterHandler *param_handler) {
  auto params = param_handler->get_params();
  set_param_values(input_params, &params);
  unchecked_set_params(params, param_handler);
}

template <typename ParameterHandler,
          typename std::enable_if_t<
              details::is_param_handler<ParameterHandler>::value, int> = 0>
inline bool set_param_if_exists(const ParameterKey &name,
                                const Parameter &param,
                                ParameterHandler *param_handler) {
  auto params = param_handler->get_params();
  if (!set_param_if_exists(name, param, &params)) {
    return false;
  }
  unchecked_set_param(name, params.at(name), param_handler);
  return true;
}

template <typename ParameterHandler,
          typename std::enable_if_t<
              details::is_param_handler<ParameterHandler>::value, int> = 0>
inline bool set_param_value_if_exists(const ParameterKey &name,
                                      const ParameterValue &value,
                                      ParameterHandler *param_handler) {
  auto params = param_handler->get_params();
  if (!set_param_value_if_exists(name, value, &params)) {
    return false;
  }
  unchecked_set_param(name, params.at(name), param_handler);
  return true;
}

template <typename ParameterHandler,
          typename std::enable_if_t<
              details::is_param_handler<ParameterHandler>::value, int> = 0>
inline bool set_params_if_exists(const ParameterStore &input_params,
                                 ParameterHandler *param_handler) {
  auto params = param_handler->get_params();
  const bool all_exist = set_params_if_exists(input_params, &params);
  unchecked_set_params(params, param_handler);
  return all_exist;
}

template <typename ParameterHandler,
          typename std::enable_if_t<
              details::is_param_handler<ParameterHandler>::value, int> = 0>
inline bool set_param_values_if_exists(
    const std::map<ParameterKey, ParameterValue> &input_params,
    ParameterHandler *param_handler) {
  auto params = param_handler->get_params();
  const bool all_exist = set_param_values_if_exists(input_params, &params);
  unchecked_set_params(params, param_handler);
  return all_exist;
}

namespace details {

inline bool
variadic_set_param_if_exists(const ParameterKey &key ALBATROSS_UNUSED,
                             const Parameter &param ALBATROSS_UNUSED) {
  return false;
}

template <typename ParameterHandler, typename... Args,
          typename std::enable_if_t<
              details::is_param_handler<ParameterHandler>::value, int> = 0>
inline bool
variadic_set_param_if_exists(const ParameterKey &key, const Parameter &param,
                             ParameterHandler *param_handler, Args... args) {
  return (set_param_if_exists(key, param, param_handler) ||
          variadic_set_param_if_exists(key, param, args...));
}

} // namespace details

template <typename... Args>
inline bool set_param_if_exists_in_any(const ParameterKey &key,
                                       const Parameter &param, Args... args) {
  return details::variadic_set_param_if_exists(key, param, args...);
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

  virtual void set_param(const ParameterKey &name, const Parameter &param) {
    albatross::set_param(name, param, &params_);
  }

protected:
  ParameterStore params_;
};

} // namespace albatross

#endif
