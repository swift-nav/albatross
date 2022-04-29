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

#ifndef ALBATROSS_CORE_PARAMETER_MACROS_H
#define ALBATROSS_CORE_PARAMETER_MACROS_H

/*
 * The for each functionality was taken from:
 *   https://codecraft.co/2014/11/25/variadic-macros-tricks/
 */

// Accept any number of args >= N, but expand to just the Nth one.
#define _GET_NTH_ARG(_1, _2, _3, _4, _5, _6, _7, _8, _9, N, ...) N

// Define some macros to help us create overrides based on the
// arity of a for-each-style macro.
#define _fe_0(_call, ...)
#define _fe_1(_call, x) _call(x)
#define _fe_2(_call, x, ...) _call(x) _fe_1(_call, __VA_ARGS__)
#define _fe_3(_call, x, ...) _call(x) _fe_2(_call, __VA_ARGS__)
#define _fe_4(_call, x, ...) _call(x) _fe_3(_call, __VA_ARGS__)
#define _fe_5(_call, x, ...) _call(x) _fe_4(_call, __VA_ARGS__)
#define _fe_6(_call, x, ...) _call(x) _fe_5(_call, __VA_ARGS__)
#define _fe_7(_call, x, ...) _call(x) _fe_6(_call, __VA_ARGS__)
#define _fe_8(_call, x, ...) _call(x) _fe_7(_call, __VA_ARGS__)

#define CALL_MACRO_X_FOR_EACH(x, ...)                                          \
  _GET_NTH_ARG("ignored", ##__VA_ARGS__, _fe_8, _fe_7, _fe_6, _fe_5, _fe_4,    \
               _fe_3, _fe_2, _fe_1, _fe_0)                                     \
  (x, ##__VA_ARGS__)

/*
 * These macros build up a function that when called with,
 *
 *   DEFINE_GET_PARAMS(1, 2, ...)
 *
 * builds a code block that looks like:
 *
 *   ParameterStore get_params() const override {
 *     return {{"$1", $1},
 *             {"$2", $2},
 *             ...
 *             };
 *    }
 */
#define ADD_MAP_ELEMENT(x) {#x, x},

#define BUILD_MAP(...)                                                         \
  { CALL_MACRO_X_FOR_EACH(ADD_MAP_ELEMENT, ##__VA_ARGS__) }

#define DEFINE_GET_PARAMS(...)                                                 \
  ParameterStore get_params() const override { return BUILD_MAP(__VA_ARGS__); };

/*
 * These macros build up a function that when called with,
 *
 *   DEFINE_SET_PARAMS_UNCHECKED(1, 2, ...)
 *
 * builds a code block that looks like:
 *
 *   void unchecked_set_param (const std::string &key,
 *                             const ParameterValue &value) override {
 *     if (key == "$1") {
 *       $1 = value;
 *     } else if (key == "$2") {
 *       $2 = value;
 *     } else if {
 *     ...
 *     } else {
 *       assert(false);
 *     }
 *   }
 */

/*
 * Similar to the CALL_MACRO_X_FOR_EACH macros, this builds
 * an if else tree
 */
#define _build_if_0(_cond, _action, ...)
#define _build_if_1(_cond, _action, x)                                         \
  if (_cond(x)) {                                                              \
    _action(x);                                                                \
  } else {                                                                     \
    return nullptr;                                                            \
  };
#define _build_if_2(_cond, _action, x, ...)                                    \
  if (_cond(x)) {                                                              \
    _action(x);                                                                \
  } else                                                                       \
    _build_if_1(_cond, _action, __VA_ARGS__)
#define _build_if_3(_cond, _action, x, ...)                                    \
  if (_cond(x)) {                                                              \
    _action(x);                                                                \
  } else                                                                       \
    _build_if_2(_cond, _action, __VA_ARGS__)
#define _build_if_4(_cond, _action, x, ...)                                    \
  if (_cond(x)) {                                                              \
    _action(x);                                                                \
  } else                                                                       \
    _build_if_3(_cond, _action, __VA_ARGS__)
#define _build_if_5(_cond, _action, x, ...)                                    \
  if (_cond(x)) {                                                              \
    _action(x);                                                                \
  } else                                                                       \
    _build_if_4(_cond, _action, __VA_ARGS__)
#define _build_if_6(_cond, _action, x, ...)                                    \
  if (_cond(x)) {                                                              \
    _action(x);                                                                \
  } else                                                                       \
    _build_if_5(_cond, _action, __VA_ARGS__)
#define _build_if_7(_cond, _action, x, ...)                                    \
  if (_cond(x)) {                                                              \
    _action(x);                                                                \
  } else                                                                       \
    _build_if_6(_cond, _action, __VA_ARGS__)
#define _build_if_8(_cond, _action, x, ...)                                    \
  if (_cond(x)) {                                                              \
    _action(x);                                                                \
  } else                                                                       \
    _build_if_7(_cond, _action, __VA_ARGS__)

#define BUILD_IF(cond, action, ...)                                            \
  _GET_NTH_ARG("ignored", ##__VA_ARGS__, _build_if_8, _build_if_7,             \
               _build_if_6, _build_if_5, _build_if_4, _build_if_3,             \
               _build_if_2, _build_if_1, _build_if_0)                          \
  (cond, action, ##__VA_ARGS__)

#define PARAMS_CONDITION(x) key == #x

#define PARAM_POINTER_ACTION(x) return &x

#define DEFINE_SET_PARAMS_UNCHECKED(...)                                       \
  Parameter *get_param_pointer(const ParameterKey &key) override {             \
    BUILD_IF(PARAMS_CONDITION, PARAM_POINTER_ACTION, __VA_ARGS__);             \
  };

/*
 * Builds a code block which declares each argument as a Parameter.
 *
 *   Parameter $1;
 *   Parameter $2;
 *
 */
#define DECLARE_PARAM(x) ::albatross::Parameter x;

#define DECLARE_PARAMS(...) CALL_MACRO_X_FOR_EACH(DECLARE_PARAM, ##__VA_ARGS__)

#define ALBATROSS_DECLARE_PARAMS(...)                                          \
  DEFINE_GET_PARAMS(__VA_ARGS__);                                              \
  DEFINE_SET_PARAMS_UNCHECKED(__VA_ARGS__);                                    \
  DECLARE_PARAMS(__VA_ARGS__);

#endif
