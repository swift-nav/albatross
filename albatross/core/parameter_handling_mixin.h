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

#include <assert.h>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#include "cereal/cereal.hpp"
#include "keys.h"
#include "map_utils.h"
#include <cereal/types/map.hpp>

namespace albatross {

using ParameterKey = std::string;
using ParameterValue = double;
// If you change the way these are stored, be sure there's
// a corresponding cereal type included or you'll get some
// really impressive compilation errors.
using ParameterStore = std::map<ParameterKey, ParameterValue>;

/*
 * Prints out a set of parameters in a way that is both
 * readable and can be easily copy/pasted into code.
 */
inline std::string pretty_params(const ParameterStore &params) {
  std::ostringstream ss;
  ss << "{" << std::endl;
  for (const auto &pair : params) {
    ss << "    {\"" << pair.first << "\", " << pair.second << "}," << std::endl;
  }
  ss << "};" << std::endl;
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

  /*
   * Provides a safe interface to the parameter values
   */
  void set_params(const ParameterStore &params) {
    const ParameterStore current_params = get_params();
    for (const auto &pair : params) {
      assert(map_contains(current_params, pair.first));
      unchecked_set_param(pair.first, pair.second);
    }
  }

  void set_param(const ParameterKey &key, const ParameterValue &value) {
    assert(map_contains(get_params(), key));
    unchecked_set_param(key, value);
  }

  /*
   * These methods which collapse a set of vectors to a vector, and
   * set them from a vector facilitate things like tuning in which
   * some function doesn't actually care what any of the parameters
   * correspond to.
   */
  std::vector<ParameterValue> get_params_as_vector() const {
    std::vector<ParameterValue> x;
    const ParameterStore params = get_params();
    for (const auto &pair : params) {
      x.push_back(pair.second);
    }
    return x;
  }

  void set_params_from_vector(const std::vector<ParameterValue> &x) {
    const std::size_t n = x.size();
    const ParameterStore params = get_params();
    assert(n == static_cast<std::size_t>(params.size()));

    const std::vector<ParameterKey> param_names = map_keys(params);
    for (std::size_t i = 0; i < n; i++) {
      unchecked_set_param(param_names[i], x[i]);
    }
  }

  /*
   * For serialization through cereal.
   */
  template <class Archive> void save(Archive &archive) const {
    archive(cereal::make_nvp("parameter_store", params_));
  };

  template <class Archive> void load(Archive &archive) {
    archive(cereal::make_nvp("parameter_store", params_));
  };

  /*
   * For debugging.
   */
  std::string pretty_string() const { return pretty_params(get_params()); }

  /*
   * The following methods are ones that may want to be overriden for
   * clasess that contain nested params (for example).
   */

  virtual ParameterStore get_params() const { return params_; }

  virtual void unchecked_set_param(const std::string &name,
                                   const double value) {
    params_[name] = value;
  }

protected:
  ParameterStore params_;
};
} // namespace albatross

#endif
