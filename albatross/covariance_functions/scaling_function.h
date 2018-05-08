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

#ifndef ALBATROSS_COVARIANCE_FUNCTIONS_SCALING_FUNCTION_H
#define ALBATROSS_COVARIANCE_FUNCTIONS_SCALING_FUNCTION_H

#include "covariance_term.h"
#include <sstream>
#include <utility>

namespace albatross {

class ScalingFunction : public ParameterHandlingMixin {
public:
  virtual std::string get_name() const = 0;

  // A scaling function should also implement operators
  // for whichever types it is intended to scale using
  // the signature:
  //   double operator(const X &x) const;
};

/*
 * A scaling term is not actually a covariance function
 * in the rigorous sense.  It doesn't describe the uncertainty
 * between variables, but instead operates deterministically
 * on other uncertain variables.  For instance, you may have
 * some random variable,
 *     y ~ N(0, S)  with  S_ij = cov(y_i, y_j)
 * And you may then make observations of that random variable
 * but through a known transformation,
 *     z = f(y) y
 * where f is a determinstic function of y that returns a scalar.
 * You might then ask what the covariance between two elements in
 * z is which is woudl be given by,
 *     cov(z_i, z_j) = f(y_i) * cov(y_i, y_j) * f(y_j)
 * but you might also be interested in the covariance between
 * some y_i and an observation z_j,
 *     cov(y_i, z_j) = cov(y_i, y_j) * f(y_j)
 * Here we see that for a typical covariance term, the covariance
 * is only defined for two pairs of the same type, in this case
 *     operator()(Y &y, Y &y)
 * but by multiplying by a ScalingTerm we end up with definitions
 * for,
 *     operator()(Y &y, Z &z)
 * which provides us with a way of computing the covariance between
 * some hidden representation of a variable (y) and the actual
 * observations (z) using a single determinstic mapping (f).
 *
 * This might be better explained by example which can be found
 * in the tests (test_scaling_function).
 */
template <typename ScalingFunction> class ScalingTerm : public CovarianceTerm {
public:
  ScalingTerm() : CovarianceTerm(){};
  virtual ~ScalingTerm(){};

  /*
   * The following methods forward any requests dealing with
   * the ParameterHandlingMixin to the ScalingFunction.
   */
  std::string get_name() const override { return scaling_function_.get_name(); }

  std::string pretty_string() const {
    return scaling_function_.pretty_string();
  }

  void set_params(const ParameterStore &params) {
    scaling_function_.set_params(params);
  }

  virtual ParameterStore get_params() const {
    return scaling_function_.get_params();
  }

  template <class Archive> void save(Archive &archive) const {
    archive(cereal::make_nvp("base_class",
                             cereal::base_class<CovarianceTerm>(this)));
    archive(cereal::make_nvp("scaling_function", scaling_function_));
  }

  template <class Archive> void load(Archive &archive) {
    archive(cereal::make_nvp("base_class",
                             cereal::base_class<CovarianceTerm>(this)));
    archive(cereal::make_nvp("scaling_function", scaling_function_));
  }

  void unchecked_set_param(const std::string &name,
                           const double value) override {
    scaling_function_.set_param(name, value);
  }

  /*
   * If both Scaling and Covariance have a valid call method for the types X
   * and Y this will return the product of the two.
   */
  template <
      typename X, typename Y,
      typename std::enable_if<(has_call_operator<ScalingFunction, X &>::value &&
                               has_call_operator<ScalingFunction, Y &>::value),
                              int>::type = 0>
  double operator()(X &x, Y &y) const {
    return this->scaling_function_(x) * this->scaling_function_(y);
  }

  /*
   * If only one of the types has a scaling function we ignore the other.
   */
  template <typename X, typename Y,
            typename std::enable_if<
                (!has_call_operator<ScalingFunction, X &>::value &&
                 has_call_operator<ScalingFunction, Y &>::value),
                int>::type = 0>
  double operator()(X &x, Y &y) const {
    return this->scaling_function_(y);
  }

  template <
      typename X, typename Y,
      typename std::enable_if<(has_call_operator<ScalingFunction, X &>::value &&
                               !has_call_operator<ScalingFunction, Y &>::value),
                              int>::type = 0>
  double operator()(X &x, Y &y) const {
    return this->scaling_function_(x);
  }

private:
  ScalingFunction scaling_function_;
};
}
#endif
