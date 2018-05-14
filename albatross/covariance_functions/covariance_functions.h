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

#ifndef ALBATROSS_COVARIANCE_FUNCTIONS_COVARIANCE_FUNCTIONS_H
#define ALBATROSS_COVARIANCE_FUNCTIONS_COVARIANCE_FUNCTIONS_H

#include "covariance_term.h"
#include "noise.h"
#include "polynomials.h"
#include "radial.h"
#include "scaling_function.h"

namespace albatross {

template <typename Term> struct CovarianceFunction {
  Term term;

  CovarianceFunction() : term(){};
  CovarianceFunction(const Term &term_) : term(term_){};

  template <typename Other>
  inline auto operator+(CovarianceFunction<Other> &other) {
    using Sum = SumOfCovarianceTerms<Term, Other>;
    auto sum = Sum(term, other.term);
    return CovarianceFunction<Sum>{sum};
  }

  template <typename Other>
  inline auto operator*(CovarianceFunction<Other> &other) {
    using Prod = ProductOfCovarianceTerms<Term, Other>;
    auto prod = Prod(term, other.term);
    return CovarianceFunction<Prod>{prod};
  }

  /*
   * We only allow the covariance function for argument types X and Y
   * to be defined if the corresponding term is defined for these
   * types.  This allows us to distinguish between covariance functions which
   * are 0. and ones that are just not defined (read: possibly bugs).
   */
  template <typename X, typename Y,
            typename std::enable_if<(has_call_operator<Term, X &, Y &>::value),
                                    int>::type = 0>
  inline auto operator()(const X &x, const Y &y) const {
    return term(x, y);
  }

  /*
   * If neither have a valid call method we fail.
   */
  template <typename X, typename Y,
            typename std::enable_if<(!has_call_operator<Term, X &, Y &>::value),
                                    int>::type = 0>
  double operator()(X &x, Y &y) const = delete; // see below for help debugging.
                                                /*
                                                 * If you encounter a deleted function error here ^ it implies that you've
                                                 * attempted to call a covariance function with arguments X, Y that are
                                                 * undefined for the corresponding CovarianceTerm(s).  The subsequent compiler
                                                 * errors should give you an indication of which types were attempted.
                                                 */

  inline auto get_name() const { return term.get_name(); };
  inline auto get_params() const { return term.get_params(); };
  inline auto set_params(const ParameterStore &params) {
    return term.set_params(params);
  };
  inline auto set_param(const ParameterKey &key, const ParameterValue &value) {
    return term.set_param(key, value);
  };
  inline auto pretty_string() const { return term.pretty_string(); };
  inline auto get_params_as_vector() const {
    return term.get_params_as_vector();
  };
  inline auto set_params_from_vector(const std::vector<ParameterValue> &x) {
    return term.set_params_from_vector(x);
  };
  inline auto unchecked_set_param(const std::string &name, const double value) {
    return term.unchecked_set_param(name, value);
  };
};

/*
 * Creates a covariance matrix given a single vector of
 * predictors.  Element i, j of the resulting covariance
 * matrix will hold
 */
template <typename Covariance, typename Feature>
Eigen::MatrixXd symmetric_covariance(const CovarianceFunction<Covariance> &f,
                                     const std::vector<Feature> &xs) {
  int n = static_cast<int>(xs.size());
  Eigen::MatrixXd C(n, n);

  int i, j;
  std::size_t si, sj;
  for (i = 0; i < n; i++) {
    si = static_cast<std::size_t>(i);
    for (j = 0; j <= i; j++) {
      sj = static_cast<std::size_t>(j);
      C(i, j) = f(xs[si], xs[sj]);
      C(j, i) = C(i, j);
    }
  }
  return C;
}

/*
 * Computes the covariance matrix between some predictors (x) and
 * a separate distinct set (y).  x and y can be of the same type,
 * which is common when making predictions at new locations, or x may
 * be of some arbitrary type, which is common when inspecting covariance
 * functions.
 */
template <typename Covariance, typename OtherFeature, typename Feature>
Eigen::MatrixXd asymmetric_covariance(const CovarianceFunction<Covariance> &f,
                                      const std::vector<OtherFeature> &xs,
                                      const std::vector<Feature> &ys) {
  int m = static_cast<int>(xs.size());
  int n = static_cast<int>(ys.size());
  Eigen::MatrixXd C(m, n);

  int i, j;
  std::size_t si, sj;
  for (i = 0; i < m; i++) {
    si = static_cast<std::size_t>(i);
    for (j = 0; j < n; j++) {
      sj = static_cast<std::size_t>(j);
      C(i, j) = f(xs[si], ys[sj]);
    }
  }
  return C;
}
}

#endif
