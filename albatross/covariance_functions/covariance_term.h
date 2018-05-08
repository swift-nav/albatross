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

#ifndef ALBATROSS_COVARIANCE_FUNCTIONS_COVARIANCE_TERM_H
#define ALBATROSS_COVARIANCE_FUNCTIONS_COVARIANCE_TERM_H

#include "../core/traits.h"
#include "core/parameter_handling_mixin.h"
#include "map_utils.h"
#include <iostream>

namespace albatross {

/*
 * An abstract class which holds anything all Covariance
 * terms should have in common.
 *
 * In addition to these abstract methods one or many
 * methods with signature,
 *     operator ()(const X &x, const Y &y)
 * should be defined.
 */
class CovarianceTerm : public ParameterHandlingMixin {
public:
  CovarianceTerm() : ParameterHandlingMixin(){};
  virtual ~CovarianceTerm(){};

  virtual std::string get_name() const = 0;

  std::string pretty_string() const {
    std::ostringstream ss;
    ss << get_name() << std::endl;
    ss << ParameterHandlingMixin::pretty_string();
    return ss.str();
  }
};

/*
 * As we start composing CovarianceTerms we want to keep track of
 * their parameters and names in a friendly way.  This class deals with
 * any of those operations that are shared for all composition operations.
 * The details of the operation (sum, product, etc ...) are left to implementing
 * classes.
 */
template <class LHS, class RHS>
class CombinationOfCovarianceTerms : public CovarianceTerm {
public:
  CombinationOfCovarianceTerms() : lhs_(), rhs_(){};
  CombinationOfCovarianceTerms(LHS &lhs, RHS &rhs) : lhs_(lhs), rhs_(rhs){};
  virtual ~CombinationOfCovarianceTerms(){};

  virtual std::string get_operation_symbol() const = 0;

  std::string get_name() const override {
    std::ostringstream oss;
    oss << "(" << lhs_.get_name() << get_operation_symbol() << rhs_.get_name()
        << ")";
    return oss.str();
  }

  ParameterStore get_params() const override {
    return map_join(lhs_.get_params(), rhs_.get_params());
  }

  void unchecked_set_param(const std::string &name,
                           const double value) override {
    if (map_contains(lhs_.get_params(), name)) {
      lhs_.set_param(name, value);
    } else {
      rhs_.set_param(name, value);
    }
  }

protected:
  LHS lhs_;
  RHS rhs_;
};

template <class LHS, class RHS>
class SumOfCovarianceTerms : public CombinationOfCovarianceTerms<LHS, RHS> {
public:
  SumOfCovarianceTerms() : CombinationOfCovarianceTerms<LHS, RHS>(){};
  SumOfCovarianceTerms(LHS &lhs, RHS &rhs)
      : CombinationOfCovarianceTerms<LHS, RHS>(lhs, rhs){};

  std::string get_operation_symbol() const { return "+"; }

  /*
   * If both LHS and RHS have a valid call method for the types X and Y
   * this will return the sum of the two.
   */
  template <typename X, typename Y,
            typename std::enable_if<(has_call_operator<LHS, X &, Y &>::value &&
                                     has_call_operator<RHS, X &, Y &>::value),
                                    int>::type = 0>
  double operator()(X &x, Y &y) const {
    return this->lhs_(x, y) + this->rhs_(x, y);
  }

  /*
   * If only LHS has a valid call method we ignore R.
   */
  template <typename X, typename Y,
            typename std::enable_if<(has_call_operator<LHS, X &, Y &>::value &&
                                     !has_call_operator<RHS, X &, Y &>::value),
                                    int>::type = 0>
  double operator()(X &x, Y &y) const {
    return this->lhs_(x, y);
  }

  /*
   * If only RHS has a valid call method we ignore L.
   */
  template <typename X, typename Y,
            typename std::enable_if<(!has_call_operator<LHS, X &, Y &>::value &&
                                     has_call_operator<RHS, X &, Y &>::value),
                                    int>::type = 0>
  double operator()(X &x, Y &y) const {
    return this->rhs_(x, y);
  }
};

template <class LHS, class RHS>
class ProductOfCovarianceTerms : public CombinationOfCovarianceTerms<LHS, RHS> {
public:
  ProductOfCovarianceTerms() : CombinationOfCovarianceTerms<LHS, RHS>(){};
  ProductOfCovarianceTerms(LHS &lhs, RHS &rhs)
      : CombinationOfCovarianceTerms<LHS, RHS>(lhs, rhs){};

  std::string get_operation_symbol() const { return "*"; }

  /*
   * If both LHS and RHS have a valid call method for the types X and Y
   * this will return the product of the two.
   */
  template <typename X, typename Y,
            typename std::enable_if<(has_call_operator<LHS, X &, Y &>::value &&
                                     has_call_operator<RHS, X &, Y &>::value),
                                    int>::type = 0>
  double operator()(X &x, Y &y) const {
    return this->lhs_(x, y) * this->rhs_(x, y);
  }

  /*
   * If only LHS has a valid call method we ignore R.
   */
  template <typename X, typename Y,
            typename std::enable_if<(has_call_operator<LHS, X &, Y &>::value &&
                                     !has_call_operator<RHS, X &, Y &>::value),
                                    int>::type = 0>
  double operator()(X &x, Y &y) const {
    return this->lhs_(x, y);
  }

  /*
   * If only RHS has a valid call method we ignore L.
   */
  template <typename X, typename Y,
            typename std::enable_if<(!has_call_operator<LHS, X &, Y &>::value &&
                                     has_call_operator<RHS, X &, Y &>::value),
                                    int>::type = 0>
  double operator()(X &x, Y &y) const {
    return this->rhs_(x, y);
  }
};

} // albatross

#endif
