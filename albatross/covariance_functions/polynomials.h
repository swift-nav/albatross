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

#ifndef ALBATROSS_COVARIANCE_FUNCTIONS_POLYNOMIALS_H
#define ALBATROSS_COVARIANCE_FUNCTIONS_POLYNOMIALS_H

namespace albatross {

constexpr double default_sigma = 100.;

struct ConstantTerm {};

/*
 * The Constant covariance term represents a single scalar
 * value that is shared across all FeatureTypes.  This can
 * be thought of as a mean term.  In fact, the only reason
 * it isn't called a mean term is to avoid ambiguity with the
 * mean of a Gaussian process and because with a prior, the
 * underlying constant term would actually be a biased estimate
 * of the mean.
 */
class Constant : public CovarianceFunction<Constant> {
public:
  Constant(double sigma_constant_ = default_sigma) {
    sigma_constant = {sigma_constant_, std::make_shared<NonNegativePrior>()};
  };

  ALBATROSS_DECLARE_PARAMS(sigma_constant);

  ~Constant(){};

  std::string name() const { return "constant"; }

  template <typename X>
  std::vector<ConstantTerm>
  get_state_space_representation(std::vector<X> &) const {
    std::vector<ConstantTerm> terms = {ConstantTerm()};
    return terms;
  }

  /*
   * This will create a covariance matrix that looks like,
   *     sigma_mean^2 * ones(m, n)
   * which is saying all observations are perfectly correlated,
   * so you can move one if you move the rest the same amount.
   */
  template <typename X, typename Y>
  double call_impl_(const X &x __attribute__((unused)),
                    const Y &y __attribute__((unused))) const {
    return sigma_constant.value * sigma_constant.value;
  }
};

template <int order>
class Polynomial : public CovarianceFunction<Polynomial<order>> {
public:
  Polynomial(double sigma = default_sigma) {
    for (int i = 0; i < order + 1; i++) {
      std::string param_name = "sigma_polynomial_" + std::to_string(i);
      param_names_[i] = param_name;
      this->params_[param_name] = {sigma, std::make_shared<NonNegativePrior>()};
    }
  };

  std::string name() const { return "polynomial_" + std::to_string(order); }

  ~Polynomial(){};

  double call_impl_(const double &x, const double &y) const {
    double cov = 0.;
    for (int i = 0; i < order + 1; i++) {
      const double sigma = this->get_param_value(param_names_.at(i));
      const double p = static_cast<double>(i);
      cov += sigma * sigma * pow(x, p) * pow(y, p);
    }
    return cov;
  }

private:
  std::map<int, std::string> param_names_;
};

} // namespace albatross

#endif
