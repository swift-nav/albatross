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
    sigma_constant = {sigma_constant_, NonNegativePrior()};
  }

  ALBATROSS_DECLARE_PARAMS(sigma_constant);

  ~Constant() {}

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
  double _call_impl(const X &x __attribute__((unused)),
                    const Y &y __attribute__((unused))) const {
    return sigma_constant.value * sigma_constant.value;
  }
};

template <int order>
class Polynomial : public CovarianceFunction<Polynomial<order>> {
public:
  Polynomial(double sigma = default_sigma) {
    for (int i = 0; i < order + 1; i++) {
      const std::string param_name =
          "sigma_polynomial_" + std::to_string(i);
      param_names_[i] = param_name;
      this->params_[param_name] = {sigma, NonNegativePrior()};
      sigma_squared_[i] = sigma * sigma;
    }
  }

  std::string name() const { return "polynomial_" + std::to_string(order); }

  ~Polynomial() {}

  // The kernel is called O(N^2) times per matrix build. We avoid the per-call
  // map lookup of the underlying parameter store by caching sigma_i^2 here,
  // and we replace pow(x, i) with a running product to avoid std::pow.
  void set_param(const ParameterKey &name, const Parameter &param) override {
    albatross::set_param(name, param, &this->params_);
    for (int i = 0; i < order + 1; i++) {
      if (param_names_[i] == name) {
        sigma_squared_[i] = param.value * param.value;
        break;
      }
    }
  }

  double _call_impl(const double &x, const double &y) const {
    double cov = 0.;
    double xp = 1.;
    double yp = 1.;
    for (int i = 0; i < order + 1; i++) {
      cov += sigma_squared_[i] * xp * yp;
      xp *= x;
      yp *= y;
    }
    return cov;
  }

private:
  std::map<int, std::string> param_names_;
  std::array<double, order + 1> sigma_squared_;
};

class LinearMean : public MeanFunction<LinearMean> {
public:
  ALBATROSS_DECLARE_PARAMS(slope, offset);

  std::string get_name() const { return "linear"; }

  LinearMean() {
    slope = {0., GaussianPrior(0., 1000.)};
    offset = {0., GaussianPrior(0., 1000.)};
  }

  double _call_impl(const double &x) const {
    return slope.value * x + offset.value;
  }
};

} // namespace albatross

#endif
