#ifndef GP_COVARIANCE_FUNCTIONS_COVARIANCE_FUNCTIONS_H
#define GP_COVARIANCE_FUNCTIONS_COVARIANCE_FUNCTIONS_H

#include "covariance_base.h"
#include "noise.h"
#include "polynomials.h"
#include "radial.h"

namespace albatross {

template <typename Covariance, typename Predictor>
struct CovarianceFunction {
  Covariance covariance;

  template <typename OtherCovariance>
  inline auto operator+(CovarianceFunction<OtherCovariance, Predictor> &other) {
    using Sum = SumOfCovariance<Covariance, OtherCovariance, Predictor>;
    auto sum = Sum(covariance, other.covariance);
    return CovarianceFunction<Sum, Predictor>{sum};
  }

  template <typename OtherCovariance>
  inline auto operator*(CovarianceFunction<OtherCovariance, Predictor> &other) {
    using Prod = ProductOfCovariance<Covariance, OtherCovariance, Predictor>;
    auto prod = Prod(covariance, other.covariance);
    return CovarianceFunction<Prod, Predictor>{prod};
  }

  inline auto operator()(const Predictor &x, const Predictor &y) const {
    return covariance(x, y);
  }

  inline auto get_name() const { return covariance.get_name(); };
  inline auto to_string() const { return covariance.to_string(); };
  inline auto get_params() const { return covariance.get_params(); };
  inline auto to_yaml() const { return covariance.to_yaml(); };
  inline auto to_file(const std::string &path) const {
    return covariance.to_file(path);
  };
  inline auto from_string(const std::string &serialized_string) {
    return covariance.from_string(serialized_string);
  };
  inline auto from_yaml(const YAML::Node &yaml_input) {
    return covariance.from_yaml(yaml_input);
  }
  inline auto set_params(const ParameterStore &params) {
    return covariance.set_params(params);
  };
  inline auto set_param(const ParameterKey &key, const ParameterValue &value) {
    return covariance.set_param(key, value);
  };
  inline auto pretty_params() const { return covariance.pretty_params(); };
  inline auto get_params_as_vector() const {
    return covariance.get_params_as_vector();
  };
  inline auto set_params_from_vector(const std::vector<ParameterValue> &x) {
    return covariance.set_params_from_vector(x);
  };
  inline auto unchecked_set_param(const std::string &name, const double value) {
    return covariance.unchecked_set_param(name, value);
  };
};

/*
 * Creates a covariance matrix given a single vector of
 * predictors.  Element i, j of the resulting covariance
 * matrix will hold
 */
template <typename Covariance, typename Predictor>
Eigen::MatrixXd symmetric_covariance(
    const CovarianceFunction<Covariance, Predictor> &f,
    const std::vector<Predictor> &xs) {
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

template <typename Covariance, typename Predictor>
Eigen::MatrixXd asymmetric_covariance(
    const CovarianceFunction<Covariance, Predictor> &f,
    const std::vector<Predictor> &xs, const std::vector<Predictor> &ys) {
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
