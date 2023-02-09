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

#ifndef ALBATROSS_COVARIANCE_FUNCTIONS_RADIAL_H
#define ALBATROSS_COVARIANCE_FUNCTIONS_RADIAL_H

constexpr double default_length_scale = 100000.;
constexpr double default_radial_sigma = 10.;
constexpr double default_nu_matern = 2.5;

namespace albatross {

inline double squared_exponential_covariance(double distance,
                                             double length_scale,
                                             double sigma = 1.) {
  if (length_scale <= 0.) {
    return 0.;
  }
  ALBATROSS_ASSERT(distance >= 0.);
  return sigma * sigma * exp(-pow(distance / length_scale, 2));
}

/*
 * SquaredExponential distance
 *    covariance(d) = sigma^2 exp(-(d/length_scale)^2)
 */
template <class DistanceMetricType>
class SquaredExponential
    : public CovarianceFunction<SquaredExponential<DistanceMetricType>> {
public:
  // The SquaredExponential radial function is not positive definite
  // when the distance is an angular (or great circle) distance.
  // See:
  // Gneiting, Strictly and non-strictly positive definite functions on spheres
  static_assert(
      !std::is_base_of<AngularDistance, DistanceMetricType>::value,
      "SquaredExponential covariance with AngularDistance is not PSD.");

  ALBATROSS_DECLARE_PARAMS(squared_exponential_length_scale,
                           sigma_squared_exponential);

  SquaredExponential(double length_scale_ = default_length_scale,
                     double sigma_squared_exponential_ = default_radial_sigma)
      : distance_metric_() {
    squared_exponential_length_scale = {length_scale_, PositivePrior()};
    sigma_squared_exponential = {sigma_squared_exponential_,
                                 NonNegativePrior()};
  };

  std::string name() const {
    return "squared_exponential[" + this->distance_metric_.get_name() + "]";
  }

  std::vector<double> _ssr_impl(const std::vector<double> &xs) const {
    double min = *std::min_element(xs.begin(), xs.end());
    double max = *std::max_element(xs.begin(), xs.end());

    double range = max - min;
    // using 1/10th of the length scale should result in grids with
    // one percent decorrelation between them. exp(- 0.1**2)
    double n = ceil(10 * range / squared_exponential_length_scale.value);
    n = std::max(n, 3.);
    return linspace(min, max, safe_cast_to_size_t(n));
  }

  // This operator is only defined when the distance metric is also defined.
  template <typename X,
            typename std::enable_if<
                has_call_operator<DistanceMetricType, X &, X &>::value,
                int>::type = 0>
  double _call_impl(const X &x, const X &y) const {
    double distance = this->distance_metric_(x, y);
    return squared_exponential_covariance(
        distance, squared_exponential_length_scale.value,
        sigma_squared_exponential.value);
  }

  DistanceMetricType distance_metric_;
};

inline double exponential_covariance(double distance, double length_scale,
                                     double sigma = 1.) {
  if (length_scale <= 0.) {
    return 0.;
  }
  ALBATROSS_ASSERT(distance >= 0.);
  return sigma * sigma * exp(-fabs(distance / length_scale));
}

/*
 * Exponential distance
 *    covariance(d) = sigma^2 exp(-|d|/length_scale)
 */
template <class DistanceMetricType>
class Exponential : public CovarianceFunction<Exponential<DistanceMetricType>> {
public:
  ALBATROSS_DECLARE_PARAMS(exponential_length_scale, sigma_exponential);

  Exponential(double length_scale_ = default_length_scale,
              double sigma_exponential_ = default_radial_sigma)
      : distance_metric_() {
    exponential_length_scale = {length_scale_, PositivePrior()};
    sigma_exponential = {sigma_exponential_, NonNegativePrior()};
  };

  std::string name() const {
    return "exponential[" + this->distance_metric_.get_name() + "]";
  }

  ~Exponential(){};

  std::vector<double> _ssr_impl(const std::vector<double> &xs) const {
    double min = *std::min_element(xs.begin(), xs.end());
    double max = *std::max_element(xs.begin(), xs.end());

    double range = max - min;
    // using 1/20th of the length scale should result in grids with
    // five percent decorrelation between them. exp(-0.05)
    double n = ceil(20 * range / exponential_length_scale.value);
    n = std::max(n, 3.);
    return linspace(min, max, safe_cast_to_size_t(n));
  }

  // This operator is only defined when the distance metric is also defined.
  template <typename X,
            typename std::enable_if<
                has_call_operator<DistanceMetricType, X &, X &>::value,
                int>::type = 0>
  double _call_impl(const X &x, const X &y) const {
    double distance = this->distance_metric_(x, y);
    return exponential_covariance(distance, exponential_length_scale.value,
                                  sigma_exponential.value);
  }

  DistanceMetricType distance_metric_;
};

inline double matern_32_covariance(double distance, double length_scale,
                                   double sigma = 1.) {
  if (length_scale <= 0.) {
    return 0.;
  }
  assert(distance >= 0.);
  const double sqrt_3_d = std::sqrt(3.) * distance / length_scale;
  return sigma * sigma * (1 + sqrt_3_d) * exp(-sqrt_3_d);
}

template <class DistanceMetricType>
class Matern32 : public CovarianceFunction<Matern32<DistanceMetricType>> {
public:
  // The Matern nu = 3/2 radial function is not positive definite
  // when the distance is an angular (or great circle) distance.
  static_assert(!std::is_base_of<AngularDistance, DistanceMetricType>::value,
                "Matern32 covariance with AngularDistance is not PSD.");

  ALBATROSS_DECLARE_PARAMS(matern_32_length_scale, sigma_matern_32);

  Matern32(double length_scale_ = default_length_scale,
           double sigma_matern_32_ = default_radial_sigma)
      : distance_metric_() {
    matern_32_length_scale = {length_scale_, PositivePrior()};
    sigma_matern_32 = {sigma_matern_32_, NonNegativePrior()};
  };

  std::string name() const {
    return "matern_32[" + this->distance_metric_.get_name() + "]";
  }

  template <typename X,
            typename std::enable_if<
                has_call_operator<DistanceMetricType, X &, X &>::value,
                int>::type = 0>
  double _call_impl(const X &x, const X &y) const {
    double distance = this->distance_metric_(x, y);
    return matern_32_covariance(distance, matern_32_length_scale.value,
                                sigma_matern_32.value);
  }

  DistanceMetricType distance_metric_;
};

inline double matern_52_covariance(double distance, double length_scale,
                                   double sigma = 1.) {
  if (length_scale <= 0.) {
    return 0.;
  }
  assert(distance >= 0.);
  const double sqrt_5_d = std::sqrt(5.) * distance / length_scale;
  return sigma * sigma * (1 + sqrt_5_d + sqrt_5_d * sqrt_5_d / 3.) *
         exp(-sqrt_5_d);
}

template <class DistanceMetricType>
class Matern52 : public CovarianceFunction<Matern52<DistanceMetricType>> {
public:
  // The Matern nu = 5/2 radial function is not positive definite
  // when the distance is an angular (or great circle) distance.
  static_assert(!std::is_base_of<AngularDistance, DistanceMetricType>::value,
                "Matern52 covariance with AngularDistance is not PSD.");

  ALBATROSS_DECLARE_PARAMS(matern_52_length_scale, sigma_matern_52);

  Matern52(double length_scale_ = default_length_scale,
           double sigma_matern_52_ = default_radial_sigma)
      : distance_metric_() {
    matern_52_length_scale = {length_scale_, PositivePrior()};
    sigma_matern_52 = {sigma_matern_52_, NonNegativePrior()};
  };

  std::string name() const {
    return "matern_52[" + this->distance_metric_.get_name() + "]";
  }

  template <typename X,
            typename std::enable_if<
                has_call_operator<DistanceMetricType, X &, X &>::value,
                int>::type = 0>
  double _call_impl(const X &x, const X &y) const {
    double distance = this->distance_metric_(x, y);
    return matern_52_covariance(distance, matern_52_length_scale.value,
                                sigma_matern_52.value);
  }

  DistanceMetricType distance_metric_;
};

inline double matern_covariance(double distance, double length_scale, double nu,
                                double sigma = 1.) {
  if (length_scale <= 0.) {
    return 0;
  }
  if (distance == 0.) {
    return sigma * sigma;
  }
  assert(nu >= 0);
  const double m = 2 * std::sqrt(nu) * distance / length_scale;
  return sigma * sigma * std::pow(2, 1 - nu) / std::tgamma(nu) *
         std::pow(m, nu) * boost::math::cyl_bessel_k<double>(nu, m);
}

template <class DistanceMetricType>
class Matern : public CovarianceFunction<Matern<DistanceMetricType>> {
 public:
  // The Matern nu = 5/2 radial function is not positive definite
  // when the distance is an angular (or great circle) distance.
  static_assert(!std::is_base_of<AngularDistance, DistanceMetricType>::value,
                "Matern covariance with AngularDistance is not PSD.");

  ALBATROSS_DECLARE_PARAMS(matern_length_scale, sigma_matern, nu_matern);

  Matern(double length_scale_ = default_length_scale,
         double sigma_matern_ = default_radial_sigma,
         double nu_matern_ = default_nu_matern)
      : distance_metric_() {
    matern_length_scale = {length_scale_, PositivePrior()};
    sigma_matern = {sigma_matern_, NonNegativePrior()};
    nu_matern = {nu_matern_, PositivePrior()};
  };

  std::string name() const {
    return "matern[" + this->distance_metric_.get_name() + "]";
  }

  template <typename X,
            typename std::enable_if<
                has_call_operator<DistanceMetricType, X &, X &>::value,
                int>::type = 0>
  double _call_impl(const X &x, const X &y) const {
    double distance = this->distance_metric_(x, y);
    return matern_covariance(distance, matern_length_scale.value,
                             sigma_matern.value);
  }

  DistanceMetricType distance_metric_;
};
} // namespace albatross
#endif
