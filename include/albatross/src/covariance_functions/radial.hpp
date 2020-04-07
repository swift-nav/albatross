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

namespace albatross {

inline double squared_exponential_covariance(double distance,
                                             double length_scale,
                                             double sigma = 1.) {
  if (length_scale <= 0.) {
    return 0.;
  }
  assert(distance >= 0.);
  return sigma * sigma * exp(-pow(distance / length_scale, 2));
}

inline double squared_exponential_covariance_derivative(double distance,
                                                        double length_scale,
                                                        double sigma = 1.) {
  if (length_scale <= 0.) {
    return 0.;
  }
  return -2 * distance / (length_scale * length_scale) *
         squared_exponential_covariance(distance, length_scale, sigma);
}

inline double squared_exponential_covariance_second_derivative(
    double distance, double length_scale, double sigma = 1.) {
  if (length_scale <= 0.) {
    return 0.;
  }
  const auto ratio = distance / length_scale;
  return (4. * ratio * ratio - 2.) / (length_scale * length_scale) *
         squared_exponential_covariance(distance, length_scale, sigma);
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

  // This operator is only defined when the distance metric is also defined.
  template <typename X,
            typename std::enable_if<
                has_call_operator<DistanceMetricType, X &, X &>::value,
                int>::type = 0>
  double _call_impl(const Derivative<X> &x, const X &y) const {
    double distance = this->distance_metric_(x.value, y);
    double distance_derivative = this->distance_metric_.derivative(x.value, y);
    return distance_derivative * squared_exponential_covariance_derivative(
                                     distance,
                                     squared_exponential_length_scale.value,
                                     sigma_squared_exponential.value);
  }

  template <typename X,
            typename std::enable_if<
                has_call_operator<DistanceMetricType, X &, X &>::value,
                int>::type = 0>
  double _call_impl(const Derivative<X> &x, const Derivative<X> &y) const {
    const double distance = this->distance_metric_(x.value, y.value);
    const double d_x = this->distance_metric_.derivative(x.value, y.value);
    const double d_y = this->distance_metric_.derivative(y.value, x.value);
    const double d_xy =
        this->distance_metric_.second_derivative(x.value, y.value);

    const double f_d = squared_exponential_covariance_derivative(
        distance, squared_exponential_length_scale.value,
        sigma_squared_exponential.value);

    const double f_dd = squared_exponential_covariance_second_derivative(
        distance, squared_exponential_length_scale.value,
        sigma_squared_exponential.value);

    std::cout << x.value << "  " << y.value << "  " << d_xy << ", " << f_d
              << ", " << d_x << ", " << d_y << ", " << f_dd << std::endl;
    return d_xy * f_d + d_x * d_y * f_dd;
  }

  // This operator is only defined when the distance metric is also defined.
  template <typename X,
            typename std::enable_if<
                has_call_operator<DistanceMetricType, X &, X &>::value,
                int>::type = 0>
  double _call_impl(const SecondDerivative<X> &x, const X &y) const {
    double d = this->distance_metric_(x.value, y);
    double d_1 = this->distance_metric_.derivative(x.value, y);
    double d_2 = this->distance_metric_.second_derivative(x.value, y);
    double f_1 = squared_exponential_covariance_derivative(
        d, squared_exponential_length_scale.value,
        sigma_squared_exponential.value);
    double f_2 = squared_exponential_covariance_second_derivative(
        d, squared_exponential_length_scale.value,
        sigma_squared_exponential.value);
    return d_2 * f_1 + d_1 * d_1 * f_2;
  }

  // This operator is only defined when the distance metric is also defined.
  template <typename X,
            typename std::enable_if<
                has_call_operator<DistanceMetricType, X &, X &>::value,
                int>::type = 0>
  double _call_impl(const SecondDerivative<X> &x,
                    const SecondDerivative<X> &y) const {
    return NAN;
  }

  DistanceMetricType distance_metric_;
};

inline double exponential_covariance(double distance, double length_scale,
                                     double sigma = 1.) {
  if (length_scale <= 0.) {
    return 0.;
  }
  assert(distance >= 0.);
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

} // namespace albatross
#endif
