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

/*
 * SquaredExponential distance
 *  - c(d) = -exp((d/length_scale)^2)
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

  template <
      typename FeatureType,
      typename std::enable_if<has_gridded_features_with_return_type<
                                  DistanceMetricType, std::vector<FeatureType>,
                                  std::vector<FeatureType>, double>::value,
                              int>::type = 0>
  std::vector<FeatureType>
  _ssr_features(const std::vector<FeatureType> &features) const {
    //  std::vector<double> _ssr_features(const std::vector<double> &features)
    //  const {
    // This is the spacing for which the correlation between points drops
    // by one percent
    double spacing = 0.1 * squared_exponential_length_scale.value;
    return distance_metric_.gridded_features(features, spacing);
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
 *  - c(d) = -exp(|d|/length_scale)
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
