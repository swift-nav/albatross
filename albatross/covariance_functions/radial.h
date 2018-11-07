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

#include <sstream>

#include "covariance_function.h"
#include "distance_metrics.h"

constexpr double default_length_scale = 100000.;
constexpr double default_radial_sigma = 10.;

namespace albatross {

inline double squared_exponential_covariance(double distance,
                                             double length_scale,
                                             double sigma = 1.) {
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

  SquaredExponential(double length_scale = default_length_scale,
                     double sigma_squared_exponential = default_radial_sigma)
      : distance_metric_(), name_() {
    this->params_["squared_exponential_length_scale"] = {
        length_scale, std::make_shared<PositivePrior>()};
    this->params_["sigma_squared_exponential"] = {
        sigma_squared_exponential, std::make_shared<NonNegativePrior>()};
    std::ostringstream oss;
    oss << "squared_exponential[" << this->distance_metric_.get_name() << "]";
    name_ = oss.str();
  };

  // This operator is only defined when the distance metric is also defined.
  template <typename X,
            typename std::enable_if<
                has_call_operator<DistanceMetricType, X &, X &>::value,
                int>::type = 0>
  double call_impl_(const X &x, const X &y) const {
    double distance = this->distance_metric_(x, y);
    double length_scale =
        this->get_param_value("squared_exponential_length_scale");
    double sigma = this->get_param_value("sigma_squared_exponential");
    return squared_exponential_covariance(distance, length_scale, sigma);
  }

  DistanceMetricType distance_metric_;
  std::string name_;
};

inline double exponential_covariance(double distance, double length_scale,
                                     double sigma = 1.) {
  return sigma * sigma * exp(-fabs(distance / length_scale));
}

/*
 * Exponential distance
 *  - c(d) = -exp(|d|/length_scale)
 */
template <class DistanceMetricType>
class Exponential : public CovarianceFunction<Exponential<DistanceMetricType>> {
public:
  Exponential(double length_scale = default_length_scale,
              double sigma_exponential = default_radial_sigma)
      : distance_metric_(), name_() {
    this->params_["exponential_length_scale"] = {
        length_scale, std::make_shared<PositivePrior>()};
    this->params_["sigma_exponential"] = {sigma_exponential,
                                          std::make_shared<NonNegativePrior>()};
    std::ostringstream oss;
    oss << "exponential[" << this->distance_metric_.get_name() << "]";
    name_ = oss.str();
  };

  ~Exponential(){};

  // This operator is only defined when the distance metric is also defined.
  template <typename X,
            typename std::enable_if<
                has_call_operator<DistanceMetricType, X &, X &>::value,
                int>::type = 0>
  double call_impl_(const X &x, const X &y) const {
    double distance = this->distance_metric_(x, y);
    double length_scale = this->get_param_value("exponential_length_scale");
    double sigma = this->get_param_value("sigma_exponential");
    return exponential_covariance(distance, length_scale, sigma);
  }

  DistanceMetricType distance_metric_;
  std::string name_;
};

} // namespace albatross
#endif
