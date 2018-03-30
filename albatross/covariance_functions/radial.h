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

#ifndef GP_COVARIANCE_FUNCTIONS_RADIAL_H
#define GP_COVARIANCE_FUNCTIONS_RADIAL_H

#include <sstream>
#include "covariance_base.h"
#include "distance_metrics.h"

namespace albatross {

/*
 * RadialCovariance functions require a distance metric which
 * computes the distance between any two predictors.  That distance
 * is then used by the implementing function to determine how
 * correlated the two elements are as a function of their distance.
 */
template <class DistanceMetricImpl>
class RadialCovariance : public CovarianceBase {
 public:
  RadialCovariance() : distance_metric_(){};

  ~RadialCovariance(){};

  ParameterStore get_params() const override {
    return map_join(this->params_, distance_metric_.get_params());
  }

  void unchecked_set_param(const std::string &name,
                           const double value) override {
    if (map_contains(this->params_, name)) {
      this->params_[name] = value;
    } else {
      distance_metric_.set_param(name, value);
    }
  }

 protected:
  DistanceMetricImpl distance_metric_;
};

/*
 * SquaredExponential distance
 *  - c(d) = -exp((d/length_scale)^2)
 */
template <class DistanceMetricImpl>
class SquaredExponential
    : public RadialCovariance<DistanceMetricImpl> {
 public:
  SquaredExponential(double length_scale = 100000.,
                     double sigma_squared_exponential = 10.) {
    this->params_["length_scale"] = length_scale;
    this->params_["sigma_squared_exponential"] = sigma_squared_exponential;
  };

  ~SquaredExponential(){};

  std::string get_name() const {
    std::ostringstream oss;
    oss << "squared_exponential[" << this->distance_metric_.get_name() << "]";
    return oss.str();
  }

  template <typename Predictor>
  double operator()(const Predictor &x, const Predictor &y) const {
    double distance = this->distance_metric_(x, y);
    double length_scale = this->params_.at("length_scale");
    distance /= length_scale;
    double sigma = this->params_.at("sigma_squared_exponential");
    return sigma * sigma * exp(-distance * distance);
  }
};
}
#endif
