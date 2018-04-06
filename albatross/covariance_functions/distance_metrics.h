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

#ifndef ALBATROSS_COVARIANCE_FUNCTIONS_DISTANCE_METRICS_H
#define ALBATROSS_COVARIANCE_FUNCTIONS_DISTANCE_METRICS_H

#include "core/parameter_handling_mixin.h"
#include <Eigen/Core>

namespace albatross {

class DistanceMetric : public ParameterHandlingMixin {
 public:
  DistanceMetric(){};
  virtual ~DistanceMetric(){};

 virtual std::string get_name() const = 0;
 protected:
};

class EuclideanDistance : public DistanceMetric {
 public:
  EuclideanDistance(){};
  ~EuclideanDistance(){};

  std::string get_name() const override { return "euclidean"; };

  double operator()(const Eigen::VectorXd &x, const Eigen::VectorXd &y) const {
    return (x - y).norm();
  }
};

class RadialDistance : public DistanceMetric {
 public:
  RadialDistance(){};
  ~RadialDistance(){};

  std::string get_name() const override { return "radial"; };

  double operator()(const Eigen::VectorXd &x, const Eigen::VectorXd &y) const {
    return fabs(x.norm() - y.norm());
  }
};
}

#endif
