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
  ~EuclideanDistance(){};

  std::string get_name() const override { return "euclidean_distance"; };

  double operator()(const double &x, const double &y) const {
    return fabs(x - y);
  }

  template <typename _Scalar, int _Rows>
  double operator()(const Eigen::Matrix<_Scalar, _Rows, 1> &x,
                    const Eigen::Matrix<_Scalar, _Rows, 1> &y) const {
    return (x - y).norm();
  }
};

class RadialDistance : public DistanceMetric {
public:
  ~RadialDistance(){};

  std::string get_name() const override { return "radial_distance"; };

  double operator()(const Eigen::VectorXd &x, const Eigen::VectorXd &y) const {
    return fabs(x.norm() - y.norm());
  }
};

class AngularDistance : public DistanceMetric {
public:
  ~AngularDistance(){};

  std::string get_name() const override { return "angular_distance"; };

  template <typename _Scalar, int _Rows>
  double operator()(const Eigen::Matrix<_Scalar, _Rows, 1> &x,
                    const Eigen::Matrix<_Scalar, _Rows, 1> &y) const {
    // The acos operator doesn't behave well near |1|.  acos(1.), for example,
    // returns NaN, so here we do some special casing,
    double dot_product = x.dot(y) / (x.norm() * y.norm());
    if (dot_product > 1. - 1e-16) {
      return 0.;
    } else if (dot_product < -1. + 1e-16) {
      return M_PI;
    } else {
      return acos(dot_product);
    }
  }
};

template <typename Feature, typename DistanceMetrixType>
Eigen::MatrixXd distance_matrix(const DistanceMetrixType &distance_metric,
                                const std::vector<Feature> &xs) {
  int n = static_cast<int>(xs.size());
  Eigen::MatrixXd D(n, n);

  int i, j;
  std::size_t si, sj;
  for (i = 0; i < n; i++) {
    si = static_cast<std::size_t>(i);
    for (j = 0; j <= i; j++) {
      sj = static_cast<std::size_t>(j);
      D(i, j) = distance_metric(xs[si], xs[sj]);
      D(j, i) = D(i, j);
    }
  }
  return D;
}

} // namespace albatross

#endif
