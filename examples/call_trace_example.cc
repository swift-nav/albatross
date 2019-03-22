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

#include "CovarianceFunctions"

namespace albatross {

class LinearScalar : public ScalingFunction {
public:
  std::string get_name() const override { return "linear_scalar"; }

  double _call_impl((const double &x) const {
    return 1. + 3. * x; }
};

auto complicated_covariance_function() {

  ScalingTerm<LinearScalar> linear_scalar;
  Constant constant;
  SquaredExponential<EuclideanDistance> squared_exp;

  return (constant + squared_exp) * linear_scalar;
}
}

int main() {

  auto cov = albatross::complicated_covariance_function();

  double x = 1.;
  double y = 0.;
  cov.call_trace().print(x, y);
}
