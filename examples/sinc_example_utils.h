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

#ifndef ALBATROSS_SINC_EXAMPLE_UTILS_H
#define ALBATROSS_SINC_EXAMPLE_UTILS_H

#include <albatross/GP>

#include "example_utils.h"

namespace albatross {

class SlopeTerm : public CovarianceFunction<SlopeTerm> {
public:
  SlopeTerm(double sigma_slope = 0.1) {
    this->params_["sigma_slope"] = sigma_slope;
  };

  ~SlopeTerm(){};

  std::string get_name() const { return "slope_term"; }

  double _call_impl(const double &x, const double &y) const {
    double sigma_slope = this->params_.at("sigma_slope").value;
    return sigma_slope * sigma_slope * x * y;
  }
};
} // namespace albatross

#endif
