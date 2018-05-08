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

#ifndef ALBATROSS_COVARIANCE_FUNCTIONS_NOISE_H
#define ALBATROSS_COVARIANCE_FUNCTIONS_NOISE_H

#include "covariance_term.h"

namespace albatross {

template <typename Observed> class IndependentNoise : public CovarianceTerm {
public:
  IndependentNoise(double sigma_noise = 0.1) {
    this->params_["sigma_independent_noise"] = sigma_noise;
  };

  ~IndependentNoise(){};

  std::string get_name() const { return "independent_noise"; }

  /*
   * This will create a scaled identity matrix, but only between
   * two different observations defined by the Observed type.
   */
  double operator()(const Observed &x, const Observed &y) const {
    if (x == y) {
      double sigma_noise = this->params_.at("sigma_independent_noise");
      return sigma_noise * sigma_noise;
    } else {
      return 0.;
    }
  }
};
}

#endif
