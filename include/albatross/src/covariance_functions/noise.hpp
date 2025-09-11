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

constexpr double default_sigma_noise = 0.1;

namespace albatross {

template <typename Observed>
class IndependentNoise : public CovarianceFunction<IndependentNoise<Observed>> {
 public:
  IndependentNoise(double sigma_noise = 0.1) {
    sigma_independent_noise = {sigma_noise, PositivePrior()};
  }

  ALBATROSS_DECLARE_PARAMS(sigma_independent_noise)

  ~IndependentNoise() {}

  std::string name() const { return "independent_noise"; }

  /*
   * This will create a scaled identity matrix, but only between
   * two different observations defined by the Observed type.
   */
  double _call_impl(const Observed &x, const Observed &y) const {
    if (x == y) {
      return sigma_independent_noise.value * sigma_independent_noise.value;
    } else {
      return 0.;
    }
  }
};

}  // namespace albatross

#endif
