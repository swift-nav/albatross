/*
 * Copyright (C) 2022 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_COVARIANCE_FUNCTIONS_NUGGET_H
#define ALBATROSS_COVARIANCE_FUNCTIONS_NUGGET_H

constexpr double default_nugget_noise = 1e-8;

namespace albatross {

/*
 * A nugget applies a small amount of noise to all feature types
 * with the intention of preserving positive definite matrices
 * even in the case where some of the features involved are
 * perfectly correlated. This difference slightly from IndependentNoise
 * which applies noise only to a specific FeatureType.  Here
 * we apply it to any "basic" FeatureType (where basic is meant
 * to mean we don't apply it to composite types (such as LinearCombination)
 * which should instead have a nugget applied to each of the
 * sub features which make up the composite.
 */

class Nugget : public CovarianceFunction<Nugget> {
public:
  ALBATROSS_DECLARE_PARAMS(nugget_sigma);

  std::string get_name() const { return "nugget"; }

  Nugget() { nugget_sigma = {default_nugget_noise, FixedPrior()}; };

  template <typename X,
            typename std::enable_if_t<is_basic_type<X>::value, int> = 0>
  double _call_impl(const X &x, const X &y) const {
    if (x == y) {
      return nugget_sigma.value * nugget_sigma.value;
    } else {
      return 0.;
    }
  }
};

} // namespace albatross

#endif
