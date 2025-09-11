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

class SincFunction : public MeanFunction<SincFunction> {

public:
  ALBATROSS_DECLARE_PARAMS(scale, translation)

  std::string get_name() const { return "sinc"; }

  SincFunction() {
    scale = {EXAMPLE_SCALE_VALUE, GaussianPrior(1., 1000.)};
    translation = {EXAMPLE_TRANSLATION_VALUE, GaussianPrior(0., 1000.)};
  }

  double _call_impl(const double &x) const {
    return scale.value * sinc(x - translation.value);
  }
};

} // namespace albatross

#endif
