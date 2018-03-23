/*
 * Copyright (C) 2017 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "tune.h"
#include <libswiftnav/pvt_engine/ranges/zip.h>

namespace albatross {

std::vector<ParameterValue> transform_parameters(
    const std::vector<ParameterValue>& x) {
  std::vector<ParameterValue> transformed(x.size());
  for (const auto& pair : pvt_engine::ranges::zip(x, transformed)) {
    std::get<1>(pair) = log(std::get<0>(pair));
  }
  return transformed;
}

std::vector<ParameterValue> inverse_parameters(
    const std::vector<ParameterValue>& x) {
  std::vector<ParameterValue> inverted(x.size());
  for (const auto& pair : pvt_engine::ranges::zip(x, inverted)) {
    std::get<1>(pair) = exp(std::get<0>(pair));
  }
  return inverted;
}
}
