/*
 * Copyright (C) 2019 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef ALBATROSS_SAMPLERS_STATE_HPP_
#define ALBATROSS_SAMPLERS_STATE_HPP_

namespace albatross {

struct SamplerState {
  std::vector<double> params;
  double log_prob;
  bool accepted;
};

} // namespace albatross

#endif /* ALBATROSS_SAMPLERS_STATE_HPP_ */
