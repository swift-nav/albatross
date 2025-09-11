/*
 * Copyright (C) 2021 Swift Navigation Inc.
 * Contact: Swift Navigation <dev@swiftnav.com>
 *
 * This source is subject to the license found in the file 'LICENSE' which must
 * be distributed together with this source. All other rights reserved.
 *
 * THIS CODE AND INFORMATION IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND,
 * EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef ALBATROSS_STATS_GAUSSIAN_HPP_
#define ALBATROSS_STATS_GAUSSIAN_HPP_

namespace albatross {

namespace gaussian {

static inline double log_pdf(double deviation, double variance) {
  double ll = -deviation * deviation / (2 * variance);
  ll -= 0.5 * std::log(2 * M_PI * variance);
  return ll;
}

static inline double pdf(double deviation, double variance) {
  return std::exp(-(deviation * deviation) / (2 * variance)) /
         std::sqrt(2 * M_PI * variance);
}

} // namespace gaussian

} // namespace albatross

#endif /* ALBATROSS_STATS_GAUSSIAN_HPP_ */
