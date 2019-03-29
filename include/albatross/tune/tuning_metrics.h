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

#ifndef ALBATROSS_TUNING_METRICS_H
#define ALBATROSS_TUNING_METRICS_H

namespace albatross {

using TuningMetricAggregator =double (*)(const std::vector<double> &);

/*
 * Returns the mean of metrics computed across multiple datasets.
 */
inline double mean_aggregator(const std::vector<double> &metrics) {
  double mean = 0.;
  for (const auto &metric : metrics) {
    mean += metric;
  }
  mean /= static_cast<double>(metrics.size());
  return mean;
}

} // namespace albatross
#endif
