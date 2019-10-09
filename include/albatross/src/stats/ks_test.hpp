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

#ifndef ALBATROSS_STATS_KS_TEST_HPP_
#define ALBATROSS_STATS_KS_TEST_HPP_

namespace albatross {

/*
 * Computes the two-sided Kolmogorov-Smirnov test to determine
 * if the samples came from a uniform distribution.
 */
inline double uniform_ks_test(const std::vector<double> &samples) {
  double largest_difference_between_sample_and_expected_cdf = 0.;
  std::vector<double> sorted(samples);
  std::sort(sorted.begin(), sorted.end());
  double n = static_cast<double>(sorted.size());
  for (std::size_t i = 0; i < sorted.size(); ++i) {

    double difference = std::max((i + 1) / n - sorted[i], sorted[i] - i / n);
    if (difference > largest_difference_between_sample_and_expected_cdf) {
      largest_difference_between_sample_and_expected_cdf = difference;
    }
  }
  return largest_difference_between_sample_and_expected_cdf;
}

} // namespace albatross

#endif /* ALBATROSS_STATS_KS_TEST_HPP_ */
