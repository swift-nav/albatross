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

#ifndef ALBATROSS_STATS_CHI_SQUARED_HPP_
#define ALBATROSS_STATS_CHI_SQUARED_HPP_

/*
 * Methods for computing quantities related to the Chi-Squared distribution
 *
 * https://en.wikipedia.org/wiki/Chi-squared_distribution
 *
 * Based largely off the implementation in the Stats C++ package:
 *
 * https://github.com/kthohr/stats/blob/master/include/stats_incl/prob/pchisq.ipp
 */
namespace albatross {

namespace details {

inline double chi_squared_cdf_unsafe(double x, std::size_t degrees_of_freedom) {
  return incomplete_gamma(0.5 * degrees_of_freedom, 0.5 * x);
}

inline double chi_squared_cdf_safe(double x, double degrees_of_freedom) {

  if (std::isnan(x) || x < 0.) {
    return NAN;
  }

  if (degrees_of_freedom == 0) {
    return 1.;
  }

  if (std::numeric_limits<double>::epsilon() > x) {
    return 0.;
  }

  if (std::isinf(x)) {
    return 1.;
  }

  if (std::isinf(degrees_of_freedom)) {
    return 0.;
  }

  return chi_squared_cdf_unsafe(x, degrees_of_freedom);
}

} // namespace details

inline double chi_squared_cdf(double x, double degrees_of_freedom) {
  return details::chi_squared_cdf_safe(x, degrees_of_freedom);
}

inline double chi_squared_cdf(const Eigen::VectorXd &deviation,
                              const Eigen::MatrixXd &covariance) {
  const double distance_squared =
      covariance.llt().matrixL().solve(deviation).squaredNorm();
  std::size_t n = static_cast<std::size_t>(deviation.size());
  return chi_squared_cdf(distance_squared, n);
}

} // namespace albatross

#endif /* ALBATROSS_STATS_CHI_SQUARED_HPP_ */
