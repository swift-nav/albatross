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

#ifndef ALBATROSS_STATS_INCOMPLETE_GAMMA_HPP_
#define ALBATROSS_STATS_INCOMPLETE_GAMMA_HPP_

/*
 * The following methods use either Gaussian Quadrature or the power series
 * expansion to integrate the incomplete gamma function
 *
 * https://en.wikipedia.org/wiki/Gaussian_quadrature
 * https://en.wikipedia.org/wiki/Incomplete_gamma_function#Holomorphic_extension
 *
 * Many thanks to the GCEM package from which the majority of this was
 * derived.
 *
 * https://github.com/kthohr/gcem/blob/master/include/gcem_incl/incomplete_gamma.hpp
 */

namespace albatross {

namespace details {

constexpr std::size_t INCOMPLETE_GAMMA_RECURSSION_LIMIT = 54;
constexpr double INCOMPLETE_GAMMA_EQUALITY_TRESHOLD = 1e-12;

inline double incomplete_gamma_quadrature_inp_vals(double lb, double ub,
                                                   std::size_t counter) {
  assert(counter < gauss_legendre_50_points.size());
  return (ub - lb) * 0.5 * gauss_legendre_50_points[counter] + 0.5 * (ub + lb);
}

inline double incomplete_gamma_quadrature_weight_vals(double lb, double ub,
                                                      std::size_t counter) {
  assert(counter < gauss_legendre_50_weights.size());
  return (ub - lb) * 0.5 * gauss_legendre_50_weights[counter];
}

inline double incomplete_gamma_quadrature_fn(double x, double a,
                                             double large_term) {
  return exp(-x + (a - 1.) * log(x) - large_term);
}

inline double incomplete_gamma_quadrature_recursive(double lb, double ub,
                                                    double a, double lg_term,
                                                    std::size_t counter) {
  if (counter < gauss_legendre_50_weights.size() - 1) {
    return incomplete_gamma_quadrature_fn(
               incomplete_gamma_quadrature_inp_vals(lb, ub, counter), a,
               lg_term) *
               incomplete_gamma_quadrature_weight_vals(lb, ub, counter) +
           incomplete_gamma_quadrature_recursive(lb, ub, a, lg_term,
                                                 counter + 1);
  } else {
    return incomplete_gamma_quadrature_fn(
               incomplete_gamma_quadrature_inp_vals(lb, ub, counter), a,
               lg_term) *
           incomplete_gamma_quadrature_weight_vals(lb, ub, counter);
  }
}

inline std::pair<double, double> incomplete_gamma_quadrature_bounds(double a,
                                                                    double z) {

  if (a > 800) {
    return std::make_pair(std::max(0., std::min(z, a) - 11 * sqrt(a)),
                          std::min(z, a + 10 * sqrt(a)));
  } else if (a > 300) {
    return std::make_pair(std::max(0., std::min(z, a) - 10 * sqrt(a)),
                          std::min(z, a + 9 * sqrt(a)));
  } else if (a > 90) {
    return std::make_pair(std::max(0., std::min(z, a) - 9 * sqrt(a)),
                          std::min(z, a + 8 * sqrt(a)));
  } else if (a > 70) {
    return std::make_pair(std::max(0., std::min(z, a) - 8 * sqrt(a)),
                          std::min(z, a + 7 * sqrt(a)));
  } else if (a > 50) {
    return std::make_pair(std::max(0., std::min(z, a) - 7 * sqrt(a)),
                          std::min(z, a + 6 * sqrt(a)));
  } else if (a > 40) {
    return std::make_pair(std::max(0., std::min(z, a) - 6 * sqrt(a)),
                          std::min(z, a + 5 * sqrt(a)));
  } else if (a > 30) {
    return std::make_pair(std::max(0., std::min(z, a) - 5 * sqrt(a)),
                          std::min(z, a + 4 * sqrt(a)));
  } else {
    return std::make_pair(std::max(0., std::min(z, a) - 4 * sqrt(a)),
                          std::min(z, a + 4 * sqrt(a)));
  }
}

inline double incomplete_gamma_quadrature(double a, double z) {
  const auto bounds = incomplete_gamma_quadrature_bounds(a, z);
  return incomplete_gamma_quadrature_recursive(bounds.first, bounds.second, a,
                                               lgamma(a), 0);
}

inline double
incomplete_gamma_continuous_fraction_numerator(double a, double z,
                                               std::size_t depth) {
  if (depth % 2 == 0) {
    return 0.5 * depth * z;
  } else {
    return -(a - 1 + 0.5 * (depth + 1)) * z;
  }
}

inline double incomplete_gamma_continuous_fraction(double a, double z,
                                                   std::size_t depth) {
  if (depth > INCOMPLETE_GAMMA_RECURSSION_LIMIT) {
    return a + depth - 1;
  } else {
    double numerator =
        incomplete_gamma_continuous_fraction_numerator(a, z, depth);
    double denominator = incomplete_gamma_continuous_fraction(a, z, depth + 1);
    return (a + depth - 1) + numerator / denominator;
  }
}

inline double incomplete_gamma_continuous_fraction(double a, double z) {
  double numerator = exp(a * log(z) - z) / tgamma(a);
  // the denominator can't be any smaller than the numerator since this
  // quantity is bounded by 0 and 1.  With numerical round off when evaluating
  // this function for really large a the continuous fraction sometimes dips
  // below the numerator (and sometimes even below zero).
  double denominator =
      std::max(numerator, incomplete_gamma_continuous_fraction(a, z, 1));
  // When `a` get's really really large the numerator (and in turn the
  // denominator) can hit zero which would turn into a NAN but we want
  // to treat it as evaluating at infinity which should yield 1.
  if (denominator - numerator < INCOMPLETE_GAMMA_EQUALITY_TRESHOLD) {
    return 1.;
  }
  return numerator / denominator;
}

} // namespace details

inline double incomplete_gamma(double a, double z) {
  if (std::isnan(a) || std::isnan(z)) {
    return NAN;
  }

  if (a < 0.) {
    return NAN;
  }

  if (std::numeric_limits<decltype(z)>::epsilon() > z) {
    return 0.;
  }

  if (std::numeric_limits<decltype(a)>::epsilon() > a) {
    return 1.;
  }

  if (a < 10) {
    return details::incomplete_gamma_continuous_fraction(a, z);
  } else {
    return details::incomplete_gamma_quadrature(a, z);
  }
}

} // namespace albatross
#endif /* INCLUDE_ALBATROSS_SRC_STATS_INCOMPLETE_GAMMA_HPP_ */
